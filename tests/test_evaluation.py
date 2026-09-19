"""Tests distinguish measured metrics, dummy provenance and held-out evaluation."""
import pytest
import torch
from torch import nn
from eval.runner import evaluate_loader
from scripts.compare import comparison
from scripts.evaluate import parse_config


def test_evaluator_global_metrics_and_mode_restore():
    model=nn.Conv2d(3,5,1).to('cpu')
    with torch.no_grad(): model.weight.zero_();model.bias.zero_();model.bias[0]=2
    data=[{'image':torch.zeros(2,3,64,64,device='cpu'),
           'mask':torch.zeros(2,64,64,dtype=torch.long,device='cpu'),'id':['a','b']}]
    metrics,batch=evaluate_loader(model,data,max_batches=1)
    assert model.training and metrics['pixel_accuracy']==1.0 and metrics['miou']==1.0
    assert metrics['image_count']==2 and batch['prediction'].shape==(2,64,64)


def test_comparison_rejects_different_samples_and_duplicate_models():
    base=dict(model='unet_vgg',mode='dummy_validation_only',dataset_signature='abc',sample_ids=['a'])
    with pytest.raises(ValueError,match='sample_ids'):
        comparison([base,dict(base,model='segformer_b0',sample_ids=['b'])])
    with pytest.raises(ValueError,match='Duplicate'):
        comparison([base,base])


def test_real_evaluation_requires_opt_in_and_checkpoint():
    with pytest.raises(SystemExit):parse_config(['--config','configs/real_template.json'])
    with pytest.raises(SystemExit):parse_config(['--config','configs/real_template.json','--run-real'])
    with pytest.raises(SystemExit):parse_config(['--run-real'])


def test_comparison_cannot_overwrite_its_json_source(tmp_path):
    from scripts.compare import main
    path=tmp_path/'metrics.json'
    path.write_text('{}')
    with pytest.raises(SystemExit): main([str(path),str(path),'--output',str(path)])
    assert path.read_text()=='{}'


def _small_batch():
    return {'image': torch.zeros(1, 3, 2, 2, device='cpu'),
            'mask': torch.zeros(1, 2, 2, dtype=torch.long, device='cpu'), 'id': ['sample']}


@pytest.mark.parametrize('mask,error', [
    (torch.full((1, 2, 2), .9), TypeError), (torch.zeros(1, 2, 2, dtype=torch.bool), TypeError),
    (torch.full((1, 2, 2), 5), ValueError), (torch.full((1, 2, 2), -1), ValueError),
    (torch.full((1, 2, 2), -1, dtype=torch.int8), ValueError),
    (torch.zeros(1, 3, 2, dtype=torch.long), ValueError),
])
def test_invalid_mask_rejected_before_forward(mask, error):
    class Forbidden(nn.Module):
        def forward(self, images):
            raise AssertionError('Invalid labels must fail before forward')
    batch = dict(_small_batch(), mask=mask)
    with pytest.raises(error, match='mask'):
        evaluate_loader(Forbidden(), [batch])


@pytest.mark.parametrize('raises', [False, True])
def test_evaluator_restores_mixed_module_modes_on_all_paths(raises):
    class Head(nn.Module):
        def forward(self, images):
            if raises:
                raise RuntimeError('forward failed')
            return torch.zeros(images.shape[0], 5, *images.shape[-2:], device='cpu')
    model = nn.Sequential(nn.BatchNorm2d(3), Head()).to('cpu')
    model.train(); model[0].eval()
    modes = [module.training for module in model.modules()]
    if raises:
        with pytest.raises(RuntimeError, match='forward failed'):
            evaluate_loader(model, [_small_batch()])
    else:
        evaluate_loader(model, [_small_batch()])
    assert [module.training for module in model.modules()] == modes


def test_evaluator_rejects_non_cpu_buffer_without_using_gpu():
    model = nn.Identity()
    model.register_buffer('placeholder', torch.empty(0, device='meta'))
    with pytest.raises(ValueError, match='CPU only'):
        evaluate_loader(model, [_small_batch()])


@pytest.mark.parametrize('limit', [True, 1.5, 0])
def test_evaluator_requires_integer_limit(limit):
    with pytest.raises(ValueError, match='positive integer'):
        evaluate_loader(nn.Identity(), [], max_batches=limit)


@pytest.mark.parametrize('ids', ['sample', [], ['a', 'b'], [7]])
def test_evaluator_rejects_misaligned_sample_ids_before_forward(ids):
    with pytest.raises(ValueError, match='one ID per image'):
        evaluate_loader(nn.Identity(), [dict(_small_batch(), id=ids)])


def test_evaluator_accepts_valid_narrow_integer_labels():
    class Constant(nn.Module):
        def forward(self, images):
            return torch.zeros(1, 5, 2, 2, device='cpu')
    batch = dict(_small_batch(), mask=torch.zeros(1, 2, 2, dtype=torch.int8))
    metrics, example = evaluate_loader(Constant(), [batch])
    assert metrics['pixel_accuracy'] == 1.0 and example['mask'].dtype == torch.long



def test_float64_overflow_is_rejected_before_evaluation_forward():
    class Forbidden(nn.Module):
        def forward(self, images):
            raise AssertionError('Cannot forward float32 overflow')
    batch = dict(_small_batch(), image=torch.full((1, 3, 2, 2), 1e300, dtype=torch.float64))
    with pytest.raises(ValueError, match='finite float32'):
        evaluate_loader(Forbidden(), [batch])


def test_real_evaluation_rejects_untrained_checkpoint_before_model(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from configs import Config
    from scripts import evaluate
    from training.checkpoint import save_checkpoint
    config = replace(Config(), dry_run=False, test_manifest='unused.csv')
    model = nn.Conv2d(3, 5, 1).to('cpu')
    path = tmp_path / 'zero_epoch.pt'
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), config=config,
                    completed_epochs=0, history=[])
    output = tmp_path / 'evaluation'
    monkeypatch.setattr(evaluate, 'parse_config', lambda argv: (SimpleNamespace(checkpoint=path), config, output))
    monkeypatch.setattr(evaluate, 'validate_split_manifests', lambda cfg: {'test': object()})
    monkeypatch.setattr(evaluate, 'collect_provenance', lambda datasets: {'test': []})
    def forbidden(*args, **kwargs):
        raise AssertionError('Untrained checkpoint must fail before model construction')
    monkeypatch.setattr(evaluate, 'create_model', forbidden)
    with pytest.raises(ValueError, match='at least one completed epoch'):
        evaluate.main([])
    assert not output.exists()


def test_dummy_evaluation_uses_validation_stream_and_records_its_identity(tmp_path, monkeypatch):
    from dataclasses import replace
    import hashlib
    import json
    from configs import Config
    from data import DummyTC4Dataset
    from scripts import evaluate
    config = replace(Config(), image_size=(32, 32), batch_size=1, dummy_length=1)
    validation = DummyTC4Dataset(length=1, image_size=config.image_size, seed=config.seed + 1)[0]
    training = DummyTC4Dataset(length=1, image_size=config.image_size, seed=config.seed)[0]
    assert not torch.equal(validation['mask'], training['mask'])
    class ExpectedValidation(nn.Module):
        def forward(self, images):
            torch.testing.assert_close(images[0], validation['image'], rtol=0, atol=0)
            return torch.nn.functional.one_hot(validation['mask'], 5).permute(2, 0, 1)[None].float()
    monkeypatch.setattr(evaluate, 'create_model', lambda *a, **k: ExpectedValidation())
    monkeypatch.setattr(evaluate, 'benchmark_model_cpu', lambda *a, **k: {'device': 'cpu'})
    monkeypatch.setattr(evaluate, 'preview', lambda *a, **k: None)
    path = tmp_path / 'config.json'; path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / 'evaluation'
    evaluate.main(['--config', str(path), '--output-dir', str(output)])
    result = json.loads((output / 'metrics.json').read_text())
    assert result['pixel_accuracy'] == 1 and result['mode'] == 'dummy_validation_only'
    identity = dict(kind='dummy', seed=config.seed + 1, length=config.dummy_length,
                    image_size=config.image_size, sigmas=config.sigmas, weights=config.iw_weights)
    assert result['dataset_signature'] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("kind", ["dummy", "real"])
def test_dataset_signature_normalizes_equivalent_numeric_parameters(kind):
    from scripts.evaluate import _dataset_signature
    base = {"kind": kind, "sigmas": (1.0, 2.0, 3.0, 4.0), "weights": (0.0, 1.0, .5, .5),
            "image_size": (32, 32), "records": [{"id": "x", "image_sha256": "a" * 64, "mask_sha256": "b" * 64}]}
    alternate = dict(base, sigmas=[1, 2, 3, 4], weights=[-0.0, 1, .5, .5])
    assert _dataset_signature(base) == _dataset_signature(alternate)
    for changed in [dict(base, sigmas=(1, 2, 3, 5)), dict(base, weights=(0, 1, .5, .6)),
                    dict(base, image_size=(64, 64)), dict(base, records=[{"id": "different"}])]:
        assert _dataset_signature(changed) != _dataset_signature(base)
