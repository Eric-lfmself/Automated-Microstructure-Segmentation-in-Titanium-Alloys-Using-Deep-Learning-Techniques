"""Configuration validation and bounded smoke settings."""
from dataclasses import replace
import json
import pytest
from configs import Config, load_config


def test_default_is_bounded_offline_cpu():
    c = load_config()
    assert c.device == "cpu" and c.dry_run and c.use_random_init
    assert c.image_size == (64,64) and c.source_image_size == (1024,1024)


@pytest.mark.parametrize("change", [dict(device="cuda"), dict(allow_download=True), dict(batch_size=5), dict(image_size=(1024,1024)), dict(epochs=3), dict(max_batches=None), dict(class_weights=(1,1,1,1,0)), dict(sigmas=(1,1,3,4)), dict(dry_run="false")])
def test_reject_invalid_or_unsafe_config(change):
    with pytest.raises(ValueError):
        replace(Config(),**change).validate()


def test_dry_run_overrides_real_template():
    c = load_config("configs/real_template.json").as_dry_run()
    assert c.use_random_init and not c.allow_download and c.max_batches == 1


def test_unknown_key_fails(tmp_path):
    p=tmp_path/"bad.json";p.write_text(json.dumps({"typo":1}))
    with pytest.raises(TypeError): load_config(p)


@pytest.mark.parametrize('change', [
    {'seed': -1}, {'seed': True}, {'seed': 2**32-1}, {'num_classes': 5.0},
    {'num_workers': False}, {'label_offset': True}, {'dropout': False},
    {'learning_rate': True}, {'class_weights': (True, 1, 1, 1, 1)},
    {'output_dir': ''}, {'checkpoint': None}, {'image_size': None}, {'sigmas': None},
])
def test_malformed_config_fails_at_shared_boundary(change):
    with pytest.raises(ValueError):
        replace(Config(), **change).validate()


@pytest.mark.parametrize('document', ['[]', '{"sigmas": null}'])
def test_json_config_requires_an_object_and_array_fields(tmp_path, document):
    path = tmp_path / 'bad.json'; path.write_text(document)
    with pytest.raises(ValueError):
        load_config(path)
