"""Installed command defaults are independent of repository-relative files."""
from dataclasses import replace
import json
import sys
from pathlib import Path
from configs import Config
from scripts import check_config, evaluate


def test_evaluate_default_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, config, output = evaluate.parse_config([])
    assert config == Config().validate()
    assert output == Path("runs/eval_dummy_unet_vgg")
    assert not list(tmp_path.iterdir())


def test_check_config_default_from_another_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_config"])
    check_config.main()
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(Config().to_dict()))


def test_explicit_config_still_resolves_from_caller_directory(tmp_path, monkeypatch, capsys):
    expected = replace(Config(), seed=7, model="segformer_b0", image_size=(32, 32))
    (tmp_path / "local.json").write_text(json.dumps(expected.to_dict()))
    monkeypatch.chdir(tmp_path)
    _, actual, _ = evaluate.parse_config(["--config", "local.json"])
    assert actual == expected
    monkeypatch.setattr(sys, "argv", ["check_config", "--config", "local.json"])
    check_config.main()
    assert json.loads(capsys.readouterr().out)["seed"] == 7
