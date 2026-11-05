import pytest

from sentiment.config import ExperimentConfig, load_experiments

from .conftest import ROOT


def test_repo_experiment_file_loads():
    cfgs = load_experiments(ROOT / "configs" / "experiments.yaml")
    names = [c.name for c in cfgs]
    assert len(names) == len(set(names)) >= 6
    repro = next(c for c in cfgs if c.stage == "reproduce")
    # the reproduction run must match the original notebook recipe
    assert (repro.model_name, repro.learning_rate, repro.num_train_epochs,
            repro.per_device_train_batch_size, repro.max_length) == (
        "distilbert/distilroberta-base", 2e-5, 2, 16, 128)
    assert all(c.hypothesis for c in cfgs)


def test_defaults_merge_and_override(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("defaults: {learning_rate: 3.0e-5, seed: 7}\nexperiments:\n"
                 "  - {name: a, model_name: m}\n  - {name: b, model_name: m, seed: 1}\n")
    a, b = load_experiments(p)
    assert a.learning_rate == 3e-5 and a.seed == 7 and b.seed == 1
    assert a.effective_batch_size == 32


@pytest.mark.parametrize("body,msg", [
    ("experiments:\n  - {name: a, model_name: m, learnin_rate: 1.0e-5}\n", "unknown"),
    ("experiments:\n  - {name: a, model_name: m}\n  - {name: a, model_name: m}\n", "duplicate"),
    ("foo: 1\n", "experiments"),
])
def test_bad_files(tmp_path, body, msg):
    p = tmp_path / "e.yaml"
    p.write_text(body)
    with pytest.raises(ValueError, match=msg):
        load_experiments(p)


@pytest.mark.parametrize("kw", [
    {"learning_rate": 0.5}, {"max_length": 4}, {"num_train_epochs": 0}, {"name": "a/b"},
    {"dropout": 1.5}, {"warmup_ratio": 1.0},
])
def test_value_validation(kw):
    with pytest.raises(ValueError):
        ExperimentConfig(**{"name": "x", "model_name": "m", **kw})
