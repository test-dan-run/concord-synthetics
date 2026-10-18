import numpy as np
import pytest
from pydantic import ValidationError

from conftest import make_cfg
from synthetics.config.distributions import Bernoulli, Categorical, IntUniform, LogNormal, Uniform, Zipf
from synthetics.config.loader import config_hash, dump_run_config, read_run_config


def test_defaults_load_and_hash_is_stable(cfg):
    assert cfg.room_classes.ratio.probs == {"small": 0.3, "large": 0.7}
    assert config_hash(cfg) == config_hash(make_cfg())
    assert len(config_hash(cfg)) == 12


def test_dotted_override_changes_hash(cfg):
    other = make_cfg(**{"turn_taking.zipf.exponent": 1.2})
    assert other.turn_taking.zipf.exponent == 1.2
    assert config_hash(other) != config_hash(cfg)


def test_frozen_config_round_trips(cfg, tmp_path):
    dump_run_config(cfg, tmp_path / "config.yaml")
    again = read_run_config(tmp_path / "config.yaml")
    assert config_hash(again) == config_hash(cfg)
    # categorical keys come back as strings; counts are still usable
    assert {int(k) for k in again.seating.presenters.count.small.weights} == {1, 2}


def test_class_boundary_is_enforced():
    with pytest.raises(ValidationError):
        make_cfg(**{"room_classes.classes.small.speakers": {"dist": "int_uniform", "lo": 2, "hi": 8}})


def test_distributions_sample_in_range():
    rng = np.random.default_rng(0)
    assert all(3 <= Uniform(lo=3, hi=5).sample(rng) <= 5 for _ in range(100))
    assert {IntUniform(lo=1, hi=3).sample(rng) for _ in range(200)} == {1, 2, 3}
    assert all(LogNormal(median=0.3, sigma=0.6, cap=2.0).sample(rng) <= 2.0 for _ in range(500))
    xs = [Bernoulli(p=0.25).sample(rng) for _ in range(4000)]
    assert abs(np.mean(xs) - 0.25) < 0.03
    c = Categorical(weights={"a": 3, "b": 1})
    assert abs(np.mean([c.sample(rng) == "a" for _ in range(4000)]) - 0.75) < 0.03
    s = Zipf(exponent=1.0).shares(4)
    assert abs(sum(s) - 1) < 1e-9 and s == sorted(s, reverse=True)
