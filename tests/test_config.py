"""Tests for config loading."""
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def test_config_exists_and_loads():
    """Config file exists and is valid YAML."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "config.yaml",
    )
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config is not None


def test_config_has_required_sections():
    """Config has required top-level keys for training."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "config.yaml",
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert "model" in config
    assert "training" in config
    assert "data" in config
    assert "paths" in config
    assert config["model"]["num_classes"] == 10
    assert config["training"]["batch_size"] > 0
    assert config["training"]["num_epochs"] > 0
