"""Config loader - loads and validates YAML config files."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from cvf.core.contracts.config.main import CVFConfig


def load_config(config_path: str) -> CVFConfig:
    """
    Load and validate a CVF config from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Validated CVFConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    
    return CVFConfig(**data)


def load_config_dict(data: Dict[str, Any]) -> CVFConfig:
    """
    Load and validate a CVF config from a dictionary.
    
    Args:
        data: Config dictionary
        
    Returns:
        Validated CVFConfig object
    """
    return CVFConfig(**data)


def save_config(config: CVFConfig, output_path: str) -> None:
    """
    Save a CVF config to YAML file.
    
    Args:
        config: CVFConfig object
        output_path: Path to save YAML file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def merge_configs(base: CVFConfig, override: Dict[str, Any]) -> CVFConfig:
    """
    Merge an override dict into a base config.
    
    Args:
        base: Base CVFConfig
        override: Override dictionary
        
    Returns:
        New merged CVFConfig
    """
    base_dict = base.model_dump()
    _deep_merge(base_dict, override)
    return CVFConfig(**base_dict)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value