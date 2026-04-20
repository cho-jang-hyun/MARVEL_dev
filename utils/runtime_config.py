"""Runtime parameter loader shared by train and evaluation code paths.

Set `MARVEL_CONFIG_MODE=test` to source constants from `test_parameter.py`.
If unset, the default mode is `train` and constants come from `parameter.py`.
"""

import importlib
import os


CONFIG_MODE_ENV = 'MARVEL_CONFIG_MODE'
_CONFIG_MODULE_BY_MODE = {
    'train': 'parameter',
    'test': 'test_parameter',
}


def _apply_test_defaults(config_module):
    default_budget = float(getattr(config_module, 'BUDGET', 1.0))
    sensor_range = float(getattr(config_module, 'SENSOR_RANGE', 10.0))
    defaults = {
        'SUCCESS_THRESHOLD': 0.99,
        'NUM_EPISODE_BUFFER': 54,
        'MAX_BUDGET': default_budget,
        'UTILITY_RANGE': 0.9 * sensor_range,
        'VISITED_BY_OTHERS_DECAY': 0.025,
        'VISITED_BY_OTHERS_MIN': 0.05,
    }
    for name, value in defaults.items():
        if not hasattr(config_module, name):
            setattr(config_module, name, value)


def _load_config_module():
    mode = os.environ.get(CONFIG_MODE_ENV, 'train').strip().lower()
    module_name = _CONFIG_MODULE_BY_MODE.get(mode)
    if module_name is None:
        valid_modes = ', '.join(sorted(_CONFIG_MODULE_BY_MODE))
        raise ValueError(
            f"Unsupported {CONFIG_MODE_ENV}='{mode}'. "
            f"Expected one of: {valid_modes}."
        )

    config_module = importlib.import_module(module_name)
    if mode == 'test':
        _apply_test_defaults(config_module)
    return mode, config_module


CONFIG_MODE, _CONFIG_MODULE = _load_config_module()

for _name in dir(_CONFIG_MODULE):
    if _name.startswith('_'):
        continue
    globals()[_name] = getattr(_CONFIG_MODULE, _name)

__all__ = [name for name in dir(_CONFIG_MODULE) if not name.startswith('_')]


def get_config_mode():
    return CONFIG_MODE
