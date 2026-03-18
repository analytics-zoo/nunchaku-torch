from importlib import import_module

from .__version__ import __version__


_EXPORTS = {
    "GenerationConfig": ("nunchaku_torch.zimage", "GenerationConfig"),
    "generate_image": ("nunchaku_torch.zimage", "generate_image"),
    "save_image": ("nunchaku_torch.zimage", "save_image"),
    "resolve_device": ("nunchaku_torch.device", "resolve_device"),
    "default_dtype": ("nunchaku_torch.device", "default_dtype"),
    "NunchakuZImageTransformer2DModel": (
        "nunchaku_torch.models.transformers",
        "NunchakuZImageTransformer2DModel",
    ),
    "NunchakuQwenImageTransformer2DModel": (
        "nunchaku_torch.models.transformers",
        "NunchakuQwenImageTransformer2DModel",
    ),
}

__all__ = [*sorted(_EXPORTS.keys()), "__version__"]


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(name)
