from importlib import import_module


_EXPORTS = {
    "NunchakuZImageTransformer2DModel": (
        "nunchaku_torch.models.transformers",
        "NunchakuZImageTransformer2DModel",
    ),
    "NunchakuQwenImageTransformer2DModel": (
        "nunchaku_torch.models.transformers",
        "NunchakuQwenImageTransformer2DModel",
    ),
    "SVDQW4A4Linear": ("nunchaku_torch.models.linear", "SVDQW4A4Linear"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(name)
