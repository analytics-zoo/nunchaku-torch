from importlib import import_module


_EXPORTS = {
    "NunchakuZImageTransformer2DModel": (
        "nunchaku_torch.models.transformers.transformer_zimage",
        "NunchakuZImageTransformer2DModel",
    ),
    "NunchakuQwenImageTransformer2DModel": (
        "nunchaku_torch.models.transformers.transformer_qwenimage",
        "NunchakuQwenImageTransformer2DModel",
    ),
}

__all__ = [
    *sorted(_EXPORTS.keys()),
    "transformer_qwenimage",
    "transformer_zimage",
]


def __getattr__(name: str):
    if name == "transformer_zimage":
        return import_module("nunchaku_torch.models.transformers.transformer_zimage")
    if name == "transformer_qwenimage":
        return import_module(
            "nunchaku_torch.models.transformers.transformer_qwenimage"
        )
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(name)
