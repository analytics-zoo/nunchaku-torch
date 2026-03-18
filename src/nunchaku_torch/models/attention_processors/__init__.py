from importlib import import_module


_EXPORTS = {
    "NunchakuZSingleStreamAttnProcessor": (
        "nunchaku_torch.models.attention_processors.zimage",
        "NunchakuZSingleStreamAttnProcessor",
    ),
    "NunchakuQwenImageNaiveFA2Processor": (
        "nunchaku_torch.models.attention_processors.qwenimage",
        "NunchakuQwenImageNaiveFA2Processor",
    ),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(name)
