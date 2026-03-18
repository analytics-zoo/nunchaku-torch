import pytest
import torch

from nunchaku_torch.device import resolve_device


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_invalid():
    with pytest.raises(ValueError):
        resolve_device("bad-device")


def test_resolve_device_auto_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
