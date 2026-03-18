from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "src" / "nunchaku_torch"


def test_runtime_source_has_no_vendor_or_shim_imports():
    offenders: list[str] = []
    for path in RUNTIME_ROOT.rglob("*.py"):
        text = path.read_text()
        if "nunchaku_zimage_torch" in text or "_vendor" in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []
