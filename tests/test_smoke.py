"""Smoke test — verifies the package imports and exposes its version."""

import alp


def test_version_present() -> None:
    assert isinstance(alp.__version__, str)
    assert alp.__version__.count(".") >= 1
