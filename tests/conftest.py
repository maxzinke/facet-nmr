"""Shared pytest configuration.

The model weights and the retrieval index are not part of the wheel -- they are
downloaded on first use (see ``facet/assets.py``). Most tests exercise readers,
writers and validation and never touch them. The few that run the model end to end
are marked ``needs_assets`` and are skipped when the required files are not already
on disk, so the suite is green on a fresh checkout, in CI without network, and from a
wheel install alike. Nothing here ever triggers a download.
"""
from __future__ import annotations

import os

import pytest

_REQUIRED_ASSETS = ("facet_v3.pt", "facet_retrieval_index.npz",
                    "facet_retrieval_index.entries.json")


def assets_available() -> bool:
    """True when every required (non-optional) asset resolves without downloading."""
    from facet import assets

    for name in _REQUIRED_ASSETS:
        try:
            if assets.resolve(name, download=False) is None:
                return False
        except assets.AssetUnavailable:
            return False
    return True


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_assets: runs the model end to end; skipped unless the weights and "
        "retrieval index are already on disk (never downloads).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if any(item.get_closest_marker("needs_assets") for item in items) and not assets_available():
        skip = pytest.mark.skip(
            reason="model weights / retrieval index not on disk "
                   "(run `python -m facet.assets` to fetch them)",
        )
        for item in items:
            if item.get_closest_marker("needs_assets"):
                item.add_marker(skip)


@pytest.fixture
def no_download(monkeypatch: pytest.MonkeyPatch):
    """Guarantee a test cannot reach the network for assets."""
    monkeypatch.setenv("FACET_NO_DOWNLOAD", "1")
    yield
