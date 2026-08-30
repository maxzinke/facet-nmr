"""Tests for facet/assets.py — the manifest and the resolution order.

The manifest (``ASSETS``) is the reproducibility claim of a release: it names the
exact files the README numbers were measured on. These tests check that whatever is
sitting in ``facet/weights/`` on this machine is those files, and that resolution
never reaches the network when told not to.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from facet import assets

WEIGHTS_DIR = Path(assets.__file__).resolve().parent / "weights"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TestManifest:
    def test_manifest_is_well_formed(self):
        assert assets.ASSETS, "manifest is empty"
        for name, asset in assets.ASSETS.items():
            assert len(asset.sha256) == 64 and int(asset.sha256, 16) >= 0, name
            assert asset.size > 0, name
            assert "/" not in name and "\\" not in name, name

    @pytest.mark.parametrize("name", sorted(assets.ASSETS))
    def test_bundled_file_matches_manifest(self, name):
        """A development checkout must carry exactly the validated artefacts."""
        path = WEIGHTS_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not bundled in this checkout")
        asset = assets.ASSETS[name]
        assert path.stat().st_size == asset.size, (
            f"{name}: size {path.stat().st_size} != manifest {asset.size}; "
            f"if the artefact was rebuilt on purpose, update ASSETS first"
        )
        assert _sha256(path) == asset.sha256, f"{name}: SHA-256 mismatch with manifest"

    def test_base_url_is_pinned_to_a_revision(self):
        """The download URL must not follow a moving branch."""
        assert assets.HF_REVISION not in ("main", "master", "")
        assert assets.HF_REVISION in assets.BASE_URL


class TestResolution:
    def test_unknown_asset_is_a_key_error(self):
        with pytest.raises(KeyError):
            assets.resolve("no-such-file.bin", download=False)

    def test_no_download_when_disabled(self, tmp_path, monkeypatch, no_download):
        """With downloads disabled and nothing cached, required assets raise and
        optional ones return None. The package weights dir is redirected to an
        empty location so a development checkout behaves like a wheel install."""
        monkeypatch.setenv("FACET_HOME", str(tmp_path))
        monkeypatch.setattr(assets, "_PKG", tmp_path / "pkg")  # no bundled weights

        required = next(n for n, a in assets.ASSETS.items() if not a.optional)
        optional = [n for n, a in assets.ASSETS.items() if a.optional]

        with pytest.raises(assets.AssetUnavailable) as exc:
            assets.resolve(required, download=False)
        assert str(tmp_path) in str(exc.value)  # tells the user where to put it

        for name in optional:
            assert assets.resolve(name, download=False) is None

    def test_env_override_wins_over_cache(self, tmp_path, monkeypatch, no_download):
        monkeypatch.setenv("FACET_HOME", str(tmp_path / "cache"))
        monkeypatch.setattr(assets, "_PKG", tmp_path / "pkg")
        override = tmp_path / "elsewhere" / "facet_v3.pt"
        override.parent.mkdir()
        override.write_bytes(b"not a real checkpoint")
        monkeypatch.setenv("FACET_V3_PT", str(override))
        assert assets.resolve("facet_v3.pt", download=False) == override

    def test_legacy_env_name_still_honoured(self, tmp_path, monkeypatch, no_download):
        monkeypatch.setenv("FACET_HOME", str(tmp_path / "cache"))
        monkeypatch.setattr(assets, "_PKG", tmp_path / "pkg")
        override = tmp_path / "ckpt.pt"
        override.write_bytes(b"x")
        monkeypatch.setenv("FACET_CHECKPOINT", str(override))
        assert assets.resolve("facet_v3.pt", download=False) == override

    def test_corrupt_cache_is_not_trusted(self, tmp_path, monkeypatch, no_download):
        """A file of the wrong size in the cache is discarded, not returned."""
        monkeypatch.setenv("FACET_HOME", str(tmp_path))
        monkeypatch.setattr(assets, "_PKG", tmp_path / "pkg")
        bad = tmp_path / "facet_v3.pt"
        bad.write_bytes(b"truncated")
        with pytest.raises(assets.AssetUnavailable):
            assets.resolve("facet_v3.pt", download=False)
        assert not bad.exists(), "corrupt cached file should have been removed"

    def test_cache_dir_honours_facet_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FACET_HOME", str(tmp_path))
        assert assets.cache_dir() == tmp_path


class TestIndexFallback:
    def test_missing_index_degrades_to_parametric(self, tmp_path, monkeypatch, no_download):
        from facet import inference

        monkeypatch.setenv("FACET_HOME", str(tmp_path))
        monkeypatch.setattr(assets, "_PKG", tmp_path / "pkg")
        assert inference._find_index() is None
