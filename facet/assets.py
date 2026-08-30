"""Locate model weights and reference data, downloading them on first use.

WHY THESE ARE NOT IN THE WHEEL
------------------------------
Two independent reasons, either of which would be sufficient.

**Packaging.** ``facet_retrieval_index.npz`` is 133 MB on its own, and PyPI rejects
any file over 100 MB. A wheel with the weights bundled measures ~155 MB and cannot be
uploaded at all. Downloading them separately is not an optimisation here; it is the
only way to publish.

**Freshness.** BMRB depositors submit corrections after release. A reference baked into
a wheel silently goes stale until the next version ships, while a downloaded one can be
refreshed without a release. This was BMRB's own suggestion when asked about
redistribution -- the data is CC0, so there is no licensing reason to fetch rather than
bundle, only a data-quality one.

RESOLUTION ORDER
----------------
For each asset, in order:

1. ``<package>/weights/<name>`` -- a development checkout, or a build that chose to
   vendor them.
2. ``$FACET_<ASSET>`` -- an explicit override, per asset.
3. ``$FACET_HOME`` or ``~/.facet/<name>`` -- the download cache.
4. Download to the cache.

Nothing is downloaded implicitly during import; the fetch happens on the first call
that actually needs the file, and can be disabled entirely with
``FACET_NO_DOWNLOAD=1`` for offline or air-gapped use.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["Asset", "ASSETS", "resolve", "cache_dir", "ensure_all", "AssetUnavailable"]


class AssetUnavailable(FileNotFoundError):
    """An asset is neither present locally nor retrievable."""


@dataclass(frozen=True)
class Asset:
    """One downloadable file, pinned by content hash."""

    sha256: str
    size: int
    #: Optional: assets the package can run without, degrading rather than failing.
    optional: bool = False


#: Every large file that used to ship inside the wheel. Sizes and hashes are of the
#: exact artifacts this release was validated against -- a mismatch means the file is
#: not the one the numbers in the README were measured on.
ASSETS: dict[str, Asset] = {
    "facet_v3.pt": Asset(
        sha256="d117fd39fa70b6f9d303c5c8cd447821bf88e6a7f32394ef250a6c8460d48cf0",
        size=5_206_151,
    ),
    "facet_v3.onnx": Asset(
        sha256="ee457594f60c797dd5bffb0343d365ca997b9b120818b3c131c28760244a1913",
        size=5_192_115,
    ),
    "facet_retrieval_index.npz": Asset(
        sha256="0efaccea5fb1af33fc56acd2f292563c9a0aee4e7246df70d2197a2d441371a4",
        size=132_621_033,
    ),
    "facet_retrieval_index.entries.json": Asset(
        sha256="12bd59518f30bd2dc4e9745f0341e78e605e9214854399294e199d6846cef53a",
        size=2_265_272,
    ),
    # The mask-safe fallback still works when this is absent -- predictions for
    # HA-missing residues fall back to the parametric head, with a warning -- so it is
    # marked optional rather than blocking a prediction run.
    "facet_shift_reference.npz": Asset(
        sha256="537119cc6679571fca087c091b9a04d4254c8b05ef762a86c031895ef9b0805e",
        size=9_620_268,
        optional=True,
    ),
}

#: Where the artifacts are published.
#:
#: A HuggingFace model repository rather than a GitHub release: it is built for model
#: weights, CDN-backed, and a 133 MB file is unremarkable there. The ``resolve`` URL
#: form serves raw file bytes over plain HTTP, so nothing here needs the
#: ``huggingface_hub`` package — this module depends only on the standard library.
#:
#: Pinned to a REVISION, not to ``main``. The SHA-256 values below describe the exact
#: files this release was validated against; if the URL followed a moving branch, a
#: later upload would break every installed copy's checksum check rather than being
#: picked up cleanly by a new release.
#:
#: Override with ``$FACET_ASSET_URL`` to mirror internally — the path layout is simply
#: ``<BASE_URL>/<filename>``.
HF_REPO = os.environ.get("FACET_ASSET_REPO", "SiXa18/facet-weights")
HF_REVISION = os.environ.get("FACET_ASSET_REVISION", "v0.4.0")
BASE_URL = os.environ.get(
    "FACET_ASSET_URL",
    f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}",
).rstrip("/")

_PKG = Path(__file__).resolve().parent


def cache_dir() -> Path:
    """Where downloaded assets live. ``$FACET_HOME`` overrides ``~/.facet``."""
    return Path(os.environ.get("FACET_HOME", str(Path.home() / ".facet")))


def _env_override(name: str) -> Optional[Path]:
    """``facet_v3.pt`` -> ``$FACET_V3_PT``; also the older explicit variable names."""
    legacy = {
        "facet_shift_reference.npz": "FACET_SHIFT_REFERENCE",
        "facet_retrieval_index.npz": "FACET_INDEX",
        "facet_v3.pt": "FACET_CHECKPOINT",
    }
    derived = name.upper().replace(".", "_").replace("-", "_")
    if derived.startswith("FACET_"):
        derived = derived[len("FACET_"):]
    for var in (legacy.get(name), "FACET_" + derived):
        if var and os.environ.get(var):
            return Path(os.environ[var])
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path: Path, asset: Asset) -> bool:
    """A truncated or corrupted download is worse than a missing one -- it fails later,
    somewhere unrelated. Check size first because it is free."""
    try:
        if path.stat().st_size != asset.size:
            return False
    except OSError:
        return False
    return _sha256(path) == asset.sha256


def _download(name: str, asset: Asset, dest: Path) -> None:
    import urllib.request

    url = f"{BASE_URL}/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    mb = asset.size / 1e6
    print(f"[facet] downloading {name} ({mb:.1f} MB) from {BASE_URL}", flush=True)

    # Download to a temporary file in the destination directory, then move into place.
    # A partial file left at the final path would be picked up by a later run and fail
    # its hash check forever, which looks like a corrupt release rather than a dropped
    # connection.
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        if not _verify(tmp, asset):
            raise AssetUnavailable(
                f"{name} downloaded from {url} but its checksum does not match the one "
                f"this release was validated against. Do not use it; report the URL."
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"[facet] saved to {dest}", flush=True)


def resolve(name: str, *, download: bool = True) -> Optional[Path]:
    """Return a usable path for ``name``, fetching it if needed.

    Args:
        name: Key in :data:`ASSETS`.
        download: Set ``False`` to look only at what is already on disk.

    Returns:
        The path, or ``None`` if the asset is optional and unavailable.

    Raises:
        AssetUnavailable: If a required asset cannot be found or fetched.
    """
    asset = ASSETS.get(name)
    if asset is None:
        raise KeyError(f"unknown asset {name!r}; known: {', '.join(sorted(ASSETS))}")

    bundled = _PKG / "weights" / name
    if bundled.exists():
        return bundled

    override = _env_override(name)
    if override is not None and override.exists():
        return override

    cached = cache_dir() / name
    if cached.exists():
        # Verify the cache rather than trusting it: a half-written file from an
        # interrupted run is otherwise indistinguishable from a good one.
        if _verify(cached, asset):
            return cached
        print(f"[facet] cached {name} failed verification, re-downloading", flush=True)
        cached.unlink(missing_ok=True)

    if not download or os.environ.get("FACET_NO_DOWNLOAD"):
        if asset.optional:
            return None
        raise AssetUnavailable(
            f"{name} is not available locally and downloads are disabled. "
            f"Fetch it from {BASE_URL}/{name} and place it in {cache_dir()}, "
            f"or unset FACET_NO_DOWNLOAD."
        )

    try:
        _download(name, asset, cached)
    except AssetUnavailable:
        raise
    except Exception as exc:  # network, DNS, 404, permissions
        message = (
            f"could not download {name} from {BASE_URL}/{name}: {exc}. "
            f"Download it manually and place it in {cache_dir()}."
        )
        if asset.optional:
            print(f"[facet] {message}", flush=True)
            return None
        raise AssetUnavailable(message) from exc
    return cached


def ensure_all(include_optional: bool = True) -> dict[str, Optional[Path]]:
    """Fetch every asset up front — for container builds and offline preparation."""
    out: dict[str, Optional[Path]] = {}
    for name, asset in ASSETS.items():
        if asset.optional and not include_optional:
            continue
        out[name] = resolve(name)
    return out


if __name__ == "__main__":  # `python -m facet.assets` prepares an offline install
    for key, path in ensure_all().items():
        print(f"{key:40} {path if path else '(unavailable, optional)'}")
