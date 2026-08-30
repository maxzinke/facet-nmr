"""Tests for FACET inference pipeline."""
import pytest
import numpy as np

from facet.io.formats import Residue, ShiftList
from facet.random_coil import RANDOM_COIL_SHIFTS, to_secondary_shifts


class TestRandomCoil:
    def test_tables_complete(self):
        """All 20 standard AAs have RC values."""
        assert len(RANDOM_COIL_SHIFTS) == 20
        for aa, shifts in RANDOM_COIL_SHIFTS.items():
            assert "CA" in shifts
            assert "N" in shifts
            if aa != "GLY":
                assert "CB" in shifts
            if aa != "PRO":
                assert "H" in shifts

    def test_secondary_shifts(self):
        """Secondary shifts = observed - random_coil."""
        # Column order: H=0, HA=1, N=2, CA=3, CB=4, C=5
        shifts = np.array([[0, 0, 0, 55.4, 0, 0]], dtype=np.float32)
        masks = np.array([[0, 0, 0, 1, 0, 0]], dtype=np.float32)
        comp_ids = ["ALA"]
        sec = to_secondary_shifts(shifts, masks, comp_ids)
        # ALA CA RC = 52.5, so secondary = 55.4 - 52.5 = 2.9
        assert sec[0, 3] == pytest.approx(2.9, abs=0.01)

    def test_missing_atoms_are_zero(self):
        """Unobserved atoms get 0 secondary shift (not NaN)."""
        shifts = np.full((1, 6), np.nan, dtype=np.float32)
        masks = np.zeros((1, 6), dtype=np.float32)
        comp_ids = ["ALA"]
        sec = to_secondary_shifts(shifts, masks, comp_ids)
        assert np.all(sec == 0.0)


class TestHardening:
    """Tests for the hardened input-validation pipeline."""

    def test_shift_sanity_rejects_miscalibrated(self):
        """Shifts wildly outside physical range should be rejected."""
        from facet import predict
        from facet.io.formats import Residue, ShiftList

        # CA at 155 ppm (should be 40-75) — simulates a 13C/15N mix-up
        residues = [
            Residue(i + 1, "ALA", {
                "H": 8.0, "HA": 4.0, "N": 120.0,
                "CA": 155.0, "CB": 19.0, "C": 177.0,
            })
            for i in range(5)
        ]
        sl = ShiftList(residues=residues)
        with pytest.raises(ValueError, match="physical range"):
            predict(sl, checkpoint="does-not-exist.pt")

    def test_too_few_residues(self):
        from facet import predict
        from facet.io.formats import Residue, ShiftList
        sl = ShiftList(residues=[
            Residue(1, "ALA", {"CA": 52.5}),
            Residue(2, "ALA", {"CA": 52.5}),
        ])
        with pytest.raises(ValueError, match="Too few residues"):
            predict(sl, checkpoint="does-not-exist.pt")


class TestValidation:
    """Input validation in predict() — non-canonical AAs, missing heavy atoms.

    Contract for non-canonical residues (since v0.2.2, commit 154bc49): they are
    skipped with a warning and prediction proceeds on the standard residues.
    A ValueError is raised only when fewer than the minimum number of standard
    residues remain after skipping.
    """

    def test_noncanonical_aas_skipped_with_warning(self, caplog):
        """Xeno AAs are dropped and logged; validation then passes."""
        from facet import predict
        from facet.io.formats import Residue, ShiftList

        residues = [
            Residue(i + 1, "ALA", {"H": 8.0, "HA": 4.0, "N": 120.0,
                                   "CA": 52.5, "CB": 19.1, "C": 177.8})
            for i in range(5)
        ]
        residues.append(Residue(6, "MSE", {"H": 8.0, "HA": 4.0, "N": 120.0, "CA": 55.0}))
        sl = ShiftList(residues=residues)

        # Validation passes, so predict() gets as far as loading the checkpoint.
        # A deliberately missing checkpoint proves we got past the AA check
        # without loading the real model.
        with caplog.at_level("WARNING", logger="facet"):
            with pytest.raises(FileNotFoundError):
                predict(sl, checkpoint="does-not-exist.pt")

        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("non-canonical" in w and "MSE" in w for w in warnings), warnings
        assert any("predicting on the 5 standard residues" in w for w in warnings), warnings

    def test_all_noncanonical_raises(self, caplog):
        """When nothing standard remains, predict() raises before model load."""
        from facet import predict
        from facet.io.formats import Residue, ShiftList

        residues = [
            Residue(1, "NVA", {"H": 8.0, "HA": 4.0, "N": 120.0, "CA": 55.0}),
            Residue(2, "CGU", {"H": 8.1, "HA": 4.1, "N": 121.0, "CA": 56.0}),
            Residue(3, "7C9", {"H": 8.2, "HA": 4.2, "N": 122.0, "CA": 57.0}),
        ]
        sl = ShiftList(residues=residues)
        with caplog.at_level("WARNING", logger="facet"):
            with pytest.raises(ValueError, match="fewer than FACET's minimum"):
                predict(sl, checkpoint="does-not-exist.pt")
        assert any("non-canonical" in r.getMessage() for r in caplog.records)

    def test_proton_only_rejected(self):
        """Dataset with only H/HA observed should raise ValueError."""
        from facet import predict
        from facet.io.formats import Residue, ShiftList

        # Standard AAs but only H and HA — should fail heavy-atom check
        residues = [
            Residue(i + 1, "ALA", {"H": 8.0 + i * 0.01, "HA": 4.0})
            for i in range(10)
        ]
        sl = ShiftList(residues=residues)
        with pytest.raises(ValueError, match="heavy-atom"):
            predict(sl, checkpoint="does-not-exist.pt")


class TestShiftList:
    def test_to_arrays(self):
        sl = ShiftList(residues=[
            Residue(1, "ALA", {"H": 8.24, "CA": 52.5, "CB": 19.1}),
            Residue(2, "GLY", {"H": 8.33, "CA": 45.4}),
        ])
        shifts, masks, comp_ids, seq_ids = sl.to_arrays()
        assert shifts.shape == (2, 6)
        assert masks[0, 0] == 1.0  # H observed for ALA
        assert masks[0, 2] == 0.0  # N not observed
        assert masks[1, 4] == 0.0  # CB not observed for GLY
        assert comp_ids == ["ALA", "GLY"]

    def test_empty(self):
        sl = ShiftList(residues=[])
        shifts, masks, comp_ids, seq_ids = sl.to_arrays()
        assert shifts.shape == (0, 6)


@pytest.mark.needs_assets
class TestEndToEnd:
    """Runs the real model + retrieval index on the bundled ubiquitin example.

    Skipped automatically when the assets are not on disk (see conftest.py).
    """

    def test_ubiquitin_example(self):
        from pathlib import Path

        from facet import predict
        from facet.io.formats import CONF_HIGH, CONF_MEDIUM

        example = Path(__file__).resolve().parents[1] / "examples" / "ubiquitin.tab"
        result = predict(example)

        assert len(result.residues) == 10
        tiers = {"High", "Medium", "Low", "Flexible"}
        for r in result.residues:
            assert r.confidence_class in tiers, r.confidence_class
            assert -180.0 <= r.phi <= 180.0
            assert -180.0 <= r.psi <= 180.0
            assert r.ss in {"H", "E", "C"}

        # Restraint gating follows the documented rule.
        assert {r.confidence_class for r in result.accepted()} <= {CONF_HIGH}
        assert {r.confidence_class for r in result.accepted(include_medium=True)} \
            <= {CONF_HIGH, CONF_MEDIUM}
        # Retrieval mode was actually used (index present), so basins are populated.
        assert any(r.basin_populations is not None for r in result.residues)
