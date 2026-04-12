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
