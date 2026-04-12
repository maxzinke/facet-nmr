"""Tests for FACET IO readers and writers."""
import json
import tempfile
from pathlib import Path

import pytest

from facet.io.formats import FACETResult, Residue, ResiduePrediction, ShiftList
from facet.io.nef import read_nef
from facet.io.nmrstar import read_nmrstar
from facet.io.readers import read_auto, read_csv, read_tab
from facet.io.writers import (
    write_aco,
    write_csv,
    write_json,
    write_nef,
    write_predtab,
    write_tbl,
)

# ────── Fixtures ──────

SAMPLE_TAB_LONG = """\
REMARK Test shifts
VARS   RESID RESNAME ATOMNAME SHIFT
FORMAT %4d %1s %4s %8.3f

   1 M    H    8.421
   1 M   HA    4.312
   1 M    N  120.500
   1 M   CA   55.400
   1 M   CB   32.900
   1 M    C  176.200
   2 Q    H    8.810
   2 Q   HA    4.290
   2 Q    N  123.200
   2 Q   CA   55.900
   3 G    H    8.060
   3 G   HA    3.840
   3 G    N  110.600
   3 G   CA   46.300
"""

SAMPLE_TAB_CONDENSED = """\
VARS   RESID RESNAME C CA CB HA H N
FORMAT %4d %s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f

   1 M  176.200   55.400   32.900    4.312    8.421  120.500
   2 Q  175.800   55.900   28.800    4.290    8.810  123.200
   3 G  174.900   46.300 9999.000    3.840    8.060  110.600
"""

SAMPLE_CSV = """\
ResID,AA,H,HA,N,CA,CB,C
1,M,8.421,4.312,120.5,55.4,32.9,176.2
2,Q,8.81,4.29,123.2,55.9,28.8,175.8
3,G,8.06,3.84,110.6,46.3,,174.9
"""


def _write_tmp(content: str, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def _sample_result() -> FACETResult:
    return FACETResult(
        residues=[
            ResiduePrediction(1, "MET", -64.8, -45.2, -3.5, "Medium", "H", chi1=1, phi_err=22.0, psi_err=22.0),
            ResiduePrediction(2, "GLN", -65.0, -44.0, -4.0, "High", "H", chi1=2, phi_err=16.0, psi_err=16.0),
            ResiduePrediction(3, "GLY", -80.0, 150.0, -5.5, "Low", "C", chi1=None, phi_err=26.0, psi_err=26.0),
        ],
        source="test",
    )


# ────── Reader tests ──────

class TestTabReader:
    def test_long_form(self):
        path = _write_tmp(SAMPLE_TAB_LONG, ".tab")
        sl = read_tab(path)
        assert sl.n_residues == 3
        assert sl.residues[0].comp_id == "MET"
        assert sl.residues[0].shifts["H"] == pytest.approx(8.421)
        assert sl.residues[0].shifts["CA"] == pytest.approx(55.4)
        assert sl.residues[2].comp_id == "GLY"
        assert "CB" not in sl.residues[2].shifts  # GLY has no CB in input

    def test_condensed_form(self):
        path = _write_tmp(SAMPLE_TAB_CONDENSED, ".tab")
        sl = read_tab(path)
        assert sl.n_residues == 3
        assert sl.residues[0].shifts["C"] == pytest.approx(176.2)
        # GLY CB = 9999 should be treated as missing
        assert "CB" not in sl.residues[2].shifts

    def test_to_arrays(self):
        path = _write_tmp(SAMPLE_TAB_LONG, ".tab")
        sl = read_tab(path)
        shifts, masks, comp_ids, seq_ids = sl.to_arrays()
        assert shifts.shape == (3, 6)
        assert masks.shape == (3, 6)
        assert comp_ids == ["MET", "GLN", "GLY"]
        assert seq_ids == [1, 2, 3]
        # MET has all 6 atoms
        assert masks[0].sum() == 6
        # GLN has 4 atoms in this sample
        assert masks[1].sum() == 4


class TestCsvReader:
    def test_read(self):
        path = _write_tmp(SAMPLE_CSV, ".csv")
        sl = read_csv(path)
        assert sl.n_residues == 3
        assert sl.residues[0].comp_id == "MET"
        assert sl.residues[2].comp_id == "GLY"
        assert "CB" not in sl.residues[2].shifts  # empty field


SAMPLE_NEF = """\
data_test

save_nef_chemical_shift_list_1
   _nef_chemical_shift_list.sf_category    nef_chemical_shift_list
   _nef_chemical_shift_list.sf_framecode   nef_chemical_shift_list_1

   loop_
      _nef_chemical_shift.chain_code
      _nef_chemical_shift.sequence_code
      _nef_chemical_shift.residue_name
      _nef_chemical_shift.atom_name
      _nef_chemical_shift.value
      _nef_chemical_shift.value_uncertainty
      _nef_chemical_shift.element
      _nef_chemical_shift.isotope_number

      A 1 MET H   8.421  0.02  H  1
      A 1 MET HA  4.312  0.02  H  1
      A 1 MET N  120.500 0.1   N  15
      A 1 MET CA 55.400  0.2   C  13
      A 1 MET CB 32.900  0.2   C  13
      A 1 MET C  176.200 0.3   C  13
      A 2 GLN H   8.810  0.02  H  1
      A 2 GLN HA  4.290  0.02  H  1
      A 2 GLN N  123.200 0.1   N  15
      A 2 GLN CA 55.900  0.2   C  13

   stop_
save_
"""


SAMPLE_NMRSTAR = """\
data_test

save_assigned_chemical_shifts_1
   _Assigned_chem_shift_list.Sf_category     assigned_chemical_shifts
   _Assigned_chem_shift_list.Sf_framecode    assigned_chem_shift_list_1

   loop_
      _Atom_chem_shift.ID
      _Atom_chem_shift.Entity_assembly_ID
      _Atom_chem_shift.Seq_ID
      _Atom_chem_shift.Comp_ID
      _Atom_chem_shift.Atom_ID
      _Atom_chem_shift.Val

      1 1 1 MET H   8.421
      2 1 1 MET HA  4.312
      3 1 1 MET N  120.500
      4 1 1 MET CA 55.400
      5 1 1 MET CB 32.900
      6 1 1 MET C  176.200
      7 1 2 GLN H   8.810
      8 1 2 GLN N  123.200
      9 1 2 GLN CA 55.900

   stop_
save_
"""


class TestNmrstarReader:
    def test_read(self):
        path = _write_tmp(SAMPLE_NMRSTAR, ".str")
        sl = read_nmrstar(path)
        assert sl.n_residues == 2
        assert sl.residues[0].comp_id == "MET"
        assert sl.residues[0].shifts["H"] == pytest.approx(8.421)
        assert sl.residues[0].shifts["C"] == pytest.approx(176.2)
        assert sl.residues[1].shifts["CA"] == pytest.approx(55.9)

    def test_auto_detect(self):
        path = _write_tmp(SAMPLE_NMRSTAR, ".str")
        sl = read_auto(path)
        assert sl.n_residues == 2


class TestNefReader:
    def test_read(self):
        path = _write_tmp(SAMPLE_NEF, ".nef")
        sl = read_nef(path)
        assert sl.n_residues == 2
        assert sl.residues[0].comp_id == "MET"
        assert sl.residues[0].shifts["H"] == pytest.approx(8.421)
        assert sl.residues[0].shifts["C"] == pytest.approx(176.2)
        assert sl.residues[1].shifts["CA"] == pytest.approx(55.9)
        # GLN in this sample has no CB
        assert "CB" not in sl.residues[1].shifts

    def test_auto_detect(self):
        path = _write_tmp(SAMPLE_NEF, ".nef")
        sl = read_auto(path)
        assert sl.n_residues == 2


class TestAutoDetect:
    def test_tab_by_extension(self):
        path = _write_tmp(SAMPLE_TAB_LONG, ".tab")
        sl = read_auto(path)
        assert sl.n_residues == 3

    def test_csv_by_extension(self):
        path = _write_tmp(SAMPLE_CSV, ".csv")
        sl = read_auto(path)
        assert sl.n_residues == 3

    def test_tab_by_content(self):
        path = _write_tmp(SAMPLE_TAB_LONG, ".txt")
        sl = read_auto(path)
        assert sl.n_residues == 3


# ────── Writer tests ──────

class TestWriters:
    def test_tbl(self, tmp_path):
        result = _sample_result()
        out = write_tbl(result, tmp_path / "test.tbl", accepted_only=True)
        text = out.read_text()
        assert "assign" in text
        # 2 accepted residues (Strong + Good). Both must emit at least one
        # dihedral. N-terminal PHI and C-terminal PSI may be skipped when
        # the neighbor isn't in the residue list — that's correct behavior.
        assert "PHI" in text or "PSI" in text
        # Residue 1 (MET) has no i-1 → PHI skipped
        assert "(resid 0 and name C)" not in text

    def test_aco(self, tmp_path):
        result = _sample_result()
        out = write_aco(result, tmp_path / "test.aco", accepted_only=True)
        text = out.read_text()
        assert "PHI" in text
        assert "CYANA" not in text  # should not contain program name in the body
        lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
        # 2 accepted residues × 2 angles = 4 lines
        assert len(lines) == 4

    def test_nef(self, tmp_path):
        result = _sample_result()
        out = write_nef(result, tmp_path / "test.nef", accepted_only=True)
        text = out.read_text()
        assert "nef_dihedral_restraint" in text
        assert "data_facet_restraints" in text
        assert "PHI" in text
        assert "PSI" in text

    def test_predtab(self, tmp_path):
        result = _sample_result()
        out = write_predtab(result, tmp_path / "test.predtab")
        text = out.read_text()
        assert "VARS" in text
        # All 3 residues (not filtered)
        data_lines = [l for l in text.splitlines()
                      if l.strip() and not l.strip().startswith(("REMARK", "VARS", "FORMAT"))]
        assert len(data_lines) == 3

    def test_csv_writer(self, tmp_path):
        result = _sample_result()
        out = write_csv(result, tmp_path / "test.csv")
        text = out.read_text()
        lines = text.strip().splitlines()
        assert lines[0].startswith("ResID")
        assert len(lines) == 4  # header + 3 residues

    def test_json_writer(self, tmp_path):
        result = _sample_result()
        out = write_json(result, tmp_path / "test.json")
        data = json.loads(out.read_text())
        assert data["n_residues"] == 3
        assert len(data["residues"]) == 3
        assert data["residues"][0]["comp_id"] == "MET"


# ────── FACETResult methods ──────

class TestFACETResult:
    def test_accepted(self):
        result = _sample_result()
        accepted = result.accepted()
        assert len(accepted) == 2  # Strong + Good
        classes = {r.confidence_class for r in accepted}
        assert "Low" not in classes

    def test_summary(self):
        result = _sample_result()
        text = result.summary()
        assert "FACET prediction" in text
        assert "MET" in text

    def test_to_tbl(self, tmp_path):
        result = _sample_result()
        result.to_tbl(str(tmp_path / "out.tbl"))
        assert (tmp_path / "out.tbl").exists()
