# One protein, end to end

BMRB **15232** — a proline-free mutant of staphylococcal nuclease (SNase V8), 149
residues in the construct, 127 with assigned backbone shifts, 96 with a usable φ/ψ
truth. It is a typical test protein: a mix of helix, strand and coil, median errors
close to the benchmark-wide ones (FACET 12.0°, TALOS-N 14.3°). Everything below is
produced by

```
python benchmarks/walkthrough.py --id 15232
```

## 1. The input

`data/inputs/15232.tab` — the same NMRPipe table TALOS-N received:

```
REMARK Chemical shift table for BMRB entry 15232

DATA FIRST_RESID 1

DATA SEQUENCE ATSTKKLHKE AATLIKAIDG DTVKLMYKGQ AMTFRLLLVD TAETKHTKKG
DATA SEQUENCE VEKYGAEASA FTKKMVENAK KIEVEFDKGQ RTDKYGRGLA YIYADGKMVN
DATA SEQUENCE EALVRQGLAK VAYVYKGNNT HEQLLRKSEA QAKKEKLNIW SEDNADSGQ

VARS   RESID RESNAME ATOMNAME SHIFT
FORMAT %4d   %1s     %4s      %8.3f

   3 S   HN      8.559
   3 S   HA      4.592
   3 S   N     119.090
   3 S   CA     58.094
   3 S   CB     63.978
   4 T   HN      8.348
   ...
```

One line per assigned nucleus. Residue 3 has no C' shift; that is normal and the
model treats it as a masked input.

## 2. Running FACET

```
facet predict benchmarks/data/inputs/15232.tab
```

writes `15232_facet.predtab`:

```
REMARK FACET backbone torsion angle prediction
REMARK facet-nmr 0.4.0 | retrieval index v0.3.0

VARS   RESID RESNAME PHI PSI DPHI DPSI SS CHI1 CLASS
FORMAT %4d %s %8.1f %8.1f %6.1f %6.1f %s %s %s

    3 S   -103.3    128.3   24.6   14.3 C  g- Medium
    4 T   -100.5    130.2   17.2   18.0 C  g+ Medium
    5 K   -112.6    141.3   21.6   23.4 C  g- Medium
    6 K    -89.1    140.7   20.0   25.0 C  g- Medium
    7 L    -87.1    140.7   14.2   22.5 C  g- Medium
```

`DPHI`/`DPSI` are the spread of the retrieved neighbour cluster (1σ, degrees), `CLASS`
is the confidence tier. Residues 3–7 are the disordered N-terminus: the model says so
(coil, Medium, wide error bars), and the structure agrees — they have no φ/ψ truth and
do not appear in the scored table. Of the 127 predicted residues, 82 are High, 32
Medium, 11 Low and 2 Flexible.

## 3. Comparing with the truth and with TALOS-N

The script joins FACET's output with `per_residue.csv`, which carries the structure's
φ/ψ and the TALOS-N prediction for the same residues. A strand, a turn and a helix,
verbatim from the script's output (error = RMS of the wrapped φ and ψ deltas):

```
resid AA  ss   phi_true  psi_true |  phi_facet psi_facet tier      err | phi_talosn psi_talosn class     err
   22 THR   E    -103.5     108.3 |    -121.5     133.4 High       21.8 |    -117.2     141.3 Strong     25.3
   23 VAL   E    -137.9     145.5 |    -132.6     143.2 High        4.1 |    -130.3     141.9 Strong      6.0
   24 LYS   E     -86.9     108.2 |    -104.7     118.4 High       14.5 |    -106.3     125.1 Strong     18.3
   25 LEU   E    -111.5     133.6 |    -102.0     122.5 High       10.3 |     -98.2     123.7 Strong     11.7
   26 MET   E     -81.6     101.3 |    -101.8     120.1 High       19.5 |    -101.1     117.5 Strong     17.9
   27 TYR   E    -104.0     140.1 |    -127.3     122.1 High       20.8 |    -129.6     123.9 Strong     21.4
   28 LYS   C      44.2      28.3 |      58.1      28.5 High        9.9 |      54.3      39.2 Strong     10.5
   29 GLY   C      89.0      18.7 |      86.7       2.1 High       11.8 |      77.3       8.7 Strong     10.9
   ...
   36 LEU   E     -57.4     121.0 |     -78.8     128.1 High       15.9 |     -72.8     129.3 Strong     12.3
   37 LEU   C     -77.2     136.0 |     -83.1     -44.0 Low       127.3 |    -100.7     -34.9 Warn      122.0
   38 LEU   C      58.2      27.9 |      62.2      27.0 Medium      2.9 |    -114.9     127.5 Warn      141.2
   39 VAL   C    -140.1     149.6 |    -127.3     157.5 Medium     10.6 |    -133.9     153.7 Strong      5.2
   ...
   56 ALA   H     -67.1     -42.9 |     -63.6     -39.9 High        3.2 |     -65.2     -40.7 Strong      2.0
   57 GLU   H     -68.3     -29.4 |     -67.9     -36.8 High        5.3 |     -67.3     -39.6 Strong      7.2
   58 ALA   H     -59.6     -48.0 |     -61.1     -44.1 High        3.0 |     -63.4     -40.9 Strong      5.7
   59 SER   H     -68.8     -37.1 |     -60.9     -38.5 High        5.7 |     -64.2     -42.1 Strong      4.7
```

What to notice:

* **Residues 28–29 (Lys-Gly, a left-handed turn, φ > 0).** Both methods get it; this
  is the kind of residue a Ramachandran prior alone would miss.
* **Residue 37.** Both methods put it in the helical region; the structure has it
  extended. The error is ~122–127° for both, and both flag it — FACET drops it to
  Low, TALOS-N says `Warn` for 37–38. This is what a large error usually looks like
  in the benchmark: a single residue in a loop, flagged, not a systematic failure.
* **Residue 38.** FACET recovers (2.9°, correctly positive φ); TALOS-N does not
  (141°). One residue, one direction of the head-to-head count.
* **The helix (54–68).** Errors of 2–17° for both methods. Helix residues are 44 % of
  the benchmark, which is why the all-residue median is dominated by "easy" residues
  and why the coil median (18.0° vs 19.3°) is the more discriminating number.

And one residue to be honest about:

```
   54 TYR   H      60.8      43.4 |     -70.1     -35.0 High      107.9 |     -66.5     -39.9 Strong    107.6
```

Residue 54 sits at a helix cap with a *positive*-φ conformation in the deposited
structure. Both methods place it in the ordinary right-handed helical region and miss
by ~108° — and both call it confident (`High` / `Strong`). Confidence tiers are
calibrated, not infallible: 9.9 % of High-tier residues still miss by more than 25°,
and this is one of them. The reverse case exists too: residue 15 carries the
`Flexible` tier (its neighbours formed no coherent cluster) yet its prediction is
8.6° off — the tier is conservative there.

## 4. Along the sequence

![walkthrough](figures/walkthrough_15232.png)

Left: per-residue error, helix shaded red, strand shaded blue, the 25° fail threshold
dotted. The two traces move together — most of the error is a property of the residue
(loop, turn, or a place where structure and shifts disagree), not of the method — and
FACET sits slightly below TALOS-N on most of the strand and coil stretches. Right:
FACET's prediction on the Ramachandran plane, coloured by error. The eight residues
above 50° are 18, 37, 54, 79, 87, 118, 119 and 142 — every one a coil residue except
the helix-cap 54, and every one except 54 carrying a Medium or Low tier.

## 5. The numbers for this protein

```
this run       : FACET median 12.0°, TALOS-N median 14.3°, FACET lower on 51% of 96 paired residues
per_residue.csv: FACET median 12.0° on the same residues (row 15232 of per_protein.csv: 12.03° vs 14.26°)
```

The second line is the reproducibility check: the benchmark table was produced by the
released package, and re-running the walkthrough reproduces every one of the 96
recorded errors to within 0.5° (96 of 96). If you install `facet-nmr` and run
`python benchmarks/walkthrough.py`, this page is what you should see.
