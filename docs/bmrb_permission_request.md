# Draft email to BMRB

**To:** help@bmrb.io
**Subject:** Redistributing BMRB-derived shifts in an open-source tool

---

Hi,

I'm developing an open-source tool that predicts backbone phi/psi angles from chemical
shifts, similar in purpose to TALOS-N. Part of it works by nearest-neighbour lookup, so
it needs a reference set built from BMRB: 3,470 entries, ~310,000 residues, around
4.8 million secondary shift values with entry and residue IDs attached, about 10 MB in
total.

I'd like to ask whether I may redistribute that as part of the software. The values are
secondary shifts rather than raw ppm, but the random-coil reference would ship
alongside them, so the original numbers are recoverable — I don't think I can honestly
call it just a statistic.

I plan to release the code openly, and may offer commercial licences for related
software later, so I'd rather ask now than assume.

If it helps, I could strip the entry and residue IDs, restrict onward redistribution,
or have the tool download the data from you on first use instead of bundling it. Happy
to add whatever citation you'd prefer.

Thanks,
Maxim

---

## Before sending

- Everything is phrased as forward-looking. Don't add a repo link — that reintroduces
  what the phrasing is avoiding.
- Keep the commercial sentence if commercial licensing is actually on the table.
  Permission granted on a partial description isn't worth much.
- The download-on-first-use offer is the real fallback if they decline. The code
  already handles the file being absent.
- Separately: the Space is currently serving this file publicly under an MIT header.
  Worth resolving that before or alongside sending, so the situation matches the email.
