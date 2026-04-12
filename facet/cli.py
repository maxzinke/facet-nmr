"""Command-line interface for FACET.

Usage::

    # Predict from TALOS-N tab file (default: outputs .tbl + .aco + pred.tab)
    facet predict shifts.tab

    # Specific output format
    facet predict shifts.tab -o restraints.tbl --format tbl

    # All output formats
    facet predict shifts.tab --all

    # From CSV
    facet predict shifts.csv -o restraints.tbl

    # Custom checkpoint
    facet predict shifts.tab --checkpoint path/to/model.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("facet")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="facet",
        description="FACET: Predict backbone torsion angles from NMR chemical shifts",
    )
    sub = parser.add_subparsers(dest="command")

    # ── predict ──
    pred = sub.add_parser("predict", help="Predict torsion angles from a shift list")
    pred.add_argument("input", help="Shift list file (.tab, .csv, .nef)")
    pred.add_argument("-o", "--output", default=None,
                      help="Output path (default: <input>_facet.<format>)")
    pred.add_argument("--format", choices=["tbl", "aco", "nef", "predtab", "csv", "json"],
                      default=None, help="Output format (default: write tbl + aco + predtab)")
    pred.add_argument("--all", action="store_true",
                      help="Write all output formats")
    pred.add_argument("--checkpoint", default=None,
                      help="Path to model checkpoint (.pt)")
    pred.add_argument("--device", default=None, help="cuda / cpu")
    pred.add_argument("--include-all", action="store_true",
                      help="Include all residues in restraint files (not just accepted)")
    pred.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.command == "predict":
        _cmd_predict(args)


def _cmd_predict(args) -> None:
    from .inference import predict
    from .io.writers import write_tbl, write_aco, write_nef, write_predtab, write_csv, write_json

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    stem = input_path.stem
    out_dir = input_path.parent

    logger.info("Predicting from %s ...", input_path)
    result = predict(
        input_path,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    n_accepted = len(result.accepted())
    n_strong = len(result.strong())
    logger.info("Predicted %d residues: %d Strong, %d Accepted",
                result.n_residues, n_strong, n_accepted)

    accepted_only = not args.include_all

    if args.format:
        # Single format
        out = Path(args.output) if args.output else out_dir / f"{stem}_facet.{args.format}"
        writers = {
            "tbl": write_tbl, "aco": write_aco, "nef": write_nef,
            "predtab": write_predtab, "csv": write_csv, "json": write_json,
        }
        writer = writers[args.format]
        if args.format in ("tbl", "aco", "nef"):
            writer(result, out, accepted_only=accepted_only)
        else:
            writer(result, out)
        logger.info("Wrote %s", out)
    else:
        # Default: tbl + aco + predtab (or --all for everything)
        formats = ["tbl", "aco", "predtab"]
        if args.all:
            formats = ["tbl", "aco", "nef", "predtab", "csv", "json"]

        for fmt in formats:
            out = out_dir / f"{stem}_facet.{fmt}"
            if fmt == "tbl":
                write_tbl(result, out, accepted_only=accepted_only)
            elif fmt == "aco":
                write_aco(result, out, accepted_only=accepted_only)
            elif fmt == "nef":
                write_nef(result, out, accepted_only=accepted_only)
            elif fmt == "predtab":
                write_predtab(result, out)
            elif fmt == "csv":
                write_csv(result, out)
            elif fmt == "json":
                write_json(result, out)
            logger.info("Wrote %s", out)

    # Print summary to stdout
    print(result.summary())


if __name__ == "__main__":
    main()
