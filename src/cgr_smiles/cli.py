import argparse
from enum import Enum

import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm

import cgr_smiles
from cgr_smiles import CgrToRxn, RxnToCgr, logger, set_verbose

console = Console()


class Direction(Enum):
    """Enumeration for the transformation directions."""

    RXN2CGR = "Reaction SMILES ➡️ CGR SMILES"
    CGR2RXN = "CGR SMILES ➡️ Reaction SMILES"


def print_banner(
    direction: Direction,
    input_col: str = None,
    output_col: str = None,
    input_file: str = None,
    output_file: str = None,
):
    """Print a Rich banner for the CLI."""
    banner_text = Text()

    banner_text.append("👋 Welcome to ", style="bold green")
    banner_text.append("CGR-SMILES\n", style="bold cyan underline")

    banner_text.append("Transforming ", style="yellow")
    banner_text.append(f"{direction.value}", style="bold yellow")

    if input_col:
        banner_text.append(f"\nInput column:    '{input_col}'", style="cyan")
    if output_col:
        banner_text.append(f"\nOutput column:   '{output_col}'", style="cyan")
    if input_file:
        banner_text.append(f"\nInput file:      {input_file}", style="cyan")
    if output_file:
        banner_text.append(f"\nOutput file:     {output_file}", style="cyan")

    console.print(
        Panel(
            Align.center(banner_text),
            title=f"[bold cyan]🚀 CGR SMILES Converter v{cgr_smiles.__version__}[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
            expand=False,
        )
    )


def main_rxn2cgr():
    """CLI entry point: convert reaction SMILES to CGR SMILES in a CSV."""
    parser = argparse.ArgumentParser(description="Convert reaction SMILES to CGR SMILES")
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Path to output CSV file (default: overwrite input)")
    parser.add_argument("--rxn-col", default="rxn_smiles", help="Column name with reaction SMILES")
    parser.add_argument("--cgr-col", default="cgr_smiles", help="Name of new column for CGR SMILES")
    parser.add_argument("--keep-atom-mapping", action="store_true", help="Preserve atom mapping")
    parser.add_argument("--remove-brackets", action="store_true", help="Remove brackets")
    parser.add_argument("--remove-hydrogens", action="store_true", help="Remove explicit hydrogens")
    parser.add_argument("--balance-rxn", action="store_true", help="Balance the given reaction")

    args = parser.parse_args()

    print_banner(
        Direction.RXN2CGR,
        input_col=args.rxn_col,
        output_col=args.cgr_col,
        input_file=args.csv_file,
        output_file=args.output,
    )

    set_verbose(verbose=True)
    logger.info(f"📂 Loading CSV file from {args.csv_file} ...")
    df = pd.read_csv(args.csv_file)
    logger.info(f"✅ Loaded {len(df)} rows.")

    logger.info(f"🔄 Transforming column '{args.rxn_col}' → '{args.cgr_col}' ...")
    transform = RxnToCgr(
        keep_atom_mapping=args.keep_atom_mapping,
        remove_brackets=args.remove_brackets,
        remove_hydrogens=args.remove_hydrogens,
        balance_rxn=args.balance_rxn,
        rxn_col=args.rxn_col,
    )

    tqdm.pandas()
    df[args.cgr_col] = df[args.rxn_col].progress_apply(transform)
    logger.info("✅ Transformation complete.")

    output_path = args.output or args.csv_file
    logger.info(f"💾 Writing results to: {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"🎉 Done! Wrote {len(df)} rows with new column '{args.cgr_col}'")


def main_cgr2rxn():
    """CLI entry point: convert CGR SMILES to reaction SMILES in a CSV."""
    parser = argparse.ArgumentParser(description="Convert CGR SMILES to reaction SMILES")
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Path to output CSV file (default: overwrite input)")
    parser.add_argument("--cgr-col", default="cgr_smiles", help="Column name with CGR SMILES")
    parser.add_argument("--rxn-col", default="rxn_smiles", help="Name of new column for reaction SMILES")

    args = parser.parse_args()

    print_banner(
        Direction.CGR2RXN,
        input_col=args.cgr_col,
        output_col=args.rxn_col,
        input_file=args.csv_file,
        output_file=args.output,
    )

    set_verbose(verbose=True)

    logger.info(f"📂 Loading CSV file from {args.csv_file} ...")
    df = pd.read_csv(args.csv_file)
    logger.info(f"✅ Loaded {len(df)} rows.")

    logger.info(f"🔄 Transforming column '{args.cgr_col}' → '{args.rxn_col}' ...")
    transform = CgrToRxn(cgr_col=args.cgr_col)

    tqdm.pandas()
    df[args.rxn_col] = df[args.cgr_col].progress_apply(transform)
    logger.info("✅ Transformation complete.")

    output_path = args.output or args.csv_file
    logger.info(f"💾 Writing results to: {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"🎉 Done! Wrote {len(df)} rows with new column '{args.rxn_col}'")
