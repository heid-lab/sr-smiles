import argparse
from enum import Enum

import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm

import sr_smiles
from sr_smiles import RxnToSr, SrToRxn, logger, set_verbose

console = Console()


class Direction(Enum):
    """Enumeration for the transformation directions."""

    RXN2SR = "Reaction SMILES ➡️ SR-SMILES"
    SR2RXN = "SR-SMILES ➡️ Reaction SMILES"


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
    banner_text.append("SR-SMILES\n", style="bold cyan underline")

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
            title=f"[bold cyan]🚀 SR-SMILES Converter v{sr_smiles.__version__}[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
            expand=False,
        )
    )


def reverse_reaction_smiles(rxn_smiles: str) -> str:
    """Reverse reaction smiles."""
    if ">>" in rxn_smiles:
        parts = rxn_smiles.split(">>")
        return ">>".join(parts[::-1])
    else:
        logger.warning(f"No '>>' found in reaction SMILES: {rxn_smiles}")
        return rxn_smiles


def main_rxn2sr():
    """CLI entry point: convert reaction SMILES to SR-SMILES in a CSV."""
    parser = argparse.ArgumentParser(description="Convert reaction SMILES to SR-SMILES")
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Path to output CSV file (default: overwrite input)")
    parser.add_argument("--rxn-col", default="rxn_smiles", help="Column name with reaction SMILES")
    parser.add_argument("--sr-col", default="sr_smiles", help="Name of new column for SR-SMILES")
    parser.add_argument("--keep-atom-mapping", action="store_true", help="Preserve atom mapping")
    parser.add_argument("--remove-hydrogens", action="store_true", help="Remove explicit hydrogens")
    parser.add_argument(
        "--use-rxnmapper",
        action="store_true",
        help="Use RxnMapper for atom mapping before SR transformation",
    )
    parser.add_argument("--balance-rxn", action="store_true", help="Balance the given reaction")
    parser.add_argument("--product-based", action="store_true", help="Balance the given reaction")
    parser.add_argument("--kekulize", action="store_true", help="Use the SMILES kekule form")
    parser.add_argument(
        "--ignore-aromatic-bonds",
        action="store_false",
        dest="keep_aromatic_bonds",
        help="",
    )
    args = parser.parse_args()

    print_banner(
        Direction.RXN2SR,
        input_col=args.rxn_col,
        output_col=args.sr_col,
        input_file=args.csv_file,
        output_file=args.output,
    )

    set_verbose(verbose=True)
    logger.info(f"📂 Loading CSV file from {args.csv_file} ...")
    df = pd.read_csv(args.csv_file)
    logger.info(f"✅ Loaded {len(df)} rows.")

    logger.info(f"🔄 Transforming column '{args.rxn_col}' → '{args.sr_col}' ...")
    transform = RxnToSr(
        keep_atom_mapping=args.keep_atom_mapping,
        remove_hydrogens=args.remove_hydrogens,
        balance_rxn=args.balance_rxn,
        rxn_col=args.rxn_col,
        kekulize=args.kekulize,
        keep_aromatic_bonds=args.keep_aromatic_bonds,
        use_rxnmapper=args.use_rxnmapper,
    )

    tqdm.pandas()
    if args.product_based:
        logger.info("Reverse reactions for product-based transformation")

        df[f"{args.rxn_col}_reversed"] = df[args.rxn_col].progress_apply(reverse_reaction_smiles)
        df[args.sr_col] = df[f"{args.rxn_col}_reversed"].progress_apply(transform)
    else:
        df[args.sr_col] = df[args.rxn_col].progress_apply(transform)

    logger.info("✅ Transformation complete.")

    output_path = args.output or args.csv_file
    logger.info(f"💾 Writing results to: {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"🎉 Done! Wrote {len(df)} rows with new column '{args.sr_col}'")


def main_sr2rxn():
    """CLI entry point: convert SR-SMILES to reaction SMILES in a CSV."""
    parser = argparse.ArgumentParser(description="Convert SR-SMILES to reaction SMILES")
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Path to output CSV file (default: overwrite input)")
    parser.add_argument("--sr-col", default="sr_smiles", help="Column name with SR-SMILES")
    parser.add_argument("--rxn-col", default="rxn_smiles", help="Name of new column for reaction SMILES")

    args = parser.parse_args()

    print_banner(
        Direction.SR2RXN,
        input_col=args.sr_col,
        output_col=args.rxn_col,
        input_file=args.csv_file,
        output_file=args.output,
    )

    set_verbose(verbose=True)

    logger.info(f"📂 Loading CSV file from {args.csv_file} ...")
    df = pd.read_csv(args.csv_file)
    logger.info(f"✅ Loaded {len(df)} rows.")

    logger.info(f"🔄 Transforming column '{args.sr_col}' → '{args.rxn_col}' ...")
    transform = SrToRxn(sr_col=args.sr_col)

    tqdm.pandas()
    df[args.rxn_col] = df[args.sr_col].progress_apply(transform)
    logger.info("✅ Transformation complete.")

    output_path = args.output or args.csv_file
    logger.info(f"💾 Writing results to: {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"🎉 Done! Wrote {len(df)} rows with new column '{args.rxn_col}'")
