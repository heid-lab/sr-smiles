
<!-- Adjust when put on github and pypi -->

<!-- <div align="center">

[![Build Status](https://github.com/heid-lab/cgr-smiles/actions/workflows/tests.yml/badge.svg)](https://github.com/heid-lab/cgr-smiles/actions)
[![Coverage](https://codecov.io/gh/heid-lab/cgr-smiles/branch/main/graph/badge.svg)](https://codecov.io/gh/heid-lab/cgr-smiles)

[![License](https://img.shields.io/github/license/heid-lab/cgr-smiles)](https://github.com/heid-lab/cgr-smiles/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cgr-smiles.svg)](https://pypi.org/project/cgr-smiles/)
[![Python versions](https://img.shields.io/pypi/pyversions/cgr-smiles.svg)](https://pypi.org/project/cgr-smiles)
[![Downloads](https://img.shields.io/github/downloads/heid-lab/cgr-smiles/total.svg)](https://github.com/heid-lab/cgr-smiles/releases)

</div> -->

[Overview](#overview) | [Installation](#installation) | [Usage](#usage) | [Contributing](#contributing) | [Citation](#citation) | [References](#references)

</div>

Note: This repository is not yet public and not yet on PyPI, hence the installation is not yet possible via pip. [WIP]

# CGR-SMILES

**CGR-SMILES** is a Python library for transforming reaction SMILES into a more compact and change-aware representation called **CGR-SMILES**. This representation explicitly encodes changes in chemical reactions, making it suitable for machine learning and data-driven applications.


---

## Overview

The CGR‑SMILES is inspired by the Condensed Graph of Reaction (CGR) representation,
a concept originating from graph‑based cheminformatics [[1]](#references).

In a classical CGR, the reactant and product graphs of a chemical reactions are superimposed and represented as a single unified graph. Atoms common to both sides are merged, and bonds are annotated with their changes (e.g., “single → double”, “added”, or “removed”).

CGR‑SMILES brings this concept to the string domain, making it suitable for language modeling applications. Instead of representing reactions as separate reactants and products, CGR‑SMILES combines them into a compact, local‑change‑aware representation that explicitly encodes how atoms and bonds transform.
It  is applicable to any organic reaction of the form `{reactant(s)}>>{product(s)}`.

While atom mappings are required to perform the transformation, the library provides workarounds for unmapped or partially mapped reactions by integrating atom‑mapping tools such as `RXNMapper` [[2]](#references).

Let's take a look at an example:

**Reaction SMILES:**

<code><span style="color:green">[F-:6]</span>.<span style="color:blue">[Br:1]</span><span style="color:red">[C@:2]</span>([H:5])([CH3:3])[NH2:4]>><span style="color:blue">[Br-:1]</span>.[CH3:3]<span style="color:red">[C@:2]</span>([H:5])(<span style="color:green">[F:6]</span>)[NH2:4]</code>

**CGR‑SMILES**:

<code><span style="color:green">{[F-]|F}</span>{~|-}<span style="color:red">{[C@]|[C@@]}</span>({-|~}<span style="color:blue">{Br|[Br-]}</span>)([H])([CH3])[NH2]</code>

👉 Notice how the CGR‑SMILES is more compact and explicitly encodes where atoms and bonds change during the reaction.

---

## Installation

[WIP]

```bash
pip install cgr-smiles
```

---

## Usage

### Your CGR-SMILES Toolkit: Seamless Reaction Conversions

The simplest use case involves mapped and balanced reactions. But don’t worry, the library also handles unmapped or unbalanced cases.

There are several ways to use CGR‑SMILES depending on your workflow:

1. **Core functions** (simple, flexible)
    - `rxn_to_cgr()`
    - `cgr_to_rxn()`
2. **Wrapper classes** (convenient for bulk data)
    - `RxnToCgr()`
    - `CgrToRxn()`
3. **CLI** (file-based workflows)
    - `rxn2cgr`
    - `cgr2rxn`

In the following sections, we’ll walk through basic examples for each option.

📘 For more detailed setups and parameter guidance, see the [example notebook](./examples/01_basic_usage.ipynb).

Let’s start with the imports:

```python
from cgr_smiles import CgrToRxn, RxnToCgr, cgr_to_rxn, rxn_to_cgr
```

### 1. Core functions (`rxn_to_cgr()` and `cgr_to_rxn()`)

These are the best place to start when exploring CGR‑SMILES. They provide a simple, direct way to understand how the library transforms reactions between RXN and CGR-SMILES.

```python
rxn_smiles = "[F-:6].[Br:1][C@:2]([H:5])([CH3:3])[NH2:4]>>[Br-:1].[CH3:3][C@:2]([H:5])([F:6])[NH2:4]"

cgr_smiles = rxn_to_cgr(
    rxn_smiles
)
rxn_back_with_mapping = cgr_to_rxn(
    cgr_smiles,
    add_atom_mapping=True  # optionally, to show atom mapping in the output smiles, defaults to False
)

print(f"RXN SMILES (original input):\n\t{rxn_smiles}\n")
print(f"CGR-SMILES (without mapping):\n\t{cgr_smiles}\n")
print(f"RXN SMILES (without mapping numbers):\n\t{rxn_back_without_mapping}\n")
```

```output
RXN SMILES (original input):
	[F-:6].[Br:1][C@:2]([H:5])([CH3:3])[NH2:4]>>[Br-:1].[CH3:3][C@:2]([H:5])([F:6])[NH2:4]

CGR-SMILES (without mapping):
	{[F-]|[F]}{~|-}{[C@]|[C@@]}({-|~}{[Br]|[Br-]})([H])([CH3])[NH2]

RXN SMILES (without mapping numbers):
	O=C(C#C[H])[H]>>[O+]#[C-].C(#C[H])[H]
```

### 2. Wrapper classes (`RxnToCgr()` and `CgrToRxn()`)

These offer a convenient, efficient interface for practical use, as they are ideal for processing large datasets or handling more complex cases like unmapped and unbalanced reactions.

**RXN to CGR:**
```python
import pandas as pd

rxn_list = [
    "[N:1]#[C:2][C@@:3]1([H:6])[C:4]([H:7])([H:8])[O:5]1>>[N:1]#[C:2][C@@:3]([C:4][H:7])([O:5][H:8])[H:6]",
    "[O:1]([C@@:2]([C:3](=[O:4])[H:9])([C:5]#[C:6][H:10])[H:8])[H:7]>>[O:1]([C@@:2]([C:3][O:4][H:9])([C:5]#[C:6][H:10])[H:8])[H:7]",
]

# using the RxnToCgr transform on a list of reactions
transform_to_cgr = RxnToCgr()
cgr_results = transform_to_cgr(rxn_list)

# using the RxnToCgr transform on a pd.DataFrame
df_data = pd.DataFrame({"reactions": rxn_list})
transform_to_cgr = RxnToCgr(
    rxn_col="reactions"    # <- in this case we need to specify the column name!
)
df_data["cgr_smiles"] = transform_to_cgr(rxn_list)

assert cgr_results == df_data["cgr_smiles"].tolist()
print("CGR-SMILES:\n\t" + "\n\t".join(cgr_results))
```

```output
CGR-SMILES:
	[N]#[C][C@@]1([H])[C]2([H]){-|~}[H]{~|-}[O]1{-|~}2
	[O]([C@@]([C]1{=|-}[O]{~|-}[H]{-|~}1)([C]#[C][H])[H])[H]
```

**And CGR back to RXN:**
```python
transform_to_rxn = CgrToRxn(add_atom_mapping=True)
rxns = transform_to_rxn(cgr_results)

print("RXNs:\n\t" + "\n\t".join(rxns))
```

```output
RXNs:
	[N:1]#[C:2][C@@:3]1([H:4])[C:5]([H:6])([H:7])[O:8]1>>[N:1]#[C:2][C@@:3]([H:4])([C:5][H:6])[O:8][H:7]
	[O:1]([C@@:2]([C:3](=[O:4])[H:5])([C:6]#[C:7][H:8])[H:9])[H:10]>>[O:1]([C@@:2]([C:3][O:4][H:5])([C:6]#[C:7][H:8])[H:9])[H:10]
```

What if your reactions aren’t atom‑mapped, and/or some are unbalanced? No problem, simply set `balance_rxn=True` and/or enable the integrated mapper with `mapping_method="rxn_mapper"`.
```python
# Example list of reactions
rxn_list = [
    "CCO>>CC=O",                # unmapped
    "N>>NC",                    # unmapped and unbalanced
    "[NH3:1]>>[NH2:1][CH3:2]",  # unbalanced
]

# using the wrapper with a pandas DataFrame
df_data = pd.DataFrame({"reaction": rxn_list})
transform_to_cgr_df = RxnToCgr(
    rxn_col="reaction",
    mapping_method="rxn_mapper",
    balance_rxn=True,
)
df_data["cgr_smiles"] = transform_to_cgr_df(df_data)
print("\nDataFrame with CGR-SMILES:\n", df_data)
```

```output
DataFrame with CGR-SMILES:
                   reaction                        cgr_smiles
0                CCO>>CC=O  [CH3]{[CH2]|[CH]}{-|=}{[OH]|[O]}
1                    N>>NC           {[NH3]|[NH2]}{~|-}[CH3]
2  [NH3:1]>>[NH2:1][CH3:2]           {[NH3]|[NH2]}{~|-}[CH3]
```


### 3. Command Line Interface

If your prefer working with a CLI tool , that be it:

```bash
╭────────── 🚀 CGR‑SMILES Converter v0.0.1 ─────────╮
│                                                   │
│   👋 Welcome to CGR‑SMILES                        │
│   Transforming Reaction SMILES ➡️ CGR‑SMILES       │
│                                                   │
│   Input column:   'rxn_smiles'                    │
│   Output column:  'cgr_smiles'                    │
│   Input file:     path/to/input.csv               │
│   Output file:    path/to/output.csv              │
│                                                   │
╰───────────────────────────────────────────────────╯
```
**The forward transformation:**
```bash
rxn2cgr path/to/input.csv \             # required (input CSV)
    -o path/to/output.csv \             # optional output CSV
    --rxn-col rxn_smiles \              # name of the RXN SMILES column
    --cgr-col cgr_smiles \              # name of the new CGR‑SMILES column
    --mapping-method rxn_mapper \       # mapping method (default: None)
    --keep-atom-mapping \               # preserve atom mapping
    --balance-rxn                       # enable reaction balancing

```


**And the backward transformation:**

```bash
cgr2rxn output_cgr.csv \                # required (input CSV)
    -o path/to/output.csv \             # optional output CSV
    --cgr-col cgr_smiles \              # name of the CGR‑SMILES column
    --rxn-col rxn_back                  # name of the new RXN SMILES column
```

---

## Contributing

We welcome contributions!
For development installation and guidelines, see [CONTRIBUTE.md](CONTRIBUTE.md).

---

## Citation

[WIP]

---

## References
[1] Heid, E.; Green, W. H. *Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction.*
   *J. Chem. Inf. Model.* **2022**, 62 (9), 2101–2110.
   DOI: [10.1021/acs.jcim.1c00975](https://doi.org/10.1021/acs.jcim.1c00975)

[2] Schwaller, P.; Hoover, B.; Reymond, J.‑L.; Strobelt, H.; Laino, T. *Extraction of Organic Chemistry Grammar from Unsupervised Learning of Chemical Reactions.*
   *Sci. Adv.* **2021**, 7 (15), eabe4166.
   DOI: [10.1126/sciadv.abe4166](https://doi.org/10.1126/sciadv.abe4166)
