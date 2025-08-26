
<!-- Adjust when put on github and pypi -->

<!-- <div align="center">

[![Build Status](https://github.com/heid-lab/cgr-smiles/actions/workflows/tests.yml/badge.svg)](https://github.com/heid-lab/cgr-smiles/actions)
[![Coverage](https://codecov.io/gh/heid-lab/cgr-smiles/branch/main/graph/badge.svg)](https://codecov.io/gh/heid-lab/cgr-smiles)

[![License](https://img.shields.io/github/license/heid-lab/cgr-smiles)](https://github.com/heid-lab/cgr-smiles/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cgr-smiles.svg)](https://pypi.org/project/cgr-smiles/)
[![Python versions](https://img.shields.io/pypi/pyversions/cgr-smiles.svg)](https://pypi.org/project/cgr-smiles)
[![Downloads](https://img.shields.io/github/downloads/heid-lab/cgr-smiles/total.svg)](https://github.com/heid-lab/cgr-smiles/releases)

</div> -->

[Installation](#installation) | [Usage](#usage) | [Contributing](#contributing) | [Citation](#citation)

</div>

# CGR-SMILES

**CGR-SMILES** is a Python library for transforming reaction SMILES into a more compact and change-aware representation called **CGR-SMILES**. This representation highlights local structural modifications in chemical reactions, making it more suitable for machine learning and data-driven applications.

TODO: Examples

---

## Installation

```bash
pip install cgr-smiles
```

---

## Usage

You can use the library in three different ways:

### 1. Import functions directly
```python
from cgr_smiles import to_cgr

rxn = "CCO>>CC=O"
cgr = to_cgr(rxn)
print(cgr)  # Example output: C[C;:0][O>>=O]
```

### 2. Use the CLI
```bash
cgr-smiles "CCO>>CC=O"
# Output: C[C;:0][O>>=O]
```

### 3. Use the class interface
```python
from cgr_smiles import CGRConverter

converter = CGRConverter()
rxn = "CCO>>CC=O"
cgr = converter.transform(rxn)
print(cgr)
```

---

## Contributing

We welcome contributions!
For development installation and guidelines, see [CONTRIBUTE.md](CONTRIBUTE.md).

---

## Citation

TODO
