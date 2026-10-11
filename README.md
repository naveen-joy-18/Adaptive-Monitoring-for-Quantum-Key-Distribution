# Adaptive QKD monitoring — Qiskit research prototype

Runnable simulation package for time-resolved BB84 channel monitoring under
nonstationary errors: an exact Qiskit channel engine, Aer shot validation,
statistical baselines and predictable adaptive test allocation, with archived,
seeded results. It is a monitoring/benchmarking study, not a QKD security proof;
`certified_secret_key_bits` is always `null` and no keys are generated.

The package is self-contained: the archived results (`results_v3/`) and the
frozen calibration inputs (`calibration/`) ship with it. You only need
**Python 3.11** and the dependencies below.

## Requirements

- **Python 3.11.x** (built and tested on 3.11.9).
- Dependencies (`requirements.txt`): `qiskit==2.5.2`, `qiskit-aer==0.17.2`,
  `numpy>=2.0,<3`, `scipy>=1.14`, `pandas>=2.0`, `matplotlib>=3.8`.

## Reproduce

Run from the project root.

**1. Check if Python 3.11 is installed**

```powershell
py -3.11 --version        # Windows
```
```bash
python3.11 --version      # macOS / Linux
```

If it prints `Python 3.11.x`, skip to step 3.

**2. Install Python 3.11 if missing**

```powershell
winget install -e --id Python.Python.3.11                       # Windows
```
```bash
brew install python@3.11                                        # macOS
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev   # Debian/Ubuntu
```

Alternatively use the official 3.11.x installer from
<https://www.python.org/downloads/> (on Windows, tick "Add python.exe to PATH").
Reopen the terminal afterwards and repeat step 1.

**3. Create and activate a virtual environment**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1                                       # Windows
```
```bash
python3.11 -m venv .venv
source .venv/bin/activate                                        # macOS / Linux
```

If Windows blocks activation: run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then retry. You can
also skip activation and call the interpreter directly
(`.venv\Scripts\python.exe ...` or `.venv/bin/python ...`).

**4. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**5. Run the test suite (seconds)**

```bash
python -m unittest discover -s tests -v
```

Expect `Ran 13 tests` followed by `OK`.

**6. Reproduce the primary results (revision 3)**

`results_v3/` already ships with the package; back it up first if you want to
keep the shipped copy (`Copy-Item results_v3 results_v3_backup -Recurse` on
Windows, `cp -r results_v3 results_v3_backup` on macOS/Linux). Then:

```bash
python experiments/run_revision3.py
python experiments/plot_revision3.py
```

Writes 18 files and four figures into `results_v3/` in about a minute.
`results_v3/metadata.json` should show adaptive threshold `4.73487...`, fixed
threshold `6.23851...`, `fixed_monitor_probability` `0.0332158435788767`, and a
`security_status` of `NO QKD SECRECY PROOF`.

**7. Optional: verify file integrity**

`MANIFEST_SHA256.json` lists the SHA-256 of every shipped file. Save as
`verify_manifest.py` (local helper) and run `python verify_manifest.py`:

```python
import hashlib, json, pathlib
root = pathlib.Path(".")
manifest = json.loads((root / "MANIFEST_SHA256.json").read_text())
bad = [r for r, want in manifest.items()
       if not (root / r).exists()
       or hashlib.sha256((root / r).read_bytes()).hexdigest() != want]
print("OK: all", len(manifest), "files match" if not bad else f"MISMATCH: {bad}")
```

Verify before step 6; re-running step 6 changes a few files (`metadata.json`
`elapsed_seconds` and figure bytes), so those hashes will then differ.

## Troubleshooting

- **`python` not found** → use `python3.11` or `py -3.11` instead.
- **`No module named qiskit`** → the virtual environment is not active, or
  step 4 ran outside it.
- **`qiskit-aer` fails to install** → confirm 64-bit Python 3.11 and run
  `python -m pip install --upgrade pip`, then retry step 4.
- **Figures do not open on a headless machine** → set `MPLBACKEND=Agg` before
  step 6; figures are saved to `results_v3/` regardless.
