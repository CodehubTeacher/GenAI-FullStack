"""Execute every course lab notebook and report which ones still work.

Each notebook runs with its own directory as the working directory, because the labs
load credentials via load_dotenv("config.env") from a file sitting beside them and
reference data files (util.py, Table_Reconstruction.pdf, ...) by relative path.

In-notebook `!pip install` cells are neutralized in the in-memory copy only -- several
would otherwise downgrade the pinned environment mid-suite. Notebooks on disk are never
modified; executed copies go to a separate output directory.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".ipynb_checkpoints", "__pycache__", "node_modules"}
INSTALL_RE = re.compile(r"^\s*[!%]\s*(pip|conda)\s+install", re.M)


def find_notebooks(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.ipynb")
        if not any(part in SKIP_DIRS for part in p.parts)
    )


def suppress_installs(nb) -> int:
    """Comment out pip/conda install lines. Returns how many cells were touched."""
    touched = 0
    for cell in nb.cells:
        if cell.cell_type != "code" or not INSTALL_RE.search(cell.source):
            continue
        cell.source = "\n".join(
            f"# [harness] suppressed: {ln}" if INSTALL_RE.match(ln) else ln
            for ln in cell.source.splitlines()
        )
        touched += 1
    return touched


def first_error(nb) -> dict | None:
    """Find the first cell whose outputs contain an error."""
    for idx, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                return {
                    "cell_index": idx,
                    "ename": out.get("ename", ""),
                    "evalue": out.get("evalue", ""),
                    "traceback": "\n".join(out.get("traceback", [])[:20]),
                }
    return None


def run_one(path: Path, out_dir: Path, timeout: int) -> dict:
    rel = path.relative_to(ROOT)
    nb = nbformat.read(path, as_version=4)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    suppressed = suppress_installs(nb)

    result = {
        "notebook": str(rel),
        "module": rel.parts[0],
        "code_cells": code_cells,
        "installs_suppressed": suppressed,
    }

    if code_cells == 0:
        result |= {"status": "empty", "seconds": 0.0}
        return result

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,  # capture the error rather than raising, so we can report it
        resources={"metadata": {"path": str(path.parent)}},
    )

    start = time.monotonic()
    try:
        client.execute()
        err = first_error(nb)
        result["status"] = "fail" if err else "pass"
        if err:
            result["error"] = err
    except CellExecutionError as e:
        result |= {"status": "fail", "error": {"ename": type(e).__name__, "evalue": str(e)[:800]}}
    except Exception as e:  # kernel death, timeout, ...
        result |= {"status": "error", "error": {"ename": type(e).__name__, "evalue": str(e)[:800]}}
    result["seconds"] = round(time.monotonic() - start, 1)

    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dest)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", help="only run notebooks whose path contains this substring")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    ap.add_argument("--out", default="/private/tmp/claude-501/lab-run", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    notebooks = find_notebooks(ROOT)
    if args.filter:
        notebooks = [p for p in notebooks if args.filter.lower() in str(p).lower()]

    if not notebooks:
        print("No notebooks matched.")
        return 1

    print(f"Running {len(notebooks)} notebook(s)\n")
    results = []
    for i, nb_path in enumerate(notebooks, 1):
        rel = nb_path.relative_to(ROOT)
        print(f"[{i}/{len(notebooks)}] {rel} ... ", end="", flush=True)
        res = run_one(nb_path, out_dir, args.timeout)
        results.append(res)
        mark = {"pass": "PASS", "fail": "FAIL", "empty": "EMPTY", "error": "ERROR"}[res["status"]]
        print(f"{mark} ({res['seconds']}s)")
        if res.get("error"):
            e = res["error"]
            print(f"      {e.get('ename')}: {e.get('evalue', '')[:200]}")

    report = out_dir / "report.json"
    report.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for r in results:
        print(f"  {r['status'].upper():6} {r['notebook']}")
    print("=" * 78)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"\nReport: {report}")
    return 0 if counts.get("fail", 0) + counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
