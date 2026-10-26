"""Executes the code cells of the generated notebooks locally (skips ! and % magics).

Usage: python docs/colab/_run_local.py statcore-pca [statcore-lda ...]
Uses the Agg matplotlib backend; the last expression of a cell is printed like a notebook would.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent


def run(name: str) -> bool:
    nb = json.loads((HERE / f"{name}.ipynb").read_text(encoding="utf-8"))
    env: dict = {"__name__": "__main__"}
    env["display"] = print
    t0 = time.time()
    for n, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = [ln for ln in "".join(cell["source"]).split("\n")
                 if not ln.lstrip().startswith(("!", "%"))]
        src = "\n".join(lines)
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, f"{name}[cell {n}]", "exec"), env)
            if last is not None:
                val = eval(compile(last, f"{name}[cell {n}]", "eval"), env)
                if val is not None:
                    print(val)
        except Exception:
            print(f"FAILED {name} cell {n}")
            traceback.print_exc()
            return False
    print(f"OK {name} in {time.time() - t0:.1f}s")
    return True


if __name__ == "__main__":
    ok = all([run(n) for n in sys.argv[1:]])
    sys.exit(0 if ok else 1)
