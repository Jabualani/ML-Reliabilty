import json, os, sys, time, traceback, pathlib, shutil

root = pathlib.Path(__file__).resolve().parent
colab = root.parent / "docs" / "colab"
names = sys.argv[1:]
os.environ.setdefault("MPLBACKEND", "Agg")
status = {}
for name in names:
    nb = json.loads((colab / f"{name}.ipynb").read_text(encoding="utf-8"))
    run_dir = root / "run" / name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    os.chdir(run_dir)
    ns = {"__name__": "__main__"}
    t0 = time.time()
    ok = True
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        try:
            exec(compile(src, f"{name}[cell {i}]", "exec"), ns)
        except Exception:
            ok = False
            print(f"--- {name} FAILED in cell {i}")
            traceback.print_exc()
            break
    status[name] = (ok, round(time.time() - t0, 1))
    print(f"=== {name}: {'OK' if ok else 'FAIL'} in {status[name][1]}s")
