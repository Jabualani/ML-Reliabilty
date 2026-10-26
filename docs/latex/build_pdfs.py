from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SOURCES = (
    "overview.tex",
    "statistical-data-cleaning.tex",
    "attp-method.tex",
    "distribution-shift.tex",
    "evaluation.tex",
    "fairness.tex",
    "deterioration.tex",
    "meta-learning.tex",
    "explainability.tex",
    "reason-audit.tex",
    "pipeline.tex",
    "statcore-pca.tex",
    "statcore-lda.tex",
    "statcore-kmeans.tex",
    "statcore-linear.tex",
    "statcore-knn.tex",
    "statcore-mlp.tex",
    "statcore-meta.tex",
    "automl.tex",
    "automl-comparison.tex",
    "reliableml.tex",
    "data-intelligence.tex",
)


def main() -> None:
    source_directory = Path(__file__).resolve().parent
    build_directory = source_directory / "build"
    pdf_directory = source_directory.parent / "pdfs"
    tectonic = shutil.which("tectonic")
    pdflatex = shutil.which("pdflatex")
    if tectonic is None and pdflatex is None:
        raise RuntimeError("Install Tectonic or pdflatex before building the PDFs.")

    build_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        if tectonic is not None:
            subprocess.run(
                [tectonic, "--outdir", str(build_directory), source],
                cwd=source_directory,
                check=True,
            )
        else:
            if pdflatex is None:
                raise RuntimeError("Install Tectonic or pdflatex before building the PDFs.")
            command = [
                pdflatex,
                "-halt-on-error",
                "-interaction=nonstopmode",
                f"-output-directory={build_directory}",
                source,
            ]
            for _ in range(2):
                subprocess.run(command, cwd=source_directory, check=True)

        pdf_name = f"{Path(source).stem}.pdf"
        shutil.copy2(build_directory / pdf_name, pdf_directory / pdf_name)
        print(f"Created docs/pdfs/{pdf_name}")


if __name__ == "__main__":
    main()
