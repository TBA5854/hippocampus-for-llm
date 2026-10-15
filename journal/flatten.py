"""Write main_single.tex: main.tex with every \\input{sections/...} inlined, as NMI asks for
the final LaTeX submission (all text in one file). With --bbl, the compiled references.bib
output (main.bbl) is pasted in place of \\bibliographystyle/\\bibliography as well.

    python3 flatten.py [--bbl]       # run after a full pdflatex/bibtex build
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
s = (root / "main.tex").read_text()
s = re.sub(r"\\input\{(sections/[^}]+)\}", lambda m: (root / (m.group(1) + ".tex")).read_text().rstrip() + "\n", s)
if "--bbl" in sys.argv:
    bbl = (root / "main.bbl").read_text().rstrip()
    s = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", lambda m: bbl, s)
(root / "main_single.tex").write_text(s)
print("wrote main_single.tex", "(with .bbl)" if "--bbl" in sys.argv else "")
