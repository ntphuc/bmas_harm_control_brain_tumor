#!/usr/bin/env bash
# Online Resource 1 (ESM) and the pipeline-generated figures of the main paper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EXP_OUT="${EXP_OUT:-outputs/revision}"
ESM_OUT="${ESM_OUT:-outputs/esm}"
"$PY" -m experiments.esm_extras --exp-out "$EXP_OUT" --output "$ESM_OUT/esm_extra.json"
"$PY" -m experiments.make_esm_figs --exp-out "$EXP_OUT" --output-dir "$ESM_OUT"
"$PY" -m experiments.make_esm --exp-out "$EXP_OUT" --extras "$ESM_OUT/esm_extra.json" \
  --competitors outputs/competitors --gpu-profile outputs/profiling_anytime_e/profile_summary_3seeds.csv \
  --output-dir "$ESM_OUT"
"$PY" -m experiments.make_paper_figs --exp-out "$EXP_OUT" --output-dir outputs/paper_figures
cp -f "$EXP_OUT"/figures/fig4_qualitative.* outputs/paper_figures/ 2>/dev/null || true
if command -v pdflatex >/dev/null; then
  (cd "$ESM_OUT" && pdflatex -interaction=nonstopmode ESM_1.tex >/dev/null && pdflatex -interaction=nonstopmode ESM_1.tex >/dev/null) \
    && echo "ESM PDF: $ESM_OUT/ESM_1.pdf" || echo "pdflatex failed; compile $ESM_OUT/ESM_1.tex manually"
else
  echo "pdflatex not found; compile $ESM_OUT/ESM_1.tex manually (e.g. on Overleaf)"
fi
