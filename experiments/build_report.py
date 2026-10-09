"""Assemble REVISION_RESULTS.md from every stage's output and the stage status log."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import md_table, write_md

ITEMS = [
    ("V1", "Official split / tumor type", "v1/split_contingency.md", "Section 3.1, Data availability"),
    ("V2", "Image-level paired statistics", "analysis/FROZEN_ANALYSES.md#V2", "Sections 5.5, 6.1, 7.1"),
    ("V3", "Empty-prediction rule", "analysis/FROZEN_ANALYSES.md#V3", "Section 3.1"),
    ("V4", "Exit paths and notation", "v4/v4_exit_graph.md", "Section 4.1"),
    ("X1", "Per-exit quality A–E", "analysis/FROZEN_ANALYSES.md#X1", "Table 2"),
    ("X2", "Conditioning on intermediate quality", "analysis/FROZEN_ANALYSES.md#X2", "Section 7.2"),
    ("X3/X4", "Severity and regret", "analysis/FROZEN_ANALYSES.md#X3", "Table 6"),
    ("X5", "New components at E4", "x5/X5_COMPONENTS.md", "Section 7.4"),
    ("X6", "Tolerance sensitivity", "analysis/FROZEN_ANALYSES.md#X6", "Section 7.5"),
    ("X7", "Qualitative figure", "figures/fig4_qualitative.png", "Fig. 4"),
    ("X8", "Subgroups incl. tumor type", "analysis/FROZEN_ANALYSES.md#X8", "Section 7.8"),
    ("X9", "Post-hoc smoothing vs E", "posthoc/POSTHOC_STOPPING.md#X9", "Table 7"),
    ("X10", "Early stopping", "posthoc/POSTHOC_STOPPING.md#X10", "Table 8"),
    ("X9b", "Damping frontier, all variants", "frontier/FRONTIER.md", "Section 7.6, Fig. 6"),
    ("X14", "External set, zero-shot", "external/eval/EXTERNAL.md", "New Section 7.9"),
    ("X11", "CPU latency", "cpu/CPU_PROFILE.md", "Table 4"),
    ("X12", "nnU-Net 2-D", "nnunet/eval/NNUNET.md", "Table 1"),
]
NOT_IN_PIPELINE = [
    ("V5", "Repository URL — check by hand."),
    ("V6", "Description of Jazbec et al. [13] — read the paper."),
    ("V7", "Equations for L_pres, L_corr, L_tail — the D/E loss code is not in this package; copy them from the "
           "original BMAS training repository."),
    ("V8", "Bibliographic details of BRISC [1] — check by hand."),
    ("X13", "Extra seeds for B and E — needs the original BMAS training script (RUN_NOGATE_3SEEDS.sh), which "
            "is not in this package."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-out", required=True)
    args = ap.parse_args()
    root = Path(args.exp_out)
    status = {}
    st = root / "stage_status.tsv"
    if st.is_file():
        for line in st.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                status[parts[0]] = (parts[1], parts[2])

    idx_rows = []
    for item, title, rel, where in ITEMS:
        path = root / rel.split("#")[0]
        idx_rows.append([item, title, where, "available" if path.exists() else "missing", rel])
    parts = ["# BMAS revision experiments — results\n",
             "## Stage status\n\n" + md_table(["Stage", "Status", "Note"],
                                             [[k, v[0], v[1]] for k, v in status.items()]),
             "## Where each item lands in the manuscript\n\n" + md_table(
                 ["ID", "Item", "Manuscript", "Output", "File"], idx_rows),
             "## Not covered by this pipeline\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in NOT_IN_PIPELINE)]
    for rel in ("discover/discover.log.md", "v1/split_contingency.md", "v4/v4_exit_graph.md",
                "analysis/FROZEN_ANALYSES.md", "x5/X5_COMPONENTS.md", "posthoc/POSTHOC_STOPPING.md", "frontier/FRONTIER.md", "external/eval/EXTERNAL.md",
                "cpu/CPU_PROFILE.md", "nnunet/eval/NNUNET.md"):
        p = root / rel
        if p.is_file():
            parts.append(f"---\n\n<!-- {rel} -->\n\n" + p.read_text(encoding="utf-8"))
    figs = sorted((root / "figures").glob("*.png")) if (root / "figures").is_dir() else []
    if figs:
        parts.append("---\n\n## Figures\n\n" + "\n".join(f"- {f.relative_to(root)}" for f in figs))
    write_md(root / "REVISION_RESULTS.md", "\n\n".join(parts))
    print(f"Wrote {root / 'REVISION_RESULTS.md'}")


if __name__ == "__main__":
    main()
