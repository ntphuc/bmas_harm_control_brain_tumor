"""Figures of Online Resource 1, drawn from the pipeline outputs."""
import argparse, csv, json, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from experiments.common import load_case_table, regression_rates
_ap = argparse.ArgumentParser()
_ap.add_argument("--exp-out", required=True)
_ap.add_argument("--output-dir", required=True)
ARGS = _ap.parse_args()
R = ARGS.exp_out
OUT = Path(ARGS.output_dir) / "figs"
OUT.mkdir(parents=True, exist_ok=True)
VMAP = json.load(open(f"{R}/discover/variant_map.json"))
def _seed_dirs(v):
    return [d for d in VMAP["variants"].get(v, {}).get("seeds", {}).values()]
def _ms(vals):
    vals = [x for x in vals if x is not None]
    return (st.fmean(vals), st.stdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"axes.titlesize":8})
V="ABCDE"; C={"A":"#1f77b4","B":"#ff7f0e","C":"#2ca02c","D":"#d62728","E":"#9467bd"}
def save(fig,name):
    fig.tight_layout(pad=0.4); fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)
# S1 regression rates (main Table values)
rates = {v: [regression_rates(t, list(t)) for t in (load_case_table(d["per_case_csv"]) for d in _seed_dirs(v) if d.get("per_case_csv"))] for v in V}
mvr = [_ms([r["mvr"] for r in rates[v]]) for v in V]; bmvr = [_ms([r["bmvr"] for r in rates[v]]) for v in V]
ld = [_ms([r["dice_viol_e3e4"] for r in rates[v]]) for v in V]; lh = [_ms([r["hd95_viol_e3e4"] for r in rates[v]]) for v in V]
fig,ax=plt.subplots(1,2,figsize=(6.6,2.4)); import numpy as np; x=np.arange(5); w=0.38
for a,(p,q,l1,l2,t) in zip(ax,[(mvr,bmvr,"MVR (Dice)","BMVR (HD95)","All transitions"),(ld,lh,"Dice","HD95","E3→E4 transition")]):
    a.bar(x-w/2,[m for m,_ in p],w,yerr=[s for _,s in p],capsize=2,label=l1,color="#4c72b0")
    a.bar(x+w/2,[m for m,_ in q],w,yerr=[s for _,s in q],capsize=2,label=l2,color="#dd8452")
    a.set_xticks(x,list(V)); a.set_ylabel("Regression rate (%)"); a.set_title(t); a.set_ylim(0,max(m+s for m,s in q)*1.25); a.legend(frameon=False,ncol=2,loc="upper center"); a.grid(axis="y",alpha=.3,lw=.5)
save(fig,"figS1_rates")
# S2 regression vs AUC
rows={r["variant"]:r for r in csv.DictReader(open(f"{R}/analysis/x1_per_exit.csv"))}
fig,ax=plt.subplots(1,2,figsize=(6.6,2.5))
for a,k,l in ((ax[0],"mvr","MVR (%)"),(ax[1],"bmvr","BMVR (%)")):
    for v in V:
        r=rows[v]; a.errorbar(float(r["auc_gflops_dice_mean"]),float(r[f"{k}_mean"]),xerr=float(r["auc_gflops_dice_sd"]),yerr=float(r[f"{k}_sd"]),fmt="o",ms=4,capsize=2,color=C[v])
        a.annotate(v,(float(r["auc_gflops_dice_mean"]),float(r[f"{k}_mean"])),textcoords="offset points",xytext=(4,3),fontsize=7)
    a.set_xlabel("Normalized area under GFLOPs–Dice curve"); a.set_ylabel(l); a.grid(alpha=.3,lw=.5)
save(fig,"figS2_regression_vs_auc")
# S3 tolerance curves
tol=list(csv.DictReader(open(f"{R}/analysis/x6_tolerance.csv")))
fig,ax=plt.subplots(1,2,figsize=(6.6,2.5))
for a,m,xl in ((ax[0],"BMVR","HD95 tolerance τ_H (px)"),(ax[1],"MVR","Dice tolerance τ_D")):
    for v in V:
        pts=sorted([(float(r["tolerance"]),float(r["mean"])) for r in tol if r["metric"]==m and r["variant"]==v])
        a.plot([p[0] for p in pts],[p[1] for p in pts],"-o",ms=3,lw=1,color=C[v],label=v)
    a.set_xlabel(xl); a.set_ylabel(f"{m} (%)"); a.grid(alpha=.3,lw=.5)
    if m=="MVR": a.set_xscale("symlog",linthresh=0.001); a.set_xticks([0,0.0005,0.001,0.005,0.01],["0","5e-4","1e-3","5e-3","1e-2"])
ax[0].legend(frameon=False,ncol=5,loc="upper right")
save(fig,"figS3_tolerance")
# S4 BHR/BCR scatter
def _summ(v, key):
    vals = []
    for d in _seed_dirs(v):
        if d.get("summary") and Path(d["summary"]).is_file():
            x = json.load(open(d["summary"])).get(key)
            if x is not None:
                vals.append(float(x))
    return _ms(vals)
bhr = {v: _summ(v, "bhr") for v in V}; bcr = {v: _summ(v, "bcr") for v in V}
fig,a=plt.subplots(figsize=(3.3,2.5))
for v in V:
    a.errorbar(bhr[v][0],bcr[v][0],xerr=bhr[v][1],yerr=bcr[v][1],fmt="o",ms=4,capsize=2,color=C[v]); a.annotate(v,(bhr[v][0],bcr[v][0]),textcoords="offset points",xytext=(4,3))
a.set_xlabel("Boundary harm rate BHR (%) ↓"); a.set_ylabel("Boundary correction rate BCR (%) ↑"); a.grid(alpha=.3,lw=.5)
save(fig,"figS4_bhr_bcr")
# S5 subgroups
sg=list(csv.DictReader(open(f"{R}/analysis/x8_subgroups.csv")))
groups=[("Imaging plane","ax","Axial"),("Imaging plane","co","Coronal"),("Imaging plane","sa","Sagittal"),("Lesion size (tertiles)","small","Small"),("Lesion size (tertiles)","medium","Medium"),("Lesion size (tertiles)","large","Large")]
fig,ax=plt.subplots(1,2,figsize=(6.6,2.5)); x=np.arange(len(groups)); w=0.16
for a,k,l in ((ax[0],"mvr_mean","MVR (%)"),(ax[1],"bmvr_mean","BMVR (%)")):
    for i,v in enumerate(V):
        vals=[float([r for r in sg if r["grouping"]==g and r["group"]==gg and r["variant"]==v][0][k]) for g,gg,_ in groups]
        a.bar(x+(i-2)*w,vals,w,color=C[v],label=v)
    a.set_xticks(x,[n for _,_,n in groups],rotation=0); a.set_ylabel(l); a.grid(axis="y",alpha=.3,lw=.5)
ax[0].legend(frameon=False,ncol=5,fontsize=6.5,loc="upper right")
save(fig,"figS5_subgroups")
# S6 frontier full
fr=list(csv.DictReader(open(f"{R}/frontier/frontier.csv")))
ph={r["model"]:r for r in csv.DictReader(open(f"{R}/posthoc/x9_posthoc.csv"))}
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
for a,k,l in ((ax[0],"mvr","MVR (%)"),(ax[1],"bmvr","BMVR (%)")):
    b=sorted([r for r in fr if r["variant"]=="B"],key=lambda r:float(r["dice_mean"]))
    if not b: continue
    a.plot([float(r["dice_mean"]) for r in b],[float(r[f"{k}_mean"]) for r in b],"-o",ms=3,lw=1,color=C["B"],label="B, damping γ")
    for r in b: a.annotate(f"γ={float(r['gamma']):g}",(float(r["dice_mean"]),float(r[f"{k}_mean"])),fontsize=6,textcoords="offset points",xytext=(-12,5))
    ca=ph["B + cumulative averaging"]; a.plot(float(ca["dice_e4_mean"]),float(ca[f"{k}_mean"]),"^",ms=5,color="0.3",label="B, cumulative avg.")
    for v in "ACDE":
        r=[x for x in fr if x["variant"]==v and float(x["gamma"])==1][0]; a.plot(float(r["dice_mean"]),float(r[f"{k}_mean"]),"s",ms=4,color=C[v],label=v)
    a.set_xlabel("Final Dice (E4)"); a.set_ylabel(l); a.grid(alpha=.3,lw=.5)
ax[0].legend(frameon=False,fontsize=6.5,loc="upper left")
save(fig,"figS6_frontier_full")
print("ok")
