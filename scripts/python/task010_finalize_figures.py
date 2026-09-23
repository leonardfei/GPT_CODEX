#!/usr/bin/env python3
"""Create the remaining compact Task010 Phase A figures from saved metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

R=Path("/data/lf_data/result/task010_representation_benchmark")
F=R/"figures"; F.mkdir(exist_ok=True)

def main():
    plt.rcParams.update({"font.size":9,"figure.dpi":120})
    fig,ax=plt.subplots(figsize=(11,3.2)); ax.axis("off")
    steps=[("Task009 V3_CORE", "frozen labels\nexact held-out batches"),("CellViT", "official detection\n+ token extraction"),("Shared cohort", "96,044 cells\n7 classes / 8 batches"),("Phikon-v2", "75 px / 263 px\nlocal-only CLS"),("Probe", "linear + MLP\nfold-wise QC")]
    xs=np.linspace(.1,.9,len(steps))
    for i,(title,sub) in enumerate(steps):
        ax.text(xs[i],.58,title,ha="center",va="center",bbox=dict(boxstyle="round,pad=.7",fc="#dcecf7" if i<3 else "#e7f4df",ec="#4472c4"),fontsize=10,weight="bold")
        ax.text(xs[i],.30,sub,ha="center",va="center")
        if i<len(steps)-1: ax.annotate("",xy=(xs[i+1]-.07,.58),xytext=(xs[i]+.07,.58),arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.set_title("Task010 Phase A design; all representations use the same frozen cells and folds")
    fig.tight_layout(); fig.savefig(F/"Fig1_task010A_design.pdf"); plt.close(fig)

    flows=pd.read_csv(R/"metrics/confusion_flows.csv")
    small=flows[flows.representation=="PHIKON_V2_SMALL"].groupby(["source","target"],as_index=False).rate.mean()
    order=["Endothelial","Mesenchymal","Myeloid","Neutrophil","Plasma cell","T and B","Tumor"]
    m=small.pivot(index="source",columns="target",values="rate").reindex(index=order,columns=order).fillna(0)
    fig,ax=plt.subplots(figsize=(7,5)); sns.heatmap(m,annot=True,fmt=".2f",cmap="mako",vmin=0,vmax=max(.45,m.to_numpy().max()),ax=ax); ax.set_title("PHIKON_V2_SMALL mean validation confusion rate"); fig.tight_layout(); fig.savefig(F/"Fig6_neutrophil_confusions.pdf"); plt.close(fig)

    d=pd.read_csv(R/"metrics/representation_paired_deltas.csv"); q=d[d.metric=="macro_f1"]
    fig,ax=plt.subplots(figsize=(7,4)); sns.boxplot(data=q,x="comparison",y="delta",ax=ax,color="#8ecae6"); sns.stripplot(data=q,x="comparison",y="delta",ax=ax,color="black",size=4); ax.axhline(0,color="grey",lw=1); ax.tick_params(axis="x",rotation=20); ax.set_title("Paired held-out fold deltas"); fig.tight_layout(); fig.savefig(F/"Fig10_fold_paired_deltas.pdf"); plt.close(fig)

    # Use raw fold metrics for a clean comparison because the grouped summary
    # intentionally contains a two-row pandas aggregation header.
    raw=pd.read_csv(R/"metrics/gt_centered_upper_bound_fold_metrics.csv")
    g=raw.groupby("representation",as_index=False).macro_f1.mean(); shared=pd.read_csv(R/"metrics/linear_probe_summary.csv"); shared=shared[shared.representation.str.startswith("PHIKON")][["representation","macro_f1"]].copy(); shared["source"]="shared detected"; shared=shared.rename(columns={"macro_f1":"value"}); g=g.rename(columns={"representation":"label","macro_f1":"value"}); g["source"]="GT-centered"; g["label"]=g.label.str.replace("PHIKON_V2_GT_CENTERED_","PHIKON_V2_",regex=False); s=shared.rename(columns={"representation":"label"}); s["label"]=s.label.str.replace("PHIKON_V2_","PHIKON_V2_",regex=False); plot=pd.concat([s[["label","value","source"]],g[["label","value","source"]]],ignore_index=True)
    fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(data=plot,x="label",y="value",hue="source",ax=ax); ax.set_ylabel("Macro-F1"); ax.set_title("Shared detected cohort versus GT-centered morphology upper bound"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(F/"Fig11_gt_centered_upper_bound.pdf"); plt.close(fig)

if __name__=="__main__": main()
