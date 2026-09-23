#!/usr/bin/env python3
"""Task010 Phikon GT-centered morphology upper bound."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyvips
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

MAIN = Path("/data/lf_data/result/task010_representation_benchmark/code/task010_phikon_phase_a.py")
sys.path.insert(0, str(MAIN.parent))
import task010_phikon_phase_a as t


def build_all() -> pd.DataFrame:
    cells = pd.read_csv(t.DATASET / "cell_to_patch.csv")
    patches = pd.read_csv(t.DATASET / "patch_metadata.csv")
    cells = cells[(cells.condition == "CORE") & (cells.split == "train")].copy()
    patches = patches[(patches.condition == "CORE") & (patches.split == "train")].copy()
    pm = patches[["image", "source_image", "patch_id", "batch", "x", "y"]]
    d = cells.merge(pm, left_on=["image", "patch_id", "batch"], right_on=["source_image", "patch_id", "batch"], how="inner", suffixes=("_cell", ""))
    d["patch_x"] = d.x.astype(int); d["patch_y"] = d.y.astype(int)
    for side, name in [(75, "small"), (263, "context")]:
        half = side // 2
        d[f"{name}_inbounds"] = ((d.he_x-half >= 0) & (d.he_y-half >= 0) & (d.he_x+half < 50000) & (d.he_y+half < 23451))
    d = d[d.small_inbounds & d.context_inbounds].copy().sort_values(["patch_id", "cell_id"], kind="stable").reset_index(drop=True)
    if d.cell_id.duplicated().any() or len(d) == 0: raise RuntimeError("Invalid GT-centered cohort")
    keep=["cell_id","batch","patch_id","class_id","class_name","original_cl1","v3_label","he_x","he_y","patch_x","patch_y","small_inbounds","context_inbounds"]
    d[keep].to_csv(t.ROOT/"metrics/gt_centered_manifest.csv.gz",index=False,compression="gzip")
    return d


def features(d: pd.DataFrame) -> None:
    from transformers import AutoImageProcessor, AutoModel
    proc=AutoImageProcessor.from_pretrained(str(t.PHIKON),local_files_only=True)
    model=AutoModel.from_pretrained(str(t.PHIKON),local_files_only=True).eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    wsi=pyvips.Image.new_from_file(str(t.WSI),access="random")
    small=[]; context=[]; ims=[]; imc=[]; tile=None; ox=oy=0; current=None; t0=time.time()
    def flush():
        if not ims:return
        with torch.inference_mode():
            inp=proc(images=ims,return_tensors="pt"); inp={k:v.to(device) for k,v in inp.items()}
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"): z=model(**inp).last_hidden_state[:,0,:].float().cpu().numpy()
            small.extend(z)
            inp=proc(images=imc,return_tensors="pt"); inp={k:v.to(device) for k,v in inp.items()}
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"): z=model(**inp).last_hidden_state[:,0,:].float().cpu().numpy()
            context.extend(z)
    for i,r in d.iterrows():
        if current != r.patch_id:
            tile,ox,oy=t._tile_array(wsi,int(r.patch_x),int(r.patch_y)); current=r.patch_id
        for side,arrs in [(75,ims),(263,imc)]:
            half=side//2; sx=int(round(r.he_x))-half-ox; sy=int(round(r.he_y))-half-oy; arr=tile[sy:sy+side,sx:sx+side,:3]
            if arr.shape != (side,side,3): raise RuntimeError(f"bad {arr.shape} {r.cell_id}")
            arrs.append(Image.fromarray(arr,mode="RGB"))
        if len(ims)>=128: flush(); ims.clear(); imc.clear()
        if i%1000==0: print(f"GT cells={i}/{len(d)} elapsed_min={(time.time()-t0)/60:.1f}",flush=True)
    flush()
    a=np.stack(small).astype(np.float32); b=np.stack(context).astype(np.float32)
    if len(a)!=len(d) or not np.isfinite(a).all() or not np.isfinite(b).all(): raise RuntimeError("GT feature QC failed")
    torch.save({"features":torch.from_numpy(a),"cell_ids":d.cell_id.tolist(),"condition":"GT_CENTERED_PHIKON_V2_SMALL","embedding_dim":a.shape[1]},t.ROOT/"features/phikon_v2_gt_centered_small.pt")
    torch.save({"features":torch.from_numpy(b),"cell_ids":d.cell_id.tolist(),"condition":"GT_CENTERED_PHIKON_V2_CONTEXT","embedding_dim":b.shape[1]},t.ROOT/"features/phikon_v2_gt_centered_context.pt")


def benchmark(d: pd.DataFrame) -> None:
    folds=pd.read_csv(t.TASK009/"metrics/split_manifest.csv"); folds=folds[folds.condition=="CORE"].sort_values("fold")
    y=d.class_id.to_numpy(int); reps={"PHIKON_V2_GT_CENTERED_SMALL":torch.load(t.ROOT/"features/phikon_v2_gt_centered_small.pt",map_location="cpu")["features"].numpy(),"PHIKON_V2_GT_CENTERED_CONTEXT":torch.load(t.ROOT/"features/phikon_v2_gt_centered_context.pt",map_location="cpu")["features"].numpy()}; rows=[]
    for rep,X in reps.items():
        for _,r in folds.iterrows():
            trb=set(str(r.train_batches).split(";")); vab=set(str(r.val_batches).split(";")); tr=np.flatnonzero(d.batch.astype(str).isin(trb)); va=np.flatnonzero(d.batch.astype(str).isin(vab)); p,_=t.fit_linear(X,y,tr,va); m=t.multiclass_metrics(y[va],p); m.update({"representation":rep,"fold":int(r.fold),"n_train":len(tr),"n_val":len(va)}); rows.append(m)
    out=pd.DataFrame(rows); out.to_csv(t.ROOT/"metrics/gt_centered_upper_bound_fold_metrics.csv",index=False); out.groupby("representation",as_index=False).agg({c:["mean","std"] for c in ["accuracy","balanced_accuracy","macro_f1","macro_auprc","macro_auroc","weighted_f1","mcc","lowest_three_f1"]}).to_csv(t.ROOT/"metrics/gt_centered_upper_bound.csv",index=False)
    (t.ROOT/"metrics/gt_centered_status.json").write_text(json.dumps({"status":"COMPLETED","n_cells":int(len(d)),"source":"Task009 V3_CORE train cell_to_patch registered centers without CellViT detection filter","production_model_modified":False},indent=2)+"\n")
    print(out.groupby("representation")[["macro_f1","macro_auprc"]].mean().to_string(),flush=True)


def main():
    t.mkdirs(); d=build_all(); features(d); benchmark(d)
if __name__=="__main__":main()
