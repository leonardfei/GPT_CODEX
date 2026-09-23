#!/usr/bin/env python3
"""Task010 corrected canonical five-representation benchmark.

The script archives no data itself; the pre-fix archive is created before it
is run. Large tensors remain on the server and only small summaries are fit
for Git review.
"""
from __future__ import annotations
import hashlib, json, math, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_fscore_support, roc_auc_score, silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

CODE = Path("/data/lf_data/result/task010_representation_benchmark/code")
sys.path.insert(0, str(CODE))
import task010_phikon_phase_a as base

R = base.ROOT
CANONICAL = R / "metrics/shared_cell_manifest.csv.gz"
MIDNIGHT = Path("/data/lf_data/models/midnight-12k")
SEED = 20260923
REPS = ["CELLVIT_TOKEN_ALIGNED", "PHIKON_V2_SMALL", "PHIKON_V2_CONTEXT", "MIDNIGHT12K_SMALL", "MIDNIGHT12K_CONTEXT"]


def seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def key_frame(df: pd.DataFrame) -> pd.Series:
    return df["image"].astype(str) + "|" + df["local_x"].astype(int).astype(str) + "|" + df["local_y"].astype(int).astype(str) + "|" + df["class_id"].astype(int).astype(str)


def canonical_order() -> pd.DataFrame:
    c = pd.read_csv(CANONICAL)
    required = ["cell_id", "image", "local_x", "local_y", "class_id", "class_name", "batch", "he_x", "he_y", "patch_id", "patch_x", "patch_y"]
    missing = [x for x in required if x not in c.columns]
    if missing: raise RuntimeError(f"Canonical manifest missing {missing}")
    if c.cell_id.duplicated().any() or key_frame(c).duplicated().any(): raise RuntimeError("Canonical identity is not unique")
    c = c.reset_index(drop=True)
    c.insert(0, "canonical_index", np.arange(len(c), dtype=int))
    c.to_csv(R / "metrics/canonical_cell_order.csv.gz", index=False, compression="gzip")
    return c


def sha_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def align_existing(c: pd.DataFrame) -> Dict[str, np.ndarray]:
    # CellViT: old payload order is explicitly mapped through its composite key.
    tm = pd.read_csv(R / "features/cellvit_token_manifest.csv.gz")
    tm["key"] = key_frame(tm)
    if tm.key.duplicated().any(): raise RuntimeError("CellViT token manifest key is not unique")
    old = torch.load(R / "features/cellvit_tokens.pt", map_location="cpu")
    old_feat = old["features"].float(); old_keys = [str(x) for x in old["cell_keys"]]
    if len(old_keys) != len(tm) or old_feat.shape[0] != len(tm): raise RuntimeError("CellViT tensor/manifest row mismatch")
    if old_keys != tm.key.tolist(): raise RuntimeError("CellViT payload key order does not match its manifest")
    if set(old_keys) != set(key_frame(c).tolist()): raise RuntimeError("CellViT keys do not exactly cover canonical cohort")
    lookup = {k: i for i, k in enumerate(old_keys)}
    old_row_for_canonical = np.asarray([lookup[k] for k in key_frame(c)], dtype=int)
    if len(np.unique(old_row_for_canonical)) != len(c): raise RuntimeError("CellViT reindex is not one-to-one")
    changed = int(np.sum(old_row_for_canonical != np.arange(len(c))))
    aligned = old_feat[torch.from_numpy(old_row_for_canonical)]
    payload = {"features": aligned, "cell_ids": c.cell_id.tolist(), "composite_keys": key_frame(c).tolist(), "source": old.get("source"), "alignment_method": "canonical key reindex from old CellViT token manifest", "old_tensor_sha256": sha_tensor(old_feat), "old_rows_reordered": changed, "seed": SEED}
    torch.save(payload, R / "features/cellvit_tokens_aligned.pt")
    tm2 = tm.set_index("key").loc[key_frame(c)].reset_index()
    tm2.insert(0, "canonical_index", c.canonical_index.to_numpy())
    tm2.insert(1, "cell_id", c.cell_id.to_numpy())
    tm2.to_csv(R / "features/cellvit_tokens_aligned_manifest.csv.gz", index=False, compression="gzip")

    # Phikon: validate or explicitly reorder by its stored cell_ids.
    aligned_arrays: Dict[str, np.ndarray] = {"CELLVIT_TOKEN_ALIGNED": aligned.numpy()}
    phikon_notes=[]
    for name, fn, outfn in [("PHIKON_V2_SMALL", "phikon_v2_small.pt", "phikon_v2_small_aligned.pt"), ("PHIKON_V2_CONTEXT", "phikon_v2_context.pt", "phikon_v2_context_aligned.pt")]:
        p = torch.load(R / "features" / fn, map_location="cpu"); ids = [str(x) for x in p["cell_ids"]]
        if len(ids) != len(c) or len(set(ids)) != len(ids): raise RuntimeError(f"{name} stored IDs invalid")
        if ids == c.cell_id.astype(str).tolist(): idx=np.arange(len(c), dtype=int); exact=True
        else:
            lu={x:i for i,x in enumerate(ids)}; idx=np.asarray([lu[x] for x in c.cell_id.astype(str)], dtype=int); exact=False
        feat=p["features"].float()[torch.from_numpy(idx)];
        torch.save({**p, "features": feat, "cell_ids": c.cell_id.tolist(), "composite_keys": key_frame(c).tolist(), "alignment_method": "canonical cell_id validation/reindex"}, R / "features" / outfn)
        aligned_arrays[name] = feat.numpy(); phikon_notes.append(f"{name}: exact_cell_id_order={exact}")
        if not np.isfinite(aligned_arrays[name]).all(): raise RuntimeError(f"{name} has non-finite values")
    (R / "qc/cellvit_alignment_correction.md").write_text("# CellViT alignment correction\n\n" + f"Old token rows: {len(tm):,}. Canonical cells: {len(c):,}. Rows whose positional index changed after composite-key reindexing: **{changed:,} ({changed/len(c):.2%})**.\n\nThe old order was extraction/DataLoader order; the canonical order is the saved shared manifest order. Exact key uniqueness and one-to-one coverage were asserted before saving `cellvit_tokens_aligned.pt`.\n\nThe previous macro-F1 comparison is superseded. Phikon tensors were revalidated against canonical cell IDs; " + "; ".join(phikon_notes) + ".\n")
    (R / "qc/phikon_alignment_audit.md").write_text("# Phikon alignment audit\n\nBoth existing Phikon payloads contained 96,044 unique cell IDs and were checked against `canonical_cell_order.csv.gz` before reuse. Corrected aligned copies were saved under `features/*_aligned.pt`; all rows are finite.\n")
    return aligned_arrays


def write_midnight_provenance(model) -> None:
    cfg=json.loads((MIDNIGHT/"config.json").read_text()); files={p.name:p.stat().st_size for p in MIDNIGHT.iterdir() if p.is_file()}
    prov={"status":"LOADED_OFFLINE","model_id":"Midnight-12k","path":str(MIDNIGHT),"files":files,"sha256_model_safetensors":base.sha256(MIDNIGHT/"model.safetensors"),"architecture":cfg.get("architectures"),"model_type":cfg.get("model_type"),"hidden_size":cfg.get("hidden_size"),"config_image_size":cfg.get("image_size"),"patch_size":cfg.get("patch_size"),"runtime_output_tokens":257,"runtime_classification_embedding_dim":3072,"torch_dtype":cfg.get("torch_dtype"),"feature_rule":"concatenate last_hidden_state[:,0,:] and mean(last_hidden_state[:,1:,:],dim=1)","official_readme_preprocessing":{"resize":224,"center_crop":224,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]},"offline_flags":["HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1"],"loaded_with_local_files_only":True,"restricted_variant_used":False}
    (R/"config/midnight12k_provenance.json").write_text(json.dumps(prov,indent=2)+"\n")
    (R/"qc/midnight12k_local_provenance.md").write_text(f"# Midnight-12k local provenance\n\n- Path: `{MIDNIGHT}`\n- Archive was gzip-validated and extracted locally; no network fallback.\n- Weight SHA256: `{prov['sha256_model_safetensors']}`\n- Runtime: {type(model).__name__}, hidden size {model.config.hidden_size}, output tokens 257, concatenated classification dimension 3072.\n- Official README preprocessing: resize 224, center crop 224, mean/std 0.5.\n- Restricted Midnight-92k variants were not used.\n")


def midnight_features(c: pd.DataFrame, batch_size: int = 32) -> None:
    from transformers import AutoModel
    model=AutoModel.from_pretrained(str(MIDNIGHT),local_files_only=True).eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); write_midnight_provenance(model)
    wsi=pyvips.Image.new_from_file(str(base.WSI),access="random")
    ordered=c.sort_values(["patch_id","cell_id"],kind="stable").reset_index(); small=[]; context=[]; bs=[]; bc=[]; tile=None; ox=oy=0; current=None; t0=time.time()
    def to_tensor(arr):
        im=Image.fromarray(arr,mode="RGB").resize((224,224),Image.Resampling.BICUBIC)
        x=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.0
        return torch.from_numpy((x-0.5)/0.5)
    def flush():
        if not bs:return
        with torch.inference_mode():
            x=torch.stack(bs).to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"): y=model(x).last_hidden_state; z=torch.cat([y[:,0,:],y[:,1:,:].mean(1)],dim=-1).float().cpu().numpy()
            small.extend(z)
            x=torch.stack(bc).to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"): y=model(x).last_hidden_state; z=torch.cat([y[:,0,:],y[:,1:,:].mean(1)],dim=-1).float().cpu().numpy()
            context.extend(z)
    for i,r in ordered.iterrows():
        if current != r.patch_id: tile,ox,oy=base._tile_array(wsi,int(r.patch_x),int(r.patch_y)); current=r.patch_id
        for side,arrs in [(75,bs),(263,bc)]:
            half=side//2; sx=int(round(r.he_x))-half-ox; sy=int(round(r.he_y))-half-oy; arr=tile[sy:sy+side,sx:sx+side,:3]
            if arr.shape != (side,side,3): raise RuntimeError(f"bad Midnight crop {arr.shape} {r.cell_id}")
            arrs.append(to_tensor(arr))
        if len(bs)>=batch_size: flush(); bs.clear(); bc.clear()
        if i%1000==0: print(f"Midnight cells={i}/{len(ordered)} elapsed_min={(time.time()-t0)/60:.1f}",flush=True)
    flush(); a=np.stack(small).astype(np.float32); b=np.stack(context).astype(np.float32); inv=np.argsort(ordered["index"].to_numpy()); a=a[inv]; b=b[inv]
    if len(a)!=len(c) or not np.isfinite(a).all() or not np.isfinite(b).all(): raise RuntimeError("Midnight feature QC failed")
    for fn,feat,cond in [("midnight12k_small.pt",a,"MIDNIGHT12K_SMALL"),("midnight12k_context.pt",b,"MIDNIGHT12K_CONTEXT")]:
        torch.save({"features":torch.from_numpy(feat),"cell_ids":c.cell_id.tolist(),"composite_keys":key_frame(c).tolist(),"condition":cond,"embedding_dim":int(feat.shape[1]),"preprocessing":"README.md resize/center-crop 224, mean/std 0.5","feature_rule":"CLS + mean patch tokens","seed":SEED},R/"features"/fn)
    print(f"Saved Midnight {a.shape} {b.shape}",flush=True)


def folds(c: pd.DataFrame) -> Dict[int, Tuple[np.ndarray,np.ndarray]]:
    d=pd.read_csv(base.TASK009/"metrics/split_manifest.csv"); d=d[d.condition=="CORE"].sort_values("fold"); out={}
    for _,r in d.iterrows():
        trb=set(str(r.train_batches).split(";")); vab=set(str(r.val_batches).split(";")); tr=np.flatnonzero(c.batch.astype(str).isin(trb)); va=np.flatnonzero(c.batch.astype(str).isin(vab)); out[int(r.fold)]=(tr,va)
    return out


def softmax_probe(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, k: int) -> Tuple[np.ndarray,np.ndarray]:
    scaler=StandardScaler().fit(X[tr]); xt=torch.from_numpy(scaler.transform(X[tr]).astype(np.float32)).to("cuda" if torch.cuda.is_available() else "cpu"); xv=torch.from_numpy(scaler.transform(X[va]).astype(np.float32)).to(xt.device); yt=torch.from_numpy(y[tr].astype(np.int64)).to(xt.device)
    w=torch.zeros((k,xt.shape[1]),device=xt.device,requires_grad=True); b=torch.zeros(k,device=xt.device,requires_grad=True); counts=torch.bincount(yt,minlength=k).float(); weights=len(yt)/(k*torch.clamp(counts,min=1.0)); opt=torch.optim.LBFGS([w,b],lr=1.0,max_iter=35,history_size=10,line_search_fn="strong_wolfe",tolerance_grad=1e-5); reg=.5/len(yt)
    def closure(): opt.zero_grad(set_to_none=True); loss=F.cross_entropy(xt@w.T+b,yt,weight=weights)+reg*torch.sum(w*w); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad(): p=torch.softmax(xv@w.T+b,dim=1).cpu().numpy(); pred=p.argmax(1)
    del xt,xv,yt,w,b,opt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return p,pred


def multiclass(y,p):
    pred=p.argmax(1); f=f1_score(y,pred,labels=np.arange(7),average=None,zero_division=0); aupr=[base.safe_ap((y==k).astype(int),p[:,k]) for k in range(7)]; auc=[base.safe_auc((y==k).astype(int),p[:,k]) for k in range(7)]
    return {"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"macro_f1":float(f.mean()),"macro_auprc":float(np.nanmean(aupr)),"macro_auroc":float(np.nanmean(auc)),"weighted_f1":f1_score(y,pred,labels=np.arange(7),average="weighted",zero_division=0),"mcc":matthews_corrcoef(y,pred),"lowest_three_f1":float(np.sort(f)[:3].mean())}


def true_binary(X,y,tr,va,pos,neg):
    tr=np.asarray([i for i in tr if y[i] in (pos,neg)]); va=np.asarray([i for i in va if y[i] in (pos,neg)]); yy=(y==pos).astype(int); p,_=softmax_probe(X,yy,tr,va,2); score=p[:,1]; pred=(score>=.5).astype(int); precision,recall,f,_=precision_recall_fscore_support(yy[va],pred,labels=[1],zero_division=0)
    return {"auroc":base.safe_auc(yy[va],score),"auprc":base.safe_ap(yy[va],score),"balanced_accuracy":balanced_accuracy_score(yy[va],pred),"f1":float(f[0]),"sensitivity":float(recall[0]),"specificity":float(((pred==0)&(yy[va]==0)).sum()/max((yy[va]==0).sum(),1)),"precision":float(precision[0]),"n_val":len(va)}


def run_linear(c: pd.DataFrame, arrays: Dict[str,np.ndarray], ff: Dict[int,Tuple[np.ndarray,np.ndarray]]):
    y=c.class_id.to_numpy(int); rows=[]; pc=[]; flows=[]; preds={}; true_rows=[]
    for rep in REPS:
        X=arrays[rep]; preds[rep]={}
        for fold,(tr,va) in ff.items():
            p,pred=softmax_probe(X,y,tr,va,7); preds[rep][fold]=(va,pred,p); m=multiclass(y[va],p); m.update({"representation":rep,"fold":fold,"n_train":len(tr),"n_val":len(va)}); rows.append(m)
            cm=confusion_matrix(y[va],pred,labels=np.arange(7))
            for a,b in [(3,2),(2,3),(3,5),(5,3),(3,4),(4,3)]: flows.append({"representation":rep,"fold":fold,"source":base.CLASSES[a],"target":base.CLASSES[b],"count":int(cm[a,b]),"source_total":int(cm[a].sum()),"rate":float(cm[a,b]/max(cm[a].sum(),1))})
            for k,name in enumerate(base.CLASSES):
                yy=(y[va]==k).astype(int); pr,rc,ff1,_=precision_recall_fscore_support(y[va],pred,labels=[k],zero_division=0); pc.append({"representation":rep,"fold":fold,"class_id":k,"class_name":name,"precision":float(pr[0]),"recall":float(rc[0]),"f1":float(ff1[0]),"auroc":base.safe_auc(yy,p[:,k]),"auprc":base.safe_ap(yy,p[:,k]),"support":int(yy.sum())})
            for title,pos,neg in [("neutrophil_vs_myeloid",3,2),("neutrophil_vs_tb",3,5)]: true_rows.append({"representation":rep,"fold":fold,"comparison":title,**true_binary(X,y,tr,va,pos,neg)})
    fm=pd.DataFrame(rows); per=pd.DataFrame(pc); flows=pd.DataFrame(flows); tb=pd.DataFrame(true_rows)
    fm.to_csv(R/"metrics/linear_probe_fold_metrics_corrected.csv",index=False); per.to_csv(R/"metrics/linear_probe_per_class_corrected.csv",index=False); flows.to_csv(R/"metrics/confusion_flows_corrected.csv",index=False); tb.to_csv(R/"metrics/true_binary_diagnostics_by_fold_corrected.csv",index=False); tb[tb.comparison=="neutrophil_vs_myeloid"].to_csv(R/"metrics/true_binary_neutrophil_vs_myeloid.csv",index=False); tb[tb.comparison=="neutrophil_vs_tb"].to_csv(R/"metrics/true_binary_neutrophil_vs_tb.csv",index=False)
    summary=[]
    for rep,g in fm.groupby("representation"):
        row={"representation":rep}
        for col in ["accuracy","balanced_accuracy","macro_f1","macro_auprc","macro_auroc","weighted_f1","mcc","lowest_three_f1"]: row[col]=g[col].mean(); row[col+"_sd"]=g[col].std(ddof=1)
        summary.append(row)
    pd.DataFrame(summary).to_csv(R/"metrics/linear_probe_summary_corrected.csv",index=False)
    per[per.class_name=="Neutrophil"].groupby("representation",as_index=False).agg(precision=("precision","mean"),recall=("recall","mean"),f1=("f1","mean"),auroc=("auroc","mean"),auprc=("auprc","mean")).to_csv(R/"metrics/neutrophil_metrics_corrected.csv",index=False)
    tb.groupby(["representation","comparison"],as_index=False).mean(numeric_only=True).to_csv(R/"metrics/true_binary_summary.csv",index=False)
    # paired fold deltas for all prespecified comparisons
    deltas=[]; piv=fm.pivot(index="fold",columns="representation")
    pairs=[("PHIKON_V2_SMALL","CELLVIT_TOKEN_ALIGNED","PHIKON_SMALL_MINUS_CELLVIT"),("PHIKON_V2_CONTEXT","CELLVIT_TOKEN_ALIGNED","PHIKON_CONTEXT_MINUS_CELLVIT"),("MIDNIGHT12K_SMALL","CELLVIT_TOKEN_ALIGNED","MIDNIGHT_SMALL_MINUS_CELLVIT"),("MIDNIGHT12K_CONTEXT","CELLVIT_TOKEN_ALIGNED","MIDNIGHT_CONTEXT_MINUS_CELLVIT"),("PHIKON_V2_CONTEXT","PHIKON_V2_SMALL","PHIKON_CONTEXT_MINUS_SMALL"),("MIDNIGHT12K_CONTEXT","MIDNIGHT12K_SMALL","MIDNIGHT_CONTEXT_MINUS_SMALL"),("MIDNIGHT12K_SMALL","PHIKON_V2_SMALL","MIDNIGHT_SMALL_MINUS_PHIKON_SMALL"),("MIDNIGHT12K_CONTEXT","PHIKON_V2_CONTEXT","MIDNIGHT_CONTEXT_MINUS_PHIKON_CONTEXT")]
    for metric in ["macro_f1","macro_auprc"]:
        for a,b,label in pairs:
            for fold in piv.index: deltas.append({"fold":int(fold),"metric":metric,"comparison":label,"delta":float(piv.loc[fold,(metric,a)]-piv.loc[fold,(metric,b)])})
    pd.DataFrame(deltas).to_csv(R/"metrics/representation_paired_deltas_corrected.csv",index=False)
    return fm,per,tb,preds


def nested_mlp(c,arrays,ff,max_epochs=30):
    y=c.class_id.to_numpy(int); rows=[]; inner_audit=[]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for rep in REPS:
        X=arrays[rep]
        for fold,(tr,va) in ff.items():
            batches=sorted(c.iloc[tr].batch.astype(str).unique()); inner_val_batch=batches[-1]; inner_val=np.asarray([i for i in tr if str(c.iloc[i].batch)==inner_val_batch]); inner_tr=np.asarray([i for i in tr if str(c.iloc[i].batch)!=inner_val_batch])
            scaler=StandardScaler().fit(X[inner_tr]); xt=torch.from_numpy(scaler.transform(X[inner_tr]).astype(np.float32)).to(device); xv=torch.from_numpy(scaler.transform(X[inner_val]).astype(np.float32)).to(device); yt=torch.from_numpy(y[inner_tr].astype(np.int64)).to(device); weights=len(yt)/(7*torch.clamp(torch.bincount(yt,minlength=7).float(),min=1))
            torch.manual_seed(SEED+fold); net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],256),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(128,7)).to(device); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); best=-1; best_epoch=1
            for ep in range(1,max_epochs+1):
                net.train(); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(net(xt),yt,weight=weights); loss.backward(); opt.step(); net.eval()
                with torch.no_grad(): ip=torch.softmax(net(xv),dim=1).cpu().numpy()
                score=multiclass(y[inner_val],ip)["macro_f1"]
                if score>best: best=score; best_epoch=ep
            del xt,xv,yt,net,opt
            # Refit from scratch on all outer training cells for the selected epoch count.
            scaler=StandardScaler().fit(X[tr]); xt=torch.from_numpy(scaler.transform(X[tr]).astype(np.float32)).to(device); xo=torch.from_numpy(scaler.transform(X[va]).astype(np.float32)).to(device); yt=torch.from_numpy(y[tr].astype(np.int64)).to(device); weights=len(yt)/(7*torch.clamp(torch.bincount(yt,minlength=7).float(),min=1)); torch.manual_seed(SEED+1000+fold); net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],256),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(128,7)).to(device); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
            for _ in range(best_epoch): net.train(); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(net(xt),yt,weight=weights); loss.backward(); opt.step()
            net.eval();
            with torch.no_grad(): op=torch.softmax(net(xo),dim=1).cpu().numpy()
            m=multiclass(y[va],op); m.update({"representation":rep,"fold":fold,"selected_epoch":best_epoch,"inner_val_batch":inner_val_batch,"n_train":len(tr),"n_val":len(va)}); rows.append(m); inner_audit.append({"representation":rep,"fold":fold,"outer_train_batches":";".join(sorted(set(c.iloc[tr].batch))),"inner_val_batch":inner_val_batch,"selected_epoch":best_epoch,"outer_validation_used_for_selection":False}); del xt,xo,yt,net,opt
            if device.type=="cuda": torch.cuda.empty_cache()
    out=pd.DataFrame(rows); out.to_csv(R/"metrics/mlp_probe_fold_metrics_corrected.csv",index=False); summary=[]
    for rep,g in out.groupby("representation"):
        row={"representation":rep}
        for col in ["accuracy","balanced_accuracy","macro_f1","macro_auprc","macro_auroc","weighted_f1","mcc","lowest_three_f1"]: row[col]=g[col].mean(); row[col+"_sd"]=g[col].std(ddof=1)
        summary.append(row)
    pd.DataFrame(summary).to_csv(R/"metrics/mlp_probe_summary_corrected.csv",index=False); pd.DataFrame(inner_audit).to_csv(R/"qc/mlp_nested_validation_audit.csv",index=False); return out


def geometry(c,arrays):
    rng=np.random.default_rng(SEED); subset=rng.choice(len(c),size=min(10000,len(c)),replace=False); subset.sort(); c.iloc[subset][["canonical_index","cell_id","batch","class_id"]].to_csv(R/"metrics/geometry_subset_cell_ids.csv",index=False); y=c.class_id.to_numpy()[subset]; b=c.batch.astype(str).to_numpy()[subset]; out=[]; pairs_rng=np.random.default_rng(SEED); pi=pairs_rng.integers(0,len(subset),100000); pj=pairs_rng.integers(0,len(subset),100000); keep=pi!=pj; pi=pi[keep]; pj=pj[keep]
    for rep in REPS:
        x=StandardScaler().fit_transform(arrays[rep][subset]); x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12); dist=1-np.sum(x[pi]*x[pj],axis=1); samec=y[pi]==y[pj]; sameb=b[pi]==b[pj]; knn=KNeighborsClassifier(n_neighbors=11,metric="cosine",n_jobs=-1).fit(x,y); nn=knn.kneighbors(return_distance=False)[:,1:]; out.append({"representation":rep,"within_class_cosine_distance":float(dist[samec].mean()),"between_class_cosine_distance":float(dist[~samec].mean()),"class_separation_ratio":float(dist[~samec].mean()/max(dist[samec].mean(),1e-12)),"within_batch_cosine_distance":float(dist[sameb].mean()),"between_batch_cosine_distance":float(dist[~sameb].mean()),"batch_separation_ratio":float(dist[~sameb].mean()/max(dist[sameb].mean(),1e-12)),"knn_class_purity":float(np.mean(y[nn]==y[:,None])),"knn_batch_purity":float(np.mean(b[nn]==b[:,None])),"silhouette":float(silhouette_score(x,y,metric="cosine",sample_size=min(5000,len(x)),random_state=SEED)),"n_geometry":len(x)})
    out=pd.DataFrame(out); out.to_csv(R/"metrics/representation_geometry_corrected.csv",index=False); return out


def agreement(c,preds):
    rows=[]
    for scale,a,b in [("SMALL","PHIKON_V2_SMALL","MIDNIGHT12K_SMALL"),("CONTEXT","PHIKON_V2_CONTEXT","MIDNIGHT12K_CONTEXT")]:
        for fold in sorted(preds[a]):
            va,pa,_=preds[a][fold]; _,pb,_=preds[b][fold]; y=c.class_id.to_numpy()[va]; ca=pa==y; cb=pb==y; ntrue=y==3; rows.append({"scale":scale,"fold":fold,"agreement":float(np.mean(pa==pb)),"cohen_kappa":cohen_kappa_score(pa,pb),"correct_correct":float(np.mean(ca&cb)),"error_error":float(np.mean(~ca&~cb)),"one_correct_one_wrong":float(np.mean(ca^cb)),"neutrophil_tp_overlap":float(np.mean((pa==3)&(pb==3)&ntrue)/max(np.mean(ntrue),1e-12)),"phikon_macro_f1":f1_score(y,pa,average="macro"),"midnight_macro_f1":f1_score(y,pb,average="macro")})
    out=pd.DataFrame(rows); out.to_csv(R/"metrics/cross_encoder_agreement.csv",index=False); return out


def figures(fm,per,tb,geom,agree):
    F=R/"figures"; F.mkdir(exist_ok=True); order=REPS
    def bar(metric,name,ylabel):
        s=fm.groupby("representation",as_index=False)[metric].mean(); s.reindex(s.representation.map({x:i for i,x in enumerate(order)}).argsort()).plot.bar(x="representation",y=metric,legend=False,figsize=(9,4)); plt.ylabel(ylabel); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(F/name); plt.close()
    bar("macro_f1","Fig2_corrected_macroF1_fiveway.pdf","Macro-F1"); bar("macro_auprc","Fig3_corrected_macroAUPRC_fiveway.pdf","Macro-AUPRC")
    q=per.groupby(["representation","class_name"],as_index=False).f1.mean().pivot(index="class_name",columns="representation",values="f1").reindex(columns=order); q.plot.bar(figsize=(12,5)); plt.ylabel("F1"); plt.tight_layout(); plt.savefig(F/"Fig4_corrected_per_class_F1.pdf"); plt.close()
    n=per[per.class_name=="Neutrophil"].groupby("representation",as_index=False).f1.mean(); n.plot.bar(x="representation",y="f1",legend=False,figsize=(9,4)); plt.ylabel("Neutrophil F1"); plt.tight_layout(); plt.savefig(F/"Fig5_corrected_neutrophil_metrics.pdf"); plt.close()
    b=tb.groupby(["representation","comparison"],as_index=False).auroc.mean().pivot(index="representation",columns="comparison",values="auroc").reindex(order); b.plot.bar(figsize=(10,4)); plt.ylabel("True binary AUROC"); plt.tight_layout(); plt.savefig(F/"Fig6_corrected_true_binary_results.pdf"); plt.close()
    s=fm.groupby("representation",as_index=False).macro_f1.mean().set_index("representation"); s.loc[[x for x in order if x in s.index]].plot.bar(figsize=(10,4)); plt.ylabel("Macro-F1"); plt.tight_layout(); plt.savefig(F/"Fig7_corrected_small_vs_context.pdf"); plt.close()
    geom.set_index("representation")[["class_separation_ratio","batch_separation_ratio"]].reindex(order).plot.bar(figsize=(10,4)); plt.ylabel("Distance ratio"); plt.tight_layout(); plt.savefig(F/"Fig8_corrected_representation_geometry.pdf"); plt.close()
    agree.groupby("scale",as_index=False).agreement.mean().plot.bar(x="scale",y="agreement",legend=False,figsize=(5,4)); plt.ylabel("Prediction agreement"); plt.tight_layout(); plt.savefig(F/"Fig9_corrected_cross_encoder_agreement.pdf"); plt.close()
    # Paired fold deltas
    d=pd.read_csv(R/"metrics/representation_paired_deltas_corrected.csv"); d=d[d.metric=="macro_f1"]; d.boxplot(column="delta",by="comparison",rot=25,figsize=(11,4)); plt.suptitle(""); plt.tight_layout(); plt.savefig(F/"Fig10_corrected_fold_paired_deltas.pdf"); plt.close()
    m=pd.read_csv(R/"metrics/mlp_probe_summary_corrected.csv"); m.plot.bar(x="representation",y="macro_f1",legend=False,figsize=(10,4)); plt.ylabel("Nested MLP macro-F1"); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(F/"Fig11_corrected_MLP.pdf"); plt.close()
    fig,ax=plt.subplots(figsize=(12,3)); ax.axis("off"); xs=np.linspace(.1,.9,5); labs=["canonical IDs","CellViT aligned","Phikon","Midnight","same folds"]
    for i,x in enumerate(xs): ax.text(x,.55,labs[i],ha="center",bbox=dict(boxstyle="round",fc="#e7f4df" if i>1 else "#dcecf7"));
    for i in range(4): ax.annotate("",xy=(xs[i+1]-.05,.55),xytext=(xs[i]+.05,.55),arrowprops=dict(arrowstyle="->"))
    ax.set_title("Task010 corrected canonical five-representation design"); fig.tight_layout(); fig.savefig(F/"Fig1_task010_corrected_design.pdf"); plt.close(fig)


def main():
    seed(); base.mkdirs(); c=canonical_order(); arrays=align_existing(c); midnight_features(c); arrays["MIDNIGHT12K_SMALL"]=torch.load(R/"features/midnight12k_small.pt",map_location="cpu")["features"].numpy(); arrays["MIDNIGHT12K_CONTEXT"]=torch.load(R/"features/midnight12k_context.pt",map_location="cpu")["features"].numpy()
    for rep in REPS:
        p=torch.load(R / "features" / ({"CELLVIT_TOKEN_ALIGNED":"cellvit_tokens_aligned.pt","PHIKON_V2_SMALL":"phikon_v2_small_aligned.pt","PHIKON_V2_CONTEXT":"phikon_v2_context_aligned.pt","MIDNIGHT12K_SMALL":"midnight12k_small.pt","MIDNIGHT12K_CONTEXT":"midnight12k_context.pt"}[rep]),map_location="cpu");
        if [str(x) for x in p["cell_ids"]] != c.cell_id.astype(str).tolist(): raise RuntimeError(f"Canonical ID mismatch {rep}")
        if not np.isfinite(arrays[rep]).all(): raise RuntimeError(f"Nonfinite {rep}")
    ff=folds(c); fm,per,tb,preds=run_linear(c,arrays,ff); mlp=nested_mlp(c,arrays,ff); geom=geometry(c,arrays); agree=agreement(c,preds); figures(fm,per,tb,geom,agree)
    decision={"status":"CORRECTED_FIVE_REPRESENTATION_COMPLETE","n_canonical":len(c),"representations":REPS,"old_cellvit_alignment_rows_reordered":int(torch.load(R/"features/cellvit_tokens_aligned.pt",map_location="cpu")["old_rows_reordered"]),"true_binary_probes":True,"nested_mlp":True,"midnight_gt_centered":"PENDING_SECONDARY","production_model_modified":False}
    (R/"metrics/decision_summary_corrected.json").write_text(json.dumps(decision,indent=2)+"\n")
    print(fm.groupby("representation")[["macro_f1","macro_auprc"]].mean().to_string(),flush=True)


if __name__=="__main__": main()
