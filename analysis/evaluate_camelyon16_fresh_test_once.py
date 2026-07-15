"""Evaluate the checksum-locked CAMELYON16 model on the fresh test set once."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import joblib, numpy as np, torch, yaml
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from analysis.attention_mil_v2 import AttentionMIL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"normal": 0, "tumor": 1}

def parse_args():
    p=argparse.ArgumentParser(description="One-time fresh-test evaluator")
    p.add_argument("--config", default="configs/wsi/camelyon16_final_test_gate.yaml")
    return p.parse_args()

def path(value):
    p=Path(value); return p.resolve() if p.is_absolute() else (PROJECT_ROOT/p).resolve()

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def load_yaml(p): return yaml.safe_load(p.read_text(encoding="utf-8-sig"))

def verify_gate(cfg):
    dev_path=path(cfg["development_lock"]); artifact_path=path(cfg["selected_artifact"])
    manifest_path=path(cfg["processing_manifest"]); holdout_path=path(cfg["fresh_holdout_lock"])
    for p in (dev_path,artifact_path,manifest_path,holdout_path):
        if not p.is_file(): raise FileNotFoundError(p)
    dev=json.loads(dev_path.read_text(encoding="utf-8")); hold=json.loads(holdout_path.read_text(encoding="utf-8")); manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if dev.get("passed") is not True: raise RuntimeError("Development lock did not pass")
    if dev["development_counts"] != {"train":30,"validation":6,"test_loaded":0}: raise RuntimeError("Development counts are not locked")
    if dev.get("test_boundary_status") != "UNTOUCHED": raise RuntimeError("Test boundary was not untouched")
    if dev.get("selected_model") != cfg["expected_model"]: raise RuntimeError("Selected model differs from gate")
    if sha256(artifact_path) != dev.get("artifact_sha256"): raise RuntimeError("Artifact checksum mismatch")
    if sha256(manifest_path) != dev.get("processing_manifest_sha256"): raise RuntimeError("Processing manifest checksum mismatch")
    expected=sorted(cfg["expected_test_slides"])
    if sorted(hold.get("fixed_test_slides",[])) != expected: raise RuntimeError("Holdout identities mismatch")
    if hold.get("test_boundary_status") != "UNTOUCHED_UNTIL_FINAL_EVALUATION": raise RuntimeError("Holdout status mismatch")
    rows=[r for r in manifest["slides"] if r["split"]=="test"]
    if sorted(r["slide"] for r in rows) != expected: raise RuntimeError("Manifest test identities mismatch")
    return dev, artifact_path, rows, {"development_lock_sha256":sha256(dev_path),"artifact_sha256":sha256(artifact_path),"processing_manifest_sha256":sha256(manifest_path),"holdout_lock_sha256":sha256(holdout_path)}

def predict_attention(artifact, rows, embedding_root):
    payload=artifact["artifact"]; mean=payload["instance_mean"]; std=payload["instance_std"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models=[]
    for state in payload["state_dicts"]:
        model=AttentionMIL(in_dim=512,hidden_dim=int(payload["hidden_dim"])).to(device); model.load_state_dict(state); model.eval(); models.append(model)
    probs=[]
    with torch.inference_mode():
        for row in rows:
            bag=np.load(embedding_root/f"{row['slide']}_embeddings.npy",allow_pickle=False).astype(np.float32)
            bag=np.ascontiguousarray((bag-mean)/std); tensor=torch.from_numpy(bag).to(device); seed_probs=[]
            for model in models:
                logit,attn=model(tensor)
                if not torch.isclose(attn.sum(),torch.tensor(1.0,device=device),atol=1e-5): raise RuntimeError("Attention normalization failed")
                seed_probs.append(float(torch.sigmoid(logit).cpu()))
            probs.append(float(np.mean(seed_probs)))
    return np.array(probs),str(device)

def metrics(y,p,t):
    pred=(p>=t).astype(int); tn,fp,fn,tp=(int(x) for x in confusion_matrix(y,pred,labels=[0,1]).ravel())
    return {"sample_count":len(y),"threshold":float(t),"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall_sensitivity":float(recall_score(y,pred,zero_division=0)),"specificity":float(tn/(tn+fp)) if tn+fp else None,"f1":float(f1_score(y,pred,zero_division=0)),"confusion_matrix":[[tn,fp],[fn,tp]]}

def main():
    args=parse_args(); config_path=path(args.config); cfg=load_yaml(config_path); out=path(cfg["output_root"]); result_path=out/"final_test_result.json"; pred_path=out/"final_test_predictions.csv"; receipt_path=out/"FINAL_TEST_EXECUTED.lock"
    if cfg.get("refuse_overwrite",True) and any(p.exists() for p in (result_path,pred_path,receipt_path)): raise RuntimeError("Final-test output already exists; overwrite refused")
    dev,artifact_path,rows,checksums=verify_gate(cfg); artifact=joblib.load(artifact_path)
    if artifact["model_name"]!=dev["selected_model"] or float(artifact["threshold"])!=float(dev["selected_threshold"]): raise RuntimeError("Artifact metadata mismatch")
    probs,device=predict_attention(artifact,rows,path(cfg["embedding_root"])); y=np.array([LABELS[r["label"]] for r in rows]); threshold=float(dev["selected_threshold"])
    predictions=[]
    for row,prob in zip(rows,probs): predictions.append({"slide":row["slide"],"label":row["label"],"label_binary":LABELS[row["label"]],"probability":float(prob),"threshold":threshold,"prediction":"tumor" if prob>=threshold else "normal","prediction_binary":int(prob>=threshold),"correct":bool(int(prob>=threshold)==LABELS[row["label"]])})
    result={"schema_version":"1.0","dataset":cfg["dataset"],"scientific_scope":"one-time six-slide fresh-test evaluation","selected_model":dev["selected_model"],"selected_threshold":threshold,"device":device,"test_slides":[r["slide"] for r in rows],"metrics":metrics(y,probs,threshold),"predictions":predictions,"frozen_checksums":checksums,"limitations":["The fresh test set contains only six slides.","One prediction changes accuracy by 16.7 percentage points.","This is a development benchmark, not a clinical-performance estimate."],"passed":True}
    out.mkdir(parents=True,exist_ok=True); result_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    with pred_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
    receipt={"result_sha256":sha256(result_path),"predictions_sha256":sha256(pred_path),"development_artifact_sha256":checksums["artifact_sha256"],"executed_once":True}
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Receipt written to: {receipt_path}"); print("PASS: One-time fresh-test evaluation completed; overwrite lock created.")
if __name__=="__main__": main()
