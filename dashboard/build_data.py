"""대시보드용 결과 파일 생성 → dashboard/data/
노트북 01~05를 실행해 results/cache/와 models/final_model.pkl이 있는 상태에서 실행합니다.
    uv run python dashboard/build_data.py
"""
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from steel import (CLASS_ORDER, CV, RS, SCORING, LabelEncodedClassifier, add_features, cached, final_model, load_data,
                   make_pipeline, plate_groups)

OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "results" / "cache"
load = lambda name: joblib.load(CACHE / f"{name}.pkl")

X_train, X_test, y_train, y_test = load_data()
XF, XF_test = add_features(X_train), add_features(X_test)
yv = y_train.values
CLS = np.array(CLASS_ORDER)
S5 = list(CV.split(XF, y_train))
SR = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS).split(XF, y_train))


def fold_metrics(pred_by_fold, splits):
    """fold별 예측 → 평균 지표."""
    rows = [{"macro_f1": f1_score(yv[va], p, average="macro"), "accuracy": accuracy_score(yv[va], p),
             "precision": precision_score(yv[va], p, average="macro", zero_division=0),
             "recall": recall_score(yv[va], p, average="macro")} for p, (_, va) in zip(pred_by_fold, splits)]
    d = pd.DataFrame(rows)
    return {**d.mean().to_dict(), "macro_f1_std": d["macro_f1"].std()}


# ---------------- 1. Baseline 6개 모델 (5-fold) ----------------
FEAT = list(X_train.columns)
BASELINES = {
    "LogisticRegression": make_pipeline(LogisticRegression(max_iter=5000, random_state=RS), FEAT, scale=True),
    "DecisionTree": make_pipeline(DecisionTreeClassifier(random_state=RS), FEAT),
    "RandomForest": make_pipeline(RandomForestClassifier(random_state=RS, n_jobs=1), FEAT),
    "GBM": make_pipeline(GradientBoostingClassifier(random_state=RS), FEAT),
    "XGBoost": make_pipeline(LabelEncodedClassifier(XGBClassifier(random_state=RS, n_jobs=1)), FEAT),
    "LightGBM": make_pipeline(LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1), FEAT),
}
def base_cv(m):
    r = cross_validate(m, X_train, y_train, cv=CV, scoring=SCORING)
    return {"macro_f1": r["test_macro_f1"].mean(), "macro_f1_std": r["test_macro_f1"].std(), "accuracy": r["test_accuracy"].mean(),
            "precision": r["test_precision"].mean(), "recall": r["test_recall"].mean()}
base = cached("dashboard_baseline_cv", lambda: dict(zip(BASELINES, Parallel(n_jobs=-1)(delayed(base_cv)(m) for m in BASELINES.values()))))
print("baseline 완료")

# ---------------- 2. 불균형 처리 (02 캐시) ----------------
imb_rows, rec_rows = [], []
oof = {}
for model, method, k, pred in load("imbalance_oof"):
    tr, va = S5[k]
    imb_rows.append({"model": model, "method": method, "fold": k, "macro_f1": f1_score(yv[va], pred, average="macro")})
    oof.setdefault((model, method), np.empty(len(yv), dtype=object))[va] = pred
imb = pd.DataFrame(imb_rows).groupby(["model", "method"], sort=False)["macro_f1"].mean().reset_index()
imb.to_csv(OUT / "imbalance_f1.csv", index=False)
for (model, method), p in oof.items():
    pr, rc, f1, _ = precision_recall_fscore_support(yv, p, labels=CLASS_ORDER, zero_division=0)
    for c, a, b, d in zip(CLASS_ORDER, pr, rc, f1):
        rec_rows.append({"model": model, "method": method, "class": c, "precision": a, "recall": b, "f1": d})
pd.DataFrame(rec_rows).to_csv(OUT / "imbalance_per_class.csv", index=False)
print("불균형 처리 완료")

# ---------------- 3. 튜닝·앙상블 (03 캐시) ----------------
P5, PR = load("oof_5fold"), load("oof_repeated")
TUNED = ["LightGBM", "XGBoost", "ExtraTrees", "RandomForest", "GBM"]
proba_pred = lambda P, members, splits: [CLS[np.mean([P[(n, i)] for n in members], axis=0).argmax(1)] for i in range(len(splits))]
combos = [c for r in range(1, 6) for c in combinations(TUNED, r)]
combo_df = pd.DataFrame([{"combo": " + ".join(c), "n_models": len(c), **fold_metrics(proba_pred(P5, c, S5), S5)} for c in combos])
combo_df.sort_values("macro_f1", ascending=False).to_csv(OUT / "ensemble_combos.csv", index=False)
print("앙상블 완료")

# ---------------- 4. 일반화 점검 (04 캐시) ----------------
groups = plate_groups(X_train).values
SPLITS = {"random 5×3": SR,
          "group 5×3": [s for r in range(3) for s in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS + r).split(XF, y_train, groups)]}
G = load("generalization_v2_pred")
gen_rows, gen_pc = [], []
for (name, cv, i), p in G.items():
    va = SPLITS[cv][i][1]
    gen_rows.append({"model": name, "split": cv, "fold": i, "macro_f1": f1_score(yv[va], p, average="macro")})
gen = pd.DataFrame(gen_rows)
gen.to_csv(OUT / "generalization_folds.csv", index=False)
for name in gen["model"].unique():
    for cv in SPLITS:
        yt = np.concatenate([yv[SPLITS[cv][i][1]] for i in range(15)])
        yp = np.concatenate([G[(name, cv, i)] for i in range(15)])
        for c, f in zip(CLASS_ORDER, f1_score(yt, yp, labels=CLASS_ORDER, average=None)):
            gen_pc.append({"model": name, "split": cv, "class": c, "f1": f})
pd.DataFrame(gen_pc).to_csv(OUT / "generalization_per_class.csv", index=False)
rep = gen[gen["split"] == "random 5×3"].groupby("model")["macro_f1"].mean()
grp = gen[gen["split"] == "group 5×3"].groupby("model")["macro_f1"].mean()
print("일반화 점검 완료")

# ---------------- 5. 최종 모델 (05) ----------------
model = joblib.load(ROOT / "models" / "final_model.pkl")
r = cached("dashboard_final_cv5", lambda: cross_validate(final_model(XF.columns), XF, y_train, cv=CV, scoring=SCORING, n_jobs=-1))
final_cv = {"macro_f1": r["test_macro_f1"].mean(), "macro_f1_std": r["test_macro_f1"].std(), "accuracy": r["test_accuracy"].mean(),
            "precision": r["test_precision"].mean(), "recall": r["test_recall"].mean()}
pred = model.predict(XF_test)
rng = np.random.default_rng(RS)
boot = [f1_score(y_test.values[i], pred[i], average="macro") for i in (rng.integers(0, len(y_test), len(y_test)) for _ in range(2000))]
rep_final = load("final35_repeated")
summary = {
    "test_macro_f1": f1_score(y_test, pred, average="macro"), "test_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "test_accuracy": accuracy_score(y_test, pred), "test_precision": precision_score(y_test, pred, average="macro"),
    "test_recall": recall_score(y_test, pred, average="macro"), "test_weighted_f1": f1_score(y_test, pred, average="weighted"),
    "cv5_macro_f1": final_cv["macro_f1"], "repeated_mean": float(rep_final.mean()), "repeated_std": float(rep_final.std()),
    "group_cv": float(grp["ET+GBM (35, count 제외)"]), "n_test": int(len(y_test)), "n_errors": int((pred != y_test.values).sum()),
}
json.dump(summary, open(OUT / "final_summary.json", "w", encoding="utf-8"), indent=2)
pd.DataFrame(confusion_matrix(y_test, pred, labels=CLASS_ORDER), index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(OUT / "final_confusion.csv")
pr, rc, f1, n = precision_recall_fscore_support(y_test, pred, labels=CLASS_ORDER)
pd.DataFrame({"class": CLASS_ORDER, "precision": pr, "recall": rc, "f1": f1, "support": n}).to_csv(OUT / "final_per_class.csv", index=False)
pi = cached("dashboard_importance", lambda: permutation_importance(model, XF_test, y_test, scoring="f1_macro", n_repeats=5, random_state=RS, n_jobs=-1))
pd.DataFrame({"feature": XF_test.columns, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values(
    "importance", ascending=False).to_csv(OUT / "final_importance.csv", index=False)
lc = pd.DataFrame(load("final35_learning_curve"), columns=["frac", "train_f1", "val_f1", "val_logloss"]).groupby("frac").mean().reset_index()
lc.to_csv(OUT / "final_learning_curve.csv", index=False)
print("최종 모델 완료")

# ---------------- 6. 모델별 평가 지표 비교 (단계별) ----------------
gbm_cw = imb[(imb["model"] == "GBM") & (imb["method"] == "클래스 가중치")]["macro_f1"].iloc[0]
single = {n: fold_metrics(proba_pred(P5, [n], S5), S5) for n in TUNED}
sv36 = fold_metrics(proba_pred(P5, ["ExtraTrees", "GBM"], S5), S5)
rows = [{"model": n, "stage": "Baseline", "features": 26, **m} for n, m in base.items()]
rows += [{"model": "GBM + 클래스 가중치", "stage": "불균형 처리", "features": 26, "macro_f1": gbm_cw},
         {"model": "GBM 튜닝 (26)", "stage": "튜닝", "features": 26, "macro_f1": load("tune_gbm_26")["score"], "macro_f1_std": load("tune_gbm_26")["std"]}]
rows += [{"model": f"{n} 튜닝 (36)", "stage": "파생변수 + 튜닝", "features": 36, **single[n]} for n in TUNED]
rows += [{"model": "ExtraTrees + GBM (36)", "stage": "앙상블", "features": 36, **sv36,
          "repeated_cv": rep["ET+GBM (36, count 포함)"], "group_cv": grp["ET+GBM (36, count 포함)"]},
         {"model": "ExtraTrees + GBM (35, 최종)", "stage": "최종 (plate_defect_count 제외)", "features": 35, **final_cv,
          "repeated_cv": rep["ET+GBM (35, count 제외)"], "group_cv": grp["ET+GBM (35, count 제외)"], "test_macro_f1": summary["test_macro_f1"]}]
comp = pd.DataFrame(rows)
comp.loc[comp["model"] == "XGBoost", ["repeated_cv", "group_cv"]] = [rep["Baseline XGBoost (26)"], grp["Baseline XGBoost (26)"]]
comp.to_csv(OUT / "model_comparison.csv", index=False)
print("완료 →", OUT)
