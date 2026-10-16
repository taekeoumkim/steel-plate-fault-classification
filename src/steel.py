"""강판 결함 분류 공통 모듈: 데이터 로드, 전처리, 파생변수, 모델 래퍼, 최종 모델."""
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RS = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)   # 모든 비교에 쓰는 5-fold

TARGET_COLS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
CLASS_ORDER = ["Other_Faults", "Bumps", "K_Scratch", "Z_Scratch", "Pastry", "Stains", "Dirtiness"]
DISCRETE = ["TypeOfSteel_A300", "Outside_Global_Index"]          # 스케일링 제외, SMOTENC 범주형
PLATE_KEYS = ["Length_of_Conveyer", "Steel_Plate_Thickness", "TypeOfSteel_A300"]   # 생산 조건 = 폭·두께·강종


SCORING = {"macro_f1": "f1_macro", "accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro"}


def cached(name, fn, rerun=False):
    """오래 걸리는 계산 결과를 results/cache/{name}.pkl에 저장·재사용."""
    path = ROOT / "results" / "cache" / f"{name}.pkl"
    if path.exists() and not rerun:
        return joblib.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = fn()
    joblib.dump(out, path)
    return out


# ---------------- 데이터 ----------------
def load_raw():
    return pd.read_csv(ROOT / "data" / "Faults.csv")


def load_data():
    """정답 7열 → target 1열, 중복 변수(A400) 제거, 80:20 층화 분할."""
    df = load_raw()
    y = df[TARGET_COLS].idxmax(axis=1).replace({"K_Scatch": "K_Scratch"}).rename("target")
    X = df.drop(columns=TARGET_COLS + ["TypeOfSteel_A400"])        # A400 = 1 - A300
    tr, te = train_test_split(df.index, test_size=0.2, random_state=RS, stratify=y)
    return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]


def plate_groups(X):
    """생산 조건 그룹 키 (폭_두께_강종)."""
    return X[PLATE_KEYS].astype(str).agg("_".join, axis=1)


# ---------------- 전처리·불균형 ----------------
def build_preprocessor(features, scale=False):
    """결측 대체(+선택적 표준화). 이산형 변수는 항상 마지막 열로 배치 → SMOTENC 인덱스로 사용."""
    features = list(features)
    disc = [c for c in DISCRETE if c in features]
    cont = [c for c in features if c not in disc]
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([("cont", Pipeline(steps), cont), ("disc", SimpleImputer(strategy="most_frequent"), disc)],
                             verbose_feature_names_out=False)


def discrete_idx(features):
    n_disc = len([c for c in DISCRETE if c in features])
    return list(range(len(features) - n_disc, len(features)))


def make_pipeline(model, features, scale=False):
    return Pipeline([("prep", build_preprocessor(features, scale)), ("model", model)])


class Oversample:
    """SMOTENC 목표 건수: 각 클래스를 max(n_c, n_max / ir)로 (ir=1이면 완전 균등)."""
    def __init__(self, ir=1):
        self.ir = ir

    def __call__(self, y):
        c = pd.Series(y).value_counts()
        return {k: int(max(n, round(c.max() / self.ir))) for k, n in c.items()}


class Undersample:
    """RandomUnderSampler 목표 건수: 각 클래스를 min(n_c, ir × n_min)으로."""
    def __init__(self, ir=1):
        self.ir = ir

    def __call__(self, y):
        c = pd.Series(y).value_counts()
        return {k: int(min(n, round(self.ir * c.min()))) for k, n in c.items()}


def sample_weight(y, alpha=1.0):
    """클래스 가중치 (n_max / n_c)^alpha. alpha=1이면 'balanced'와 같은 비율."""
    c = pd.Series(y).value_counts()
    return pd.Series(y).map({k: (c.max() / n) ** alpha for k, n in c.items()}).to_numpy()


class BalancedWeightClassifier(ClassifierMixin, BaseEstimator):
    """fit에 들어온 y(학습 fold)로 클래스 가중치를 계산해 sample_weight로 전달 (class_weight 인자가 없는 GBM용)."""
    def __init__(self, estimator, alpha=1.0):
        self.estimator, self.alpha = estimator, alpha

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **({"sample_weight": sample_weight(y, self.alpha)} if self.alpha else {}))
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """문자열 라벨을 정수로 바꿔 학습 (XGBoost용)."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.le_.transform(y), **fit_params)
        return self

    def predict(self, X):
        return self.le_.inverse_transform(np.asarray(self.estimator_.predict(X)).astype(int).ravel())

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# ---------------- 파생변수 ----------------
ROW_FEATURES = ["compactness", "perimeter_ratio_log", "x_straightness",      # 형상
                "lum_range", "lum_contrast", "lum_mean_position",            # 밝기
                "x_center_rel", "x_center_dist"]                             # 위치


def add_features(X):
    """행 단위 파생변수 8개 (다른 행·정답을 쓰지 않음 → 누수 없음)."""
    out = X.copy()
    w = (X["X_Maximum"] - X["X_Minimum"]).clip(lower=0)
    out["compactness"] = (X["X_Perimeter"] + X["Y_Perimeter"]) / np.sqrt(X["Pixels_Areas"])   # 면적 대비 둘레
    out["perimeter_ratio_log"] = np.log((X["X_Perimeter"] + 1) / (X["Y_Perimeter"] + 1))       # 가로/세로 윤곽 비
    out["x_straightness"] = X["X_Perimeter"] / (w + 1)                                        # 가로 윤곽 굴곡
    mx, mn = X["Maximum_of_Luminosity"], X["Minimum_of_Luminosity"]
    out["lum_range"] = mx - mn
    out["lum_contrast"] = (mx - mn) / (mx + mn).replace(0, 1)
    out["lum_mean_position"] = (X["Sum_of_Luminosity"] / X["Pixels_Areas"] - mn) / (mx - mn + 1)
    xc = (X["X_Minimum"] + X["X_Maximum"]) / 2 / X["Length_of_Conveyer"]
    out["x_center_rel"] = xc                                                                  # 폭 방향 상대 위치
    out["x_center_dist"] = (xc - 0.5).abs()
    return out


class PlateContext(TransformerMixin, BaseEstimator):
    """
    강판 맥락 파생변수 (Pipeline 안에서 학습 fold로만 계산)
      plate_defect_count     : 학습 데이터에서 같은 생산 조건의 결함 수 (처음 보는 조건은 0)
      thickness_per_conveyer : 두께 / 폭
    """
    def __init__(self, use_count=True):
        self.use_count = use_count

    def fit(self, X, y=None):
        self.counts_ = plate_groups(X).value_counts().to_dict()
        return self

    def transform(self, X):
        out = X.copy()
        if self.use_count:
            out["plate_defect_count"] = plate_groups(X).map(self.counts_).fillna(0).astype(float)
        out["thickness_per_conveyer"] = X["Steel_Plate_Thickness"] / X["Length_of_Conveyer"]
        return out


def fe_features(base_cols, use_count=True):
    """PlateContext 이후의 모델 입력 피처 목록."""
    return list(base_cols) + (["plate_defect_count"] if use_count else []) + ["thickness_per_conveyer"]


def fe_pipeline(model, base_cols, use_count=True, sampler="passthrough", scale=False):
    """강판 맥락 → 전처리 → (샘플러) → 모델. 입력은 add_features 적용 데이터."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([("plate", PlateContext(use_count)), ("prep", build_preprocessor(fe_features(base_cols, use_count), scale)),
                        ("sampler", sampler), ("model", model)])


# ---------------- 최종 모델 ----------------
FINAL_PARAMS = {   # 03 노트북 튜닝 결과 (36개 피처)
    "ExtraTrees": {"model__class_weight": "balanced", "model__max_depth": 15, "model__max_features": 0.7,
                   "model__min_samples_leaf": 1, "model__n_estimators": 438},
    "RandomForest": {"model__class_weight": None, "model__max_depth": None, "model__max_features": 0.3,
                     "model__min_samples_leaf": 1, "model__n_estimators": 483},
    "GBM": {"model__alpha": 1, "model__estimator__learning_rate": 0.08793919399591556, "model__estimator__max_depth": 4,
            "model__estimator__max_features": 0.5, "model__estimator__min_samples_leaf": 5,
            "model__estimator__n_estimators": 240, "model__estimator__subsample": 0.9079017872741743},
}


def member_models(base_cols, params=None, use_count=True):
    """Soft Voting 구성 모델 3개 (샘플러 없음, 클래스 가중치는 파라미터로 지정)."""
    p = params or FINAL_PARAMS
    return [
        ("extratrees", fe_pipeline(ExtraTreesClassifier(random_state=RS, n_jobs=1), base_cols, use_count).set_params(**p["ExtraTrees"])),
        ("rf", fe_pipeline(RandomForestClassifier(random_state=RS, n_jobs=1), base_cols, use_count).set_params(**p["RandomForest"])),
        ("gbm", fe_pipeline(BalancedWeightClassifier(GradientBoostingClassifier(random_state=RS)), base_cols, use_count).set_params(**p["GBM"])),
    ]


def final_model(base_cols, params=None, use_count=True):
    """최종 모델: ExtraTrees + RandomForest + GBM Soft Voting (확률 평균)."""
    return VotingClassifier(member_models(base_cols, params, use_count), voting="soft")


# ---------------- 튜닝 탐색 공간 (36개 / 35개 피처 공통) ----------------
def search_spaces(base_cols, use_count=True, models=("LightGBM", "XGBoost", "ExtraTrees", "RandomForest", "GBM")):
    """RandomizedSearchCV용 (파이프라인, 탐색 공간, 반복 횟수). 샘플러(SMOTENC/없음)도 탐색 대상."""
    from imblearn.over_sampling import SMOTENC
    from lightgbm import LGBMClassifier
    from scipy.stats import loguniform, randint, uniform
    from xgboost import XGBClassifier

    feats = fe_features(base_cols, use_count)
    smote = lambda: SMOTENC(categorical_features=discrete_idx(feats), sampling_strategy=Oversample(1), k_neighbors=5, random_state=RS)
    pipe = lambda m, s="passthrough": fe_pipeline(m, base_cols, use_count, sampler=s)
    XP, GP = "model__estimator__estimator__", "model__estimator__"
    spaces = {
        "LightGBM": (pipe(LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1, subsample_freq=1), smote()), 25,
                     {"sampler": [smote(), "passthrough"], "model__class_weight": [None, "balanced"],
                      "model__n_estimators": randint(200, 900), "model__learning_rate": loguniform(0.01, 0.15),
                      "model__num_leaves": randint(8, 64), "model__max_depth": [-1, 4, 6, 8, 10], "model__min_child_samples": randint(5, 40),
                      "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.3, 0.7), "model__reg_lambda": loguniform(1e-3, 10)}),
        "XGBoost": (pipe(BalancedWeightClassifier(LabelEncodedClassifier(XGBClassifier(random_state=RS, n_jobs=1)), alpha=0), smote()), 25,
                    {"sampler": [smote(), "passthrough"], "model__alpha": [0, 0.5, 1],
                     XP + "n_estimators": randint(150, 900), XP + "learning_rate": loguniform(0.01, 0.15), XP + "max_depth": randint(3, 9),
                     XP + "min_child_weight": randint(1, 8), XP + "subsample": uniform(0.6, 0.4), XP + "colsample_bytree": uniform(0.3, 0.7),
                     XP + "reg_lambda": loguniform(1e-2, 10), XP + "gamma": loguniform(1e-3, 1)}),
        "ExtraTrees": (pipe(ExtraTreesClassifier(random_state=RS, n_jobs=1), smote()), 15,
                       {"sampler": [smote(), "passthrough"], "model__class_weight": [None, "balanced", "balanced_subsample"],
                        "model__n_estimators": randint(300, 900), "model__max_depth": [None, 15, 25], "model__min_samples_leaf": randint(1, 5),
                        "model__max_features": ["sqrt", 0.3, 0.5, 0.7]}),
        "RandomForest": (pipe(RandomForestClassifier(random_state=RS, n_jobs=1), smote()), 12,
                         {"sampler": [smote(), "passthrough"], "model__class_weight": [None, "balanced_subsample"],
                          "model__n_estimators": randint(300, 700), "model__max_depth": [None, 15, 25], "model__min_samples_leaf": randint(1, 5),
                          "model__max_features": ["sqrt", 0.3, 0.5]}),
        "GBM": (pipe(BalancedWeightClassifier(GradientBoostingClassifier(random_state=RS), alpha=1)), 8,
                {"model__alpha": [0.5, 1], GP + "n_estimators": randint(100, 350), GP + "learning_rate": loguniform(0.03, 0.2),
                 GP + "max_depth": randint(3, 6), GP + "subsample": uniform(0.6, 0.4), GP + "max_features": [None, "sqrt", 0.5],
                 GP + "min_samples_leaf": randint(1, 10)}),
    }
    return {k: spaces[k] for k in models}


def tune(name, spec, X, y, cv=CV):
    """RandomizedSearchCV (Macro F1). 반환: 최고 파라미터, 최고 점수, 전체 결과."""
    from sklearn.model_selection import RandomizedSearchCV
    pipe, n_iter, space = spec
    rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="f1_macro", cv=cv, n_jobs=-1, random_state=RS + 7, refit=False)
    rs.fit(X, y)
    r = pd.DataFrame(rs.cv_results_)
    best = r.loc[r["rank_test_score"].idxmin()]
    return {"params": best["params"], "score": best["mean_test_score"], "std": best["std_test_score"]}
