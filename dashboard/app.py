"""강판 결함 분류 대시보드
    uv run streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from steel import CLASS_ORDER, load_data, plate_groups  # noqa: E402

DATA = ROOT / "dashboard" / "data"
FIG = ROOT / "figures"
COLORS = dict(zip(CLASS_ORDER, px.colors.qualitative.Set2))

st.set_page_config(page_title="Steel Plate Fault Classification", page_icon="🔩", layout="wide")


@st.cache_data
def load_all():
    X_train, X_test, y_train, y_test = load_data()
    csv = lambda n: pd.read_csv(DATA / f"{n}.csv")
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "comp": csv("model_comparison"), "imb": csv("imbalance_f1"), "imb_pc": csv("imbalance_per_class"),
        "combos": csv("ensemble_combos"), "gen": csv("generalization_folds"), "gen_pc": csv("generalization_per_class"),
        "cm": pd.read_csv(DATA / "final_confusion.csv", index_col=0), "pc": csv("final_per_class"),
        "imp": csv("final_importance"), "lc": csv("final_learning_curve"),
        "summary": json.load(open(DATA / "final_summary.json", encoding="utf-8")),
    }


if not (DATA / "final_summary.json").exists():
    st.error("dashboard/data/ 가 없습니다. 먼저 `uv run python dashboard/build_data.py`를 실행하세요.")
    st.stop()
D = load_all()
S = D["summary"]

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.title("🔩 강판 결함 분류")
    st.markdown(
        "- 데이터: UCI Steel Plates Faults (1,941건, 결함 7종)\n"
        "- 지표: **Macro F1**\n"
        "- 검증: Train 5-fold CV · 반복 CV(5×3) · 그룹 CV / Test 1회\n"
        "- 최종 모델: **ExtraTrees + GBM Soft Voting** (35개 피처)"
    )
    st.caption("반복 CV: 같은 5-fold를 난수만 바꿔 3번(15 fold)\n\n그룹 CV: 같은 생산 조건(폭·두께·강종)을 학습·검증 한쪽에만 배치")

st.title("Steel Plate Fault Classification Dashboard")
k = st.columns(5)
k[0].metric("Test Macro F1", f"{S['test_macro_f1']:.4f}", help=f"95% CI {S['test_ci'][0]:.4f} – {S['test_ci'][1]:.4f}")
k[1].metric("Test Accuracy", f"{S['test_accuracy']:.4f}")
k[2].metric("5-fold CV", f"{S['cv5_macro_f1']:.4f}")
k[3].metric("반복 CV (5×3)", f"{S['repeated_mean']:.4f}", help=f"± {S['repeated_std']:.4f}")
k[4].metric("그룹 CV (처음 보는 조건)", f"{S['group_cv']:.4f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터", "🏆 모델 비교", "⚖️ 불균형 처리", "🎯 최종 모델"])

# ================= 1. 데이터 =================
with tab1:
    Xtr, ytr, Xte, yte = D["X_train"], D["y_train"], D["X_test"], D["y_test"]
    c = st.columns(4)
    c[0].metric("전체 데이터", f"{len(Xtr) + len(Xte):,}건", help=f"Train {len(Xtr):,} / Test {len(Xte):,} (80:20 층화 분할)")
    c[1].metric("입력 변수", f"{Xtr.shape[1]}개", help="원본 27개 중 TypeOfSteel_A400 제외 (A300과 완전 중복)")
    c[2].metric("결함 유형", "7종")
    c[3].metric("최다 / 최소 비율", f"{ytr.value_counts().max() / ytr.value_counts().min():.1f}배")

    left, right = st.columns(2)
    with left:
        st.subheader("클래스 분포")
        dist = pd.DataFrame({"Train": ytr.value_counts(), "Test": yte.value_counts()}).loc[CLASS_ORDER].reset_index(names="class")
        long = dist.melt(id_vars="class", var_name="set", value_name="count")
        fig = px.bar(long, x="class", y="count", color="set", barmode="group", text="count")
        fig.update_layout(height=380, xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Train·Test 비율이 같도록 층화 분할. 가장 많은 Other_Faults(34.7%)와 가장 적은 Dirtiness(2.8%)가 약 12배 차이 → Macro F1 사용")
    with right:
        st.subheader("변수 분포 (유형별)")
        feat = st.selectbox("변수", list(Xtr.columns), index=list(Xtr.columns).index("Pixels_Areas"))
        log = st.checkbox("로그 스케일", value=True)
        df = pd.DataFrame({"value": Xtr[feat], "class": ytr})
        fig = px.box(df, x="class", y="value", color="class", category_orders={"class": CLASS_ORDER}, color_discrete_map=COLORS, log_y=log)
        fig.update_layout(height=380, showlegend=False, xaxis_title=None, yaxis_title=feat)
        st.plotly_chart(fig, width="stretch")

    st.subheader("생산 조건(폭·두께·강종) 구조")
    g_tr, g_te = plate_groups(Xtr), plate_groups(Xte)
    ct = pd.crosstab(g_tr, ytr)
    c = st.columns(3)
    c[0].metric("Train 생산 조건 수", f"{g_tr.nunique()}개")
    c[1].metric("조건 내 최다 유형 비율", f"{ct.max(axis=1).sum() / ct.values.sum():.3f}", help="정답을 무작위로 섞으면 약 0.43")
    c[2].metric("Test 중 Train에 같은 조건이 있는 비율", f"{g_te.isin(set(g_tr)).mean():.1%}")
    top = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index[:15], CLASS_ORDER]
    top.index = [f"{i.split('_')[0]}·{i.split('_')[1]}·{'A300' if i.split('_')[2] == '1' else 'A400'}" for i in top.index]
    fig = px.bar(top.reset_index(names="condition").melt(id_vars="condition", var_name="class", value_name="count"),
                 x="condition", y="count", color="class", color_discrete_map=COLORS, category_orders={"class": CLASS_ORDER})
    fig.update_layout(height=380, xaxis_title="결함 수 상위 15개 생산 조건 (폭·두께·강종)", legend_title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption("같은 생산 조건 안에서는 한두 유형이 몰려 나옴 → 무작위 분할 점수는 '이미 본 조건' 기준 성능. 처음 보는 조건은 그룹 CV로 따로 측정")

# ================= 2. 모델 비교 =================
with tab2:
    comp = D["comp"].copy()
    num = comp.columns.drop(["model", "stage"])
    comp[num] = comp[num].apply(pd.to_numeric, errors="coerce")
    st.subheader("모델별 평가 지표 (Train 5-fold CV)")
    metric = st.radio("지표", ["macro_f1", "accuracy", "precision", "recall"], horizontal=True,
                      format_func=lambda m: {"macro_f1": "Macro F1", "accuracy": "Accuracy", "precision": "Macro Precision", "recall": "Macro Recall"}[m])
    d = comp.dropna(subset=[metric])
    fig = px.bar(d, x="model", y=metric, color="stage", text=d[metric].round(4))
    fig.update_layout(height=420, xaxis_title=None, yaxis_title=metric, legend_title="단계",
                      yaxis_range=[max(0, d[metric].min() - 0.05), min(1, d[metric].max() + 0.03)])
    st.plotly_chart(fig, width="stretch")
    show = comp.rename(columns={"model": "모델", "stage": "단계", "features": "피처 수", "macro_f1": "Macro F1", "macro_f1_std": "std",
                                "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "repeated_cv": "반복 CV",
                                "group_cv": "그룹 CV", "test_macro_f1": "Test Macro F1"})
    for col in show.columns.drop(["모델", "단계", "피처 수"]):
        show[col] = show[col].map(lambda v: "–" if pd.isna(v) else f"{v:.4f}")
    st.dataframe(show, width="stretch", hide_index=True, height=35 * (len(show) + 1) + 3)
    st.caption("5-fold 1회 값은 분할 운에 흔들림(std ≈ 0.02) → 후보 간 비교는 반복 CV(15 fold)로 판단")

    left, right = st.columns(2)
    with left:
        st.subheader("Soft Voting 조합 상위 10개")
        cb = D["combos"].head(10).sort_values("macro_f1")
        fig = px.bar(cb, x="macro_f1", y="combo", orientation="h", text=cb["macro_f1"].round(4), color="n_models")
        fig.update_layout(height=420, xaxis_range=[cb["macro_f1"].min() - 0.01, cb["macro_f1"].max() + 0.005], yaxis_title=None,
                          coloraxis_colorbar_title="모델 수")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("이미 본 조건 vs 처음 보는 조건")
        g = D["gen"].groupby(["model", "split"])["macro_f1"].agg(["mean", "std"]).reset_index()
        g["split"] = g["split"].map({"random 5×3": "무작위 분할 (이미 본 조건)", "group 5×3": "그룹 분할 (처음 보는 조건)"})
        fig = px.bar(g, x="model", y="mean", color="split", barmode="group", text=g["mean"].round(4))
        fig.update_layout(height=420, xaxis_title=None, yaxis_title="Macro F1 (15 fold 평균)", yaxis_range=[0.6, 0.9], legend_title=None,
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
        st.caption("plate_defect_count(같은 생산 조건의 결함 수)는 새 조건에서 항상 0 → 처음 보는 조건 성능을 떨어뜨려 최종 모델에서 제외")

# ================= 3. 불균형 처리 =================
with tab3:
    imb, imb_pc = D["imb"], D["imb_pc"]
    st.subheader("처리 방법별 Macro F1 (5-fold CV)")
    pv = imb.pivot(index="model", columns="method", values="macro_f1")
    methods = ["처리 없음", "클래스 가중치", "SMOTENC (균등화)", "언더샘플링 (균등화)"]
    pv = pv[[m for m in methods if m in pv.columns]]
    left, right = st.columns([3, 2])
    with left:
        fig = px.imshow(pv, text_auto=".4f", color_continuous_scale="RdYlGn", zmin=0.65, zmax=0.83, aspect="auto")
        fig.update_layout(height=360, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        delta = pv.sub(pv["처리 없음"], axis=0).drop(columns="처리 없음")
        st.markdown("**처리 없음 대비 변화**")
        color = lambda v: f"color: {'#1a7f37' if v > 0 else '#cf222e'}; font-weight: 600"
        st.dataframe(delta.style.format("{:+.4f}").map(color), width="stretch")
        st.caption("언더샘플링은 학습 데이터가 1/5로 줄어 크게 악화. 부스팅 계열은 SMOTENC·클래스 가중치가 효과적")

    st.subheader("유형별 지표 전후 비교")
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("모델", list(pv.index), index=list(pv.index).index("GBM"))
    method = c2.selectbox("처리 방법", [m for m in methods if m != "처리 없음"])
    met = c3.selectbox("지표", ["recall", "precision", "f1"])
    d = imb_pc[(imb_pc["model"] == model) & (imb_pc["method"].isin(["처리 없음", method]))]
    fig = px.bar(d, x="class", y=met, color="method", barmode="group", category_orders={"class": CLASS_ORDER}, text=d[met].round(3))
    fig.update_layout(height=380, xaxis_title=None, legend_title=None, yaxis_range=[0, 1.05])
    st.plotly_chart(fig, width="stretch")
    st.caption("소수·혼동 유형(Pastry 등)의 Recall이 오르는 대신 Other_Faults Recall이 내려감 → 판별력 향상보다 판정 경계 이동에 가까움")

# ================= 4. 최종 모델 =================
with tab4:
    st.markdown(f"**ExtraTrees + GBM Soft Voting** · 35개 피처(원본 26 + 파생 9) · Test {S['n_test']}건 중 오분류 {S['n_errors']}건 · "
                f"Test Macro F1 95% CI {S['test_ci'][0]:.4f} – {S['test_ci'][1]:.4f}")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Confusion Matrix (Test)")
        norm = st.toggle("행 기준 비율(%)로 보기", value=False)
        cm = D["cm"]
        z = (cm.div(cm.sum(axis=1), axis=0) * 100).round(1) if norm else cm
        fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues", labels=dict(x="예측", y="실제", color="%" if norm else "건수"))
        fig.update_layout(height=470)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("유형별 Precision · Recall · F1 (Test)")
        pc = D["pc"]
        fig = px.bar(pc.melt(id_vars=["class", "support"], value_vars=["precision", "recall", "f1"], var_name="metric"),
                     x="class", y="value", color="metric", barmode="group")
        fig.update_layout(height=470, xaxis_title=None, legend_title=None, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")
    st.dataframe(pc.set_index("class").style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}), width="stretch")
    cm = D["cm"]
    pair = lambda a, b: int(cm.loc[a, b] + cm.loc[b, a])
    st.info(f"오분류 {S['n_errors']}건 중 Bumps↔Other_Faults {pair('Bumps', 'Other_Faults')}건, Pastry↔Other_Faults {pair('Pastry', 'Other_Faults')}건 "
            "→ 오류 대부분이 서로 닮은 세 유형의 경계에 집중. 건수가 가장 적은 Stains·Dirtiness는 오히려 잘 맞힘")

    left, right = st.columns(2)
    with left:
        st.subheader("변수 중요도 (Permutation, 상위 10)")
        imp = D["imp"].head(10).sort_values("importance")
        fig = px.bar(imp, x="importance", y="feature", orientation="h", text=imp["importance"].round(4))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, xaxis_title="Test Macro F1 감소량", yaxis_title=None, xaxis_range=[0, imp["importance"].max() * 1.2])
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("학습 곡선 (5-fold)")
        lc = D["lc"]
        fig = go.Figure()
        fig.add_scatter(x=lc["frac"] * 100, y=lc["train_f1"], name="Train F1", mode="lines+markers")
        fig.add_scatter(x=lc["frac"] * 100, y=lc["val_f1"], name="Validation F1", mode="lines+markers")
        fig.add_scatter(x=lc["frac"] * 100, y=lc["val_logloss"], name="Validation log loss", mode="lines+markers", yaxis="y2")
        fig.update_layout(height=420, xaxis_title="학습 데이터 비율 (%)", yaxis=dict(title="Macro F1"),
                          yaxis2=dict(title="log loss", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    st.caption("데이터가 늘수록 검증 F1↑·손실↓ → 분류 성능 기준 과적합 아님")
    if (FIG / "loss_curve.png").exists():
        with st.expander("손실 곡선 (트리 수에 따른 Log Loss)"):
            st.image(str(FIG / "loss_curve.png"), width="stretch")
            st.caption("Soft Voting은 검증 손실이 평탄하게 수렴. GBM 단독은 후반에 확률을 과신해 손실이 오르지만 앙상블에서 상쇄")
