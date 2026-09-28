# -*- coding: utf-8 -*-
"""
고령 보행자 사고다발지역과 노인보호구역의 공간·유형 불일치 분석

입력 (data/)
  - 전국교통사고다발지역표준데이터_폴리곤제외.csv  (한국도로교통공단, 공공데이터포털)
  - 전국노인장애인보호구역표준데이터.csv          (행정안전부 표준데이터, 공공데이터포털)
  - 주민등록_5세단위_시군구_202608.csv            (행정안전부 주민등록 인구통계)

출력 (out/)
  - fig1_place_type.png  장소유형 분포 대조
  - fig2_distance.png    최근접 보호구역 거리 분포
  - fig3_sigungu.png     시군구 위험-대응 산점도
  - fig4_speed_cctv.png  보호구역 안전시설 현황
  - table_*.csv          수치표
  - results.json         본문에 인용한 모든 수치

실행:  python3 02_analyze.py
"""
import json
import os
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figprint import figsize_mm, save

# 분석보고서 양식의 그림 칸에서 실제로 쓸 수 있는 폭(figprint.slot_mm 실측, 2026-09-15).
# 전에는 이 값을 재지 않고 9.2인치(234mm)로 그려 150mm로 욱여넣어 그림이 잘리고 글자가 뭉갰다.
W = 121.9
matplotlib.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

R = {}   # 본문 인용 수치 저장소

# 색상(무채색 + 강조 1색) — 인쇄·흑백 복사에도 구분되도록 명도차를 둔다
C_RISK, C_ZONE, C_GREY = "#C0392B", "#2C5F8A", "#9AA0A6"


# ──────────────────────────── 1. 적재 ────────────────────────────
def load():
    acc = pd.read_csv(os.path.join(DATA, "전국교통사고다발지역표준데이터_폴리곤제외.csv"))
    zone = pd.read_csv(os.path.join(DATA, "전국노인장애인보호구역표준데이터.csv"), encoding="cp949")
    pop = pd.read_csv(os.path.join(DATA, "주민등록_5세단위_시군구_202608.csv"), encoding="cp949")
    return acc, zone, pop


def norm_sgg(s):
    """'서울특별시 종로구 (1111000000)' → ('서울특별시','종로구')"""
    s = re.sub(r"\s*\([0-9]+\)\s*$", "", str(s)).strip()
    parts = s.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


# ──────────────────────────── 2. 장소 유형 분류 ────────────────────────────
# 사고 지점 위치명(괄호 안 부가설명)과 보호구역 대상시설명을 같은 축으로 분류한다.
# 한 지점이 여러 범주에 걸치므로(예: '약수역사거리') 우선순위를 두어 주범주 하나를 준다.
PLACE_RULES = [
    ("전통시장·상권", r"시장|상가|상점|쇼핑|마트|백화점|패션타운|도매|벼룩"),
    ("역·터미널",     r"[가-힣0-9]역(?![원장사무])|터미널|정류장|정류소|[0-9]호선|전철|지하철"),
    ("의료·약국",     r"약국|병원|의원|보건소|한의원|치과|메디컬"),
    ("복지·경로시설", r"경로당|노인정|노인회관|마을회관|복지관|복지회관|요양|노인복지|경로회관|다목적회관"),
    ("행정·공공",     r"주민센터|행정복지|구청|시청|군청|면사무소|읍사무소|지구대|파출소|우체국|도서관|학교|어린이집|유치원"),
    ("교차로·도로",   r"사거리|삼거리|오거리|교차로|로터리|지하차도|육교|고가|횡단보도|입구|앞"),
]


def classify(text):
    t = str(text)
    for name, pat in PLACE_RULES:
        if re.search(pat, t):
            return name
    return "기타"


def place_of_accident(name):
    """'서울특별시 중구 신당동(약수역 6호선 부근)' → '약수역 6호선'"""
    m = re.search(r"\(([^)]*)\)", str(name))
    body = m.group(1) if m else str(name)
    return re.sub(r"\s*부근\s*$", "", body).strip()


# ──────────────────────────── 3. 거리 ────────────────────────────
def haversine_matrix(lat1, lon1, lat2, lon2):
    """(n,) x (m,) → (n,m) 미터. 지구반경 6,371,008.8 m (IUGG 평균반경)."""
    la1 = np.radians(lat1)[:, None]
    lo1 = np.radians(lon1)[:, None]
    la2 = np.radians(lat2)[None, :]
    lo2 = np.radians(lon2)[None, :]
    dla = la2 - la1
    dlo = lo2 - lo1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ──────────────────────────── 4. 본 분석 ────────────────────────────
def main():
    acc, zone, pop = load()

    # 4-1. 대상 표본 확정 ---------------------------------------------------
    # 사고다발지역 선정기준은 2021년에 바뀌었다(반경 200m·1년 → 반경 100m·최근 3년,
    # 사망·중상 5건 이상). 기준이 다른 연도를 섞으면 개소 수 비교가 성립하지 않으므로
    # 최신 스냅샷인 2024년 지정분(=2022~2024년 사고 누적)만 본 분석의 대상으로 삼는다.
    old = acc[(acc["사고유형구분"] == "보행노인")].copy()
    ped = old[old["사고연도"] == 2024].copy()
    child = acc[(acc["사고유형구분"].isin(["보행어린이", "스쿨존어린이"])) & (acc["사고연도"] == 2024)].copy()
    bike = acc[(acc["사고유형구분"] == "자전거") & (acc["사고연도"] == 2024)].copy()

    R["기준연도"] = 2024
    R["보행노인_다발지_개소"] = int(len(ped))
    R["보행노인_사고건수"] = int(ped["사고건수"].sum())
    R["보행노인_사상자수"] = int(ped["사상자수"].sum())
    R["보행노인_사망자수"] = int(ped["사망자수"].sum())
    R["보행노인_중상자수"] = int(ped["중상자수"].sum())
    R["보행어린이_다발지_개소"] = int(len(child))
    R["자전거_다발지_개소"] = int(len(bike))
    R["보행노인_전기간_개소"] = int(len(old))
    R["사고데이터_기준일"] = str(acc["데이터기준일자"].max())

    zone_old = zone[zone["장소유형코드"] == 1].copy()     # 1 = 노인보호구역
    zone_dis = zone[zone["장소유형코드"] == 2].copy()     # 2 = 장애인보호구역
    R["노인보호구역_개소"] = int(len(zone_old))
    R["장애인보호구역_개소"] = int(len(zone_dis))
    R["보호구역데이터_기준일"] = str(zone["데이터기준일자"].max())
    R["보호구역_등록시군구수"] = int(zone_old[["시도명", "시군구명"]].drop_duplicates().shape[0])

    # 4-2. 장소 유형 대조 ---------------------------------------------------
    ped["장소명"] = ped["사고지역위치명"].map(place_of_accident)
    ped["장소유형"] = ped["장소명"].map(classify)
    zone_old["장소유형"] = zone_old["대상시설명"].map(classify)

    order = [n for n, _ in PLACE_RULES] + ["기타"]
    t_acc = ped["장소유형"].value_counts().reindex(order).fillna(0).astype(int)
    t_zone = zone_old["장소유형"].value_counts().reindex(order).fillna(0).astype(int)
    tbl = pd.DataFrame({
        "고령 보행자 사고다발지역(개소)": t_acc,
        "고령 보행자 사고다발지역(%)": (t_acc / t_acc.sum() * 100).round(1),
        "노인보호구역(개소)": t_zone,
        "노인보호구역(%)": (t_zone / t_zone.sum() * 100).round(1),
    })
    tbl.to_csv(os.path.join(OUT, "table_place_type.csv"), encoding="utf-8-sig")

    R["사고_복지경로시설_비율"] = float(tbl.loc["복지·경로시설", "고령 보행자 사고다발지역(%)"])
    R["보호구역_복지경로시설_비율"] = float(tbl.loc["복지·경로시설", "노인보호구역(%)"])
    R["사고_시장상권_비율"] = float(tbl.loc["전통시장·상권", "고령 보행자 사고다발지역(%)"])
    R["보호구역_시장상권_비율"] = float(tbl.loc["전통시장·상권", "노인보호구역(%)"])
    R["사고_역터미널_비율"] = float(tbl.loc["역·터미널", "고령 보행자 사고다발지역(%)"])
    R["보호구역_역터미널_비율"] = float(tbl.loc["역·터미널", "노인보호구역(%)"])
    R["사고_생활이동축_비율"] = round(
        R["사고_시장상권_비율"] + R["사고_역터미널_비율"] + float(tbl.loc["의료·약국", "고령 보행자 사고다발지역(%)"]), 1)
    R["보호구역_생활이동축_비율"] = round(
        R["보호구역_시장상권_비율"] + R["보호구역_역터미널_비율"] + float(tbl.loc["의료·약국", "노인보호구역(%)"]), 1)
    R["사고_의료약국_비율"] = float(tbl.loc["의료·약국", "고령 보행자 사고다발지역(%)"])
    R["보호구역_의료약국_비율"] = float(tbl.loc["의료·약국", "노인보호구역(%)"])

    fig, ax = plt.subplots(figsize=figsize_mm(W, 0.52))
    y = np.arange(len(order))
    ax.barh(y + 0.19, tbl["고령 보행자 사고다발지역(%)"], 0.36, color=C_RISK, label=f"고령 보행자 사고다발지역 (n={t_acc.sum()})")
    ax.barh(y - 0.19, tbl["노인보호구역(%)"], 0.36, color=C_ZONE, label=f"노인보호구역 (n={t_zone.sum()})")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("구성비 (%)")
    ax.set_title("[그림 1] 노인은 '이동하는 곳'에서 다치고,\n보호구역은 '머무는 곳'에 있다", pad=8)
    for i, (a, b) in enumerate(zip(tbl["고령 보행자 사고다발지역(%)"], tbl["노인보호구역(%)"])):
        ax.text(a + 0.7, i + 0.19, f"{a:g}%", va="center", fontsize=6.8, color=C_RISK)
        ax.text(b + 0.7, i - 0.19, f"{b:g}%", va="center", fontsize=6.8, color=C_ZONE)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(tbl["고령 보행자 사고다발지역(%)"].max(), tbl["노인보호구역(%)"].max()) * 1.18)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig1_place_type.png"), W)
    plt.close(fig)

    # 4-3. 공간 불일치 -------------------------------------------------------
    d = haversine_matrix(ped["위도"].values, ped["경도"].values,
                         zone_old["위도"].values, zone_old["경도"].values)
    nearest = d.min(axis=1)
    ped["최근접보호구역_m"] = nearest
    ped["최근접보호구역명"] = zone_old["대상시설명"].values[d.argmin(axis=1)]

    for th in (300, 500, 1000):
        R[f"커버율_{th}m"] = round(float((nearest <= th).mean() * 100), 1)
    R["최근접거리_중앙값_m"] = int(np.median(nearest))
    R["최근접거리_평균_m"] = int(nearest.mean())

    # 보호구역이 한 곳이라도 등록된 시군구로 한정한 부분집합(데이터 커버리지 보정)
    zkey = set(zip(zone_old["시도명"], zone_old["시군구명"]))
    ped[["시도", "시군구"]] = ped["사고다발지역시도시군구"].map(
        lambda s: pd.Series(str(s).split()[:2] if len(str(s).split()) >= 2 else [str(s), ""])).apply(pd.Series) \
        if False else pd.DataFrame(
        [(str(s).split()[0], re.sub(r"\d+$", "", str(s).split()[1]) if len(str(s).split()) > 1 else "")
         for s in ped["사고다발지역시도시군구"]], index=ped.index, columns=["시도", "시군구"])
    sub = ped[[(a, b) in zkey for a, b in zip(ped["시도"], ped["시군구"])]]
    R["보호구역등록시군구_내_사고다발지"] = int(len(sub))
    R["보호구역등록시군구_내_커버율_300m"] = round(float((sub["최근접보호구역_m"] <= 300).mean() * 100), 1)

    ped.sort_values("최근접보호구역_m", ascending=False).to_csv(
        os.path.join(OUT, "table_accident_points_2024.csv"), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=figsize_mm(W, 0.50))
    bins = np.arange(0, 5001, 250)
    ax.hist(np.clip(nearest, 0, 5000), bins=bins, color=C_GREY, edgecolor="white")
    ax.axvline(300, color=C_RISK, lw=2)
    ax.text(330, ax.get_ylim()[1] * 0.88,
            f"보호구역 지정 반경 300m\n이 안에 든 사고다발지 {R['커버율_300m']}%",
            color=C_RISK, fontsize=7.0, va="top")
    ax.set_xlabel("가장 가까운 노인보호구역까지의 거리 (m · 5,000m 초과는 5,000)")
    ax.set_ylabel("사고다발지역 수 (개소)")
    ax.set_title(f"[그림 2] 고령 보행자 사고다발지역 {len(ped)}개소의 절반은\n"
                 f"보호구역에서 {R['최근접거리_중앙값_m']:,}m 넘게 떨어져 있다", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig2_distance.png"), W)
    plt.close(fig)

    # 4-3b. 역방향: 이미 지정된 보호구역은 위험지점을 얼마나 끼고 있나 ---------
    rev = d.min(axis=0)                       # 보호구역 각각 → 가장 가까운 사고다발지
    R["보호구역_사고다발지300m내_비율"] = round(float((rev <= 300).mean() * 100), 1)
    R["보호구역_사고다발지300m내_개소"] = int((rev <= 300).sum())
    R["보호구역_사고다발지1km밖_비율"] = round(float((rev > 1000).mean() * 100), 1)

    # 4-3c. 전국 배치도 -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize_mm(W, 1.19))
    ax.scatter(zone_old["경도"], zone_old["위도"], s=5, color=C_ZONE, alpha=.35,
               label=f"노인보호구역 {len(zone_old):,}개소", linewidths=0)
    ax.scatter(ped["경도"], ped["위도"], s=26, color=C_RISK, alpha=.85,
               label=f"고령 보행자 사고다발지역 {len(ped)}개소", edgecolor="white", linewidths=.4)
    ax.set_xlabel("경도"); ax.set_ylabel("위도")
    ax.set_aspect(1 / np.cos(np.radians(36)))
    ax.set_title("[그림 5] 노인보호구역(파랑)과\n고령 보행자 사고다발지역(빨강)은 서로 다른 곳에 있다", pad=8, fontsize=8.0)
    ax.legend(loc="lower left", frameon=False, fontsize=6.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig5_map.png"), W)
    plt.close(fig)

    # 4-4. 시군구 단위 위험 대비 대응 ----------------------------------------
    pop = pop.copy()
    pop[["시도", "시군구"]] = pd.DataFrame(
        [norm_sgg(x) for x in pop["행정구역"]], index=pop.index, columns=["시도", "시군구"])
    agecols = [c for c in pop.columns if "_계_" in c and re.search(r"(6[5-9]|[7-9]\d|100)", c.split("_계_")[1])]
    agecols = [c for c in agecols if re.match(r"(6[5-9]|7\d|8\d|9\d)~|100세", c.split("_계_")[1])]
    for c in agecols:
        pop[c] = pd.to_numeric(pop[c].astype(str).str.replace(",", ""), errors="coerce")
    pop["고령인구"] = pop[agecols].sum(axis=1)
    pop_sgg = pop[pop["시군구"] != ""][["시도", "시군구", "고령인구"]]
    R["전국_고령인구"] = int(pop_sgg["고령인구"].sum())
    R["인구기준월"] = "2026-08"

    a_cnt = ped.groupby(["시도", "시군구"]).agg(
        사고다발지=("사고지역관리번호", "count"),
        사상자=("사상자수", "sum"),
        사망자=("사망자수", "sum")).reset_index()
    z_cnt = zone_old.groupby(["시도명", "시군구명"]).size().rename("노인보호구역").reset_index() \
        .rename(columns={"시도명": "시도", "시군구명": "시군구"})
    m = pop_sgg.merge(a_cnt, on=["시도", "시군구"], how="left").merge(z_cnt, on=["시도", "시군구"], how="left")
    m[["사고다발지", "사상자", "사망자", "노인보호구역"]] = m[["사고다발지", "사상자", "사망자", "노인보호구역"]].fillna(0)
    m["고령1만명당_사고다발지"] = (m["사고다발지"] / m["고령인구"] * 10000).round(3)
    m["사고다발지1개소당_보호구역"] = np.where(m["사고다발지"] > 0, m["노인보호구역"] / m["사고다발지"], np.nan)
    m.sort_values("고령1만명당_사고다발지", ascending=False).to_csv(
        os.path.join(OUT, "table_sigungu.csv"), index=False, encoding="utf-8-sig")

    have = m[m["사고다발지"] > 0].copy()
    R["사고다발지_보유_시군구수"] = int(len(have))
    R["사고다발지_있으나_보호구역0_시군구수"] = int((have["노인보호구역"] == 0).sum())
    R["사고다발지_있으나_보호구역0_지점수"] = int(have.loc[have["노인보호구역"] == 0, "사고다발지"].sum())

    top = have.sort_values("고령1만명당_사고다발지", ascending=False).head(10)
    R["위험도상위10_시군구"] = [
        {"시군구": f"{r['시도']} {r['시군구']}", "고령1만명당": float(r["고령1만명당_사고다발지"]),
         "사고다발지": int(r["사고다발지"]), "노인보호구역": int(r["노인보호구역"]),
         "고령인구": int(r["고령인구"])} for _, r in top.iterrows()]

    fig, ax = plt.subplots(figsize=figsize_mm(W, 0.62))
    ax.scatter(have["고령1만명당_사고다발지"], have["노인보호구역"], s=28, color=C_GREY, alpha=.75, edgecolor="white")
    med_x = have["고령1만명당_사고다발지"].median()
    med_y = have["노인보호구역"].median()
    ax.axvline(med_x, color="#CCCCCC", lw=1)
    ax.axhline(med_y, color="#CCCCCC", lw=1)
    risky = have[(have["고령1만명당_사고다발지"] > med_x) & (have["노인보호구역"] <= med_y)]
    ax.scatter(risky["고령1만명당_사고다발지"], risky["노인보호구역"], s=40, color=C_RISK, edgecolor="white", zorder=3)
    R["미스매치_시군구수"] = int(len(risky))
    R["미스매치_시군구"] = [f"{r['시도']} {r['시군구']}" for _, r in
                       risky.sort_values("고령1만명당_사고다발지", ascending=False).head(12).iterrows()]
    lab = risky.sort_values("고령1만명당_사고다발지", ascending=False).head(6)
    for k, (_, r) in enumerate(lab.iterrows()):
        dy = 9 if k % 2 == 0 else -14      # 위아래 번갈아 배치해 라벨 겹침을 막는다
        ax.annotate(f"{r['시도'][:2]} {r['시군구']}", (r["고령1만명당_사고다발지"], r["노인보호구역"]),
                    textcoords="offset points", xytext=(5, dy * 0.8), fontsize=6.6, color=C_RISK)
    ax.set_xlabel("고령인구 1만 명당 고령 보행자 사고다발지역 수 (개소)")
    ax.set_ylabel("노인보호구역 지정 수 (개소)")
    ax.set_title(f"[그림 3] 위험은 높은데 보호구역은 적은 시군구 {len(risky)}곳 (붉은 점)", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig3_sigungu.png"), W)
    plt.close(fig)

    # 4-5. 지정된 보호구역의 설비 상태 ---------------------------------------
    sp = pd.to_numeric(zone_old["제한속도"], errors="coerce")
    R["보호구역_제한속도30이하_비율"] = round(float((sp <= 30).sum() / sp.notna().sum() * 100), 1)
    R["보호구역_제한속도40이상_개소"] = int((sp >= 40).sum())
    R["보호구역_CCTV없음_비율"] = round(float((zone_old["CCTV설치여부"] == "N").mean() * 100), 1)
    R["보호구역_CCTV없음_개소"] = int((zone_old["CCTV설치여부"] == "N").sum())
    width = pd.to_numeric(zone_old["보호구역도로폭"], errors="coerce")
    R["보호구역_도로폭_중앙값_cm"] = int(width.median())

    fig, axes = plt.subplots(2, 1, figsize=figsize_mm(W, 1.02))
    v = zone_old["CCTV설치여부"].value_counts()
    axes[0].bar(["미설치", "설치"], [v.get("N", 0), v.get("Y", 0)], color=[C_RISK, C_ZONE], width=.55)
    for i, val in enumerate([v.get("N", 0), v.get("Y", 0)]):
        axes[0].text(i, val + 40, f"{val:,}개소", ha="center", fontsize=7.0)
    axes[0].set_title(f"노인보호구역 CCTV 설치 여부 (n={len(zone_old):,})", fontsize=7.8)
    axes[0].set_ylabel("개소")
    axes[0].spines[["top", "right"]].set_visible(False)

    grp = pd.cut(sp, [-1, 29, 30, 39, 49, 200], labels=["30 미만", "30", "31~39", "40~49", "50 이상"])
    g = grp.value_counts().reindex(["30 미만", "30", "31~39", "40~49", "50 이상"]).fillna(0)
    axes[1].bar(g.index.astype(str), g.values, color=[C_GREY, C_ZONE, C_GREY, C_RISK, C_RISK], width=.6)
    for i, val in enumerate(g.values):
        axes[1].text(i, val + 40, f"{int(val):,}", ha="center", fontsize=6.8)
    axes[1].set_title("노인보호구역 제한속도 (km/h)", fontsize=7.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("[그림 4] 지정만 하고 설비는 따라가지 않았다", y=0.995, fontsize=8.6)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig4_speed_cctv.png"), W)
    plt.close(fig)

    # 4-6. 우선 지정 후보 산출 ----------------------------------------------
    # 점수 = 사망자×3 + 중상자×1 (도로교통법상 보호구역 지정의 목적이 사망·중상 예방이므로
    # 사고 '건수'가 아니라 피해 심각도로 가중한다). 300m 안에 보호구역이 없는 지점만 후보.
    cand = ped[ped["최근접보호구역_m"] > 300].copy()
    cand["우선순위점수"] = cand["사망자수"] * 3 + cand["중상자수"]
    cand = cand.sort_values(["우선순위점수", "사상자수"], ascending=False)
    keep = ["사고지역위치명", "장소유형", "사고건수", "사망자수", "중상자수", "사상자수",
            "우선순위점수", "최근접보호구역_m", "최근접보호구역명", "위도", "경도"]
    cand[keep].to_csv(os.path.join(OUT, "table_priority_candidates.csv"), index=False, encoding="utf-8-sig")
    R["후보지점_개소"] = int(len(cand))
    R["후보지점_사상자"] = int(cand["사상자수"].sum())
    R["후보지점_사망자"] = int(cand["사망자수"].sum())
    R["상위20_후보"] = [
        {"위치": r["사고지역위치명"], "유형": r["장소유형"], "사망": int(r["사망자수"]),
         "중상": int(r["중상자수"]), "점수": int(r["우선순위점수"]),
         "최근접보호구역_m": int(r["최근접보호구역_m"])}
        for _, r in cand.head(20).iterrows()]

    # 상위 20개소가 전체 후보 피해에서 차지하는 비중(선택과 집중의 근거)
    R["상위20_사상자"] = int(cand.head(20)["사상자수"].sum())
    R["상위20_사상자_비중"] = round(R["상위20_사상자"] / R["후보지점_사상자"] * 100, 1)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    for k, v in R.items():
        if not isinstance(v, list):
            print(f"{k}: {v}")
    print("\n[장소유형 대조]")
    print(tbl.to_string())
    print("\n[위험도 상위 10 시군구]")
    print(top[["시도", "시군구", "고령인구", "사고다발지", "노인보호구역", "고령1만명당_사고다발지"]].to_string(index=False))
    print("\n[우선 지정 후보 상위 15]")
    print(cand.head(15)[["사고지역위치명", "장소유형", "사망자수", "중상자수", "우선순위점수", "최근접보호구역_m"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
