# -*- coding: utf-8 -*-
"""
반복 출현 지점 분석 — 한 해만 튄 곳인가, 해마다 같은 자리인가

02_analyze.py가 2024년 스냅샷 한 장을 보호구역과 대조했다면, 이 스크립트는
2021~2024년 네 장의 스냅샷을 겹쳐 '같은 자리가 몇 번 다발지로 뽑혔는지'를 센다.
선정기준이 2021년에 바뀌었으므로(반경 200m·1년 → 반경 100m·최근 3년) 2021년 이후만 쓴다.

지정 시점 정보가 공개 데이터에 없어 '보호구역이 생긴 뒤 사고가 줄었는가'는 물을 수 없다.
그래서 인과를 주장하지 않고, 반복 횟수라는 기술통계만으로 우선순위를 세운다.

입력 (data/)  02_analyze.py와 동일
출력 (out/)   fig6_repeat.png · table_repeat_points.csv · results_persistence.json

실행:  python3 03_persistence.py
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figprint import figsize_mm, save

W = 121.9          # 분석보고서 그림 칸의 실제 폭(figprint.slot_mm 실측, 2026-09-15)
matplotlib.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
C_RISK, C_ZONE, C_GREY = "#C0392B", "#2C5F8A", "#9AA0A6"
R = {}

# 선정기준 반경이 100m이므로 두 해의 지점이 100m 안에 있으면 같은 자리로 본다.
SAME_PLACE_M = 100


def haversine_matrix(lat1, lon1, lat2, lon2):
    la1, lo1 = np.radians(lat1)[:, None], np.radians(lon1)[:, None]
    la2, lo2 = np.radians(lat2)[None, :], np.radians(lon2)[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cluster(df):
    """100m 이내를 한 자리로 묶는다(단일연결). 지점 수가 1,004개라 전수 거리행렬로 충분하다."""
    d = haversine_matrix(df["위도"].values, df["경도"].values, df["위도"].values, df["경도"].values)
    n = len(df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in np.where(d[i] <= SAME_PLACE_M)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    return [find(i) for i in range(n)]


def main():
    acc = pd.read_csv(os.path.join(DATA, "전국교통사고다발지역표준데이터_폴리곤제외.csv"))
    zone = pd.read_csv(os.path.join(DATA, "전국노인장애인보호구역표준데이터.csv"), encoding="cp949")
    zone_old = zone[zone["장소유형코드"] == 1]

    ped = acc[(acc["사고유형구분"] == "보행노인") & (acc["사고연도"].between(2021, 2024))].copy()
    ped = ped.reset_index(drop=True)
    R["대상기간"] = "2021~2024"
    R["기간내_지정건수"] = int(len(ped))

    ped["자리"] = cluster(ped)
    g = ped.groupby("자리")
    site = pd.DataFrame({
        "반복횟수": g["사고연도"].nunique(),
        "위치명": g["사고지역위치명"].last(),
        "최근연도": g["사고연도"].max(),
        "누적사고건수": g["사고건수"].sum(),
        "누적사망자수": g["사망자수"].sum(),
        "누적중상자수": g["중상자수"].sum(),
        "위도": g["위도"].mean(),
        "경도": g["경도"].mean(),
    }).reset_index(drop=True)

    R["서로다른자리_개소"] = int(len(site))
    dist = site["반복횟수"].value_counts().sort_index()
    R["반복분포"] = {int(k): int(v) for k, v in dist.items()}
    R["반복2회이상_개소"] = int((site["반복횟수"] >= 2).sum())
    R["반복2회이상_비율"] = round(float((site["반복횟수"] >= 2).mean() * 100), 1)
    R["반복3회이상_개소"] = int((site["반복횟수"] >= 3).sum())
    R["반복4회_개소"] = int((site["반복횟수"] == 4).sum())

    # 반복 자리가 피해에서 차지하는 비중
    rep = site[site["반복횟수"] >= 2]
    R["반복2회이상_누적사고건수"] = int(rep["누적사고건수"].sum())
    R["전체_누적사고건수"] = int(site["누적사고건수"].sum())
    R["반복2회이상_사고비중"] = round(R["반복2회이상_누적사고건수"] / R["전체_누적사고건수"] * 100, 1)

    # 보호구역과의 거리
    d = haversine_matrix(site["위도"].values, site["경도"].values,
                         zone_old["위도"].values, zone_old["경도"].values)
    site["최근접보호구역_m"] = d.min(axis=1)
    site["최근접보호구역명"] = zone_old["대상시설명"].values[d.argmin(axis=1)]
    site["보호구역300m내"] = site["최근접보호구역_m"] <= 300

    R["반복2회이상_보호구역300m밖_개소"] = int(((site["반복횟수"] >= 2) & ~site["보호구역300m내"]).sum())
    R["반복2회이상_보호구역300m밖_비율"] = round(
        float((~site.loc[site["반복횟수"] >= 2, "보호구역300m내"]).mean() * 100), 1)
    R["반복1회_보호구역300m밖_비율"] = round(
        float((~site.loc[site["반복횟수"] == 1, "보호구역300m내"]).mean() * 100), 1)

    # 최우선 후보: 2회 이상 반복 × 보호구역 300m 밖
    top = site[(site["반복횟수"] >= 2) & (~site["보호구역300m내"])].copy()
    top["피해점수"] = top["누적사망자수"] * 3 + top["누적중상자수"]
    top = top.sort_values(["반복횟수", "피해점수"], ascending=False)
    top[["위치명", "반복횟수", "최근연도", "누적사고건수", "누적사망자수", "누적중상자수",
         "피해점수", "최근접보호구역_m", "최근접보호구역명", "위도", "경도"]].to_csv(
        os.path.join(OUT, "table_repeat_points.csv"), index=False, encoding="utf-8-sig")
    R["최우선후보_개소"] = int(len(top))
    R["최우선후보_누적사망자"] = int(top["누적사망자수"].sum())
    R["최우선후보_누적중상자"] = int(top["누적중상자수"].sum())
    R["최우선후보_상위10"] = [
        {"위치": r["위치명"], "반복": int(r["반복횟수"]), "사망": int(r["누적사망자수"]),
         "중상": int(r["누적중상자수"]), "보호구역거리_m": int(r["최근접보호구역_m"])}
        for _, r in top.head(10).iterrows()]

    # 어린이와의 제도 비대칭: 사고유형구분 자체에 '스쿨존어린이' 범주가 있다
    for y in (2024,):
        sub = acc[acc["사고연도"] == y]
        sz = int((sub["사고유형구분"] == "스쿨존어린이").sum())
        cw = int((sub["사고유형구분"] == "보행어린이").sum())
        R["어린이_스쿨존범주_개소"] = sz
        R["어린이_보행범주_개소"] = cw
        R["어린이_보호구역내범주_비율"] = round(sz / (sz + cw) * 100, 1)
    R["노인_보호구역내범주_존재"] = bool((acc["사고유형구분"].str.contains("노인")
                                 & acc["사고유형구분"].str.contains("보호구역|실버")).any())
    R["사고유형구분_전체"] = sorted(acc["사고유형구분"].unique().tolist())

    # 그림 6
    fig, axes = plt.subplots(2, 1, figsize=figsize_mm(W, 1.05))
    xs = [1, 2, 3, 4]
    ys = [R["반복분포"].get(k, 0) for k in xs]
    axes[0].bar([str(x) + "회" for x in xs], ys, color=[C_GREY, C_RISK, C_RISK, C_RISK], width=.6)
    for i, v in enumerate(ys):
        axes[0].text(i, v + 6, f"{v:,}", ha="center", fontsize=7.0)
    axes[0].set_title(f"2021~2024년 같은 자리가 뽑힌 횟수 (n={len(site):,}자리)", fontsize=7.8)
    axes[0].set_ylabel("자리 수 (개소)")
    axes[0].spines[["top", "right"]].set_visible(False)

    cats = ["1회만 뽑힌 자리", "2회 이상 반복된 자리"]
    vals = [R["반복1회_보호구역300m밖_비율"], R["반복2회이상_보호구역300m밖_비율"]]
    axes[1].bar(cats, vals, color=[C_GREY, C_RISK], width=.5)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 1.2, f"{v}%", ha="center", fontsize=7.0)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("보호구역 300m 밖 비율 (%)")
    axes[1].set_title("반복된 자리라고 해서 더 보호받고 있지 않다", fontsize=7.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("[그림 6] 해마다 같은 자리에서 다친다", y=0.995, fontsize=8.6)
    fig.tight_layout()
    save(fig, os.path.join(OUT, "fig6_repeat.png"), W)
    plt.close(fig)

    with open(os.path.join(OUT, "results_persistence.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    for k, v in R.items():
        print(f"{k}: {v}")
    print("\n[최우선 후보 상위 15]")
    print(top.head(15)[["위치명", "반복횟수", "누적사망자수", "누적중상자수", "최근접보호구역_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
