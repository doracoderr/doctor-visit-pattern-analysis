"""Doctor Visit Pattern Analysis - AICTE TIRTC DIY Project"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

PATH = "1776250375-P2-Healthcare_Analytics_for_Doctor_Visits__1_.csv"
import os
os.makedirs("out2/charts/", exist_ok=True)
OUT = "out2/charts/"
TEAL, ORANGE = "#0F766E", "#E07A1F"

# 1. Load and prepare
raw = pd.read_csv(PATH).drop(columns=["Unnamed: 0"])
df = raw.copy()
df["age_years"] = (df["age"] * 100).round().astype(int)
for c in ["private", "freepoor", "freerepat", "nchronic", "lchronic"]:
    df[c] = (df[c] == "yes").astype(int)
df["female"] = (df["gender"] == "female").astype(int)
df["visited"] = (df["visits"] > 0).astype(int)
print(df.shape, df.isna().sum().sum(), df["visited"].mean())

# 2. Charts
fig, ax = plt.subplots(figsize=(7.5, 5.6))
cols = ["visits", "age", "income", "illness", "reduced", "health", "private", "freepoor", "freerepat", "nchronic", "lchronic", "female"]
sns.heatmap(df[cols].corr(), cmap="YlGnBu", annot=True, fmt=".2f", annot_kws={"size": 7}, cbar=False, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(OUT + "heatmap.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ag = pd.cut(df["age_years"], [0, 25, 35, 50, 65, 100], labels=["<=25", "26-35", "36-50", "51-65", "65+"])
df.groupby(ag, observed=True)["visited"].mean().mul(100).plot.bar(ax=ax, color=TEAL)
ax.set(title="Share of People Who Visited a Doctor, by Age Group", xlabel="Age group", ylabel="% with 1+ visit")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(OUT + "age_share.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 4.2))
groups = {"Male": df.female == 0, "Female": df.female == 1, "Private cover": df.private == 1, "No private": df.private == 0,
          "Free-poor": df.freepoor == 1, "Free-repat": df.freerepat == 1, "Long-term chronic": df.lchronic == 1}
vals = {k: df.loc[v, "visits"].mean() for k, v in groups.items()}
pd.Series(vals).plot.barh(ax=ax, color=[ORANGE if k in ("Long-term chronic", "Free-repat") else TEAL for k in vals])
ax.set(title="Average Visits by Group", xlabel="Average visits")
plt.tight_layout(); plt.savefig(OUT + "group_avg.png", dpi=170); plt.close()

# 3. Models
feats = ["female", "age", "income", "illness", "reduced", "health", "private", "freepoor", "freerepat", "nchronic", "lchronic"]
Xtr, Xte, ytr, yte = train_test_split(df[feats], df["visited"], test_size=0.25, random_state=7, stratify=df["visited"])
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
gb = GradientBoostingClassifier(random_state=7).fit(Xtr, ytr)
res = {}
for name, m, X in [("lr", lr, sc.transform(Xte)), ("gb", gb, Xte)]:
    p = m.predict(X); pr = m.predict_proba(X)[:, 1]
    res[name] = dict(acc=accuracy_score(yte, p), auc=roc_auc_score(yte, pr), recall=recall_score(yte, p))
print(res)
odds = pd.Series(np.exp(lr.coef_[0]), index=feats).sort_values()
print(odds)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
odds.plot.barh(ax=ax, color=[ORANGE if v < 1 else TEAL for v in odds])
ax.axvline(1, color="black", lw=1)
ax.set(title="Logistic Regression Odds Ratios (per 1 std. dev.)", xlabel="Odds ratio (>1 = more likely to visit)")
plt.tight_layout(); plt.savefig(OUT + "odds_ratio.png", dpi=170); plt.close()
json.dump(dict(res=res, odds=odds.to_dict(), vals=vals), open("out2/metrics.json", "w"), indent=1)
