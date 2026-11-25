"""
Fraud Detection System
----------------------
End-to-end ML pipeline: EDA -> preprocessing -> imbalance handling ->
multi-model training -> evaluation (Precision / Recall / F1 / ROC-AUC / PR-AUC)
-> threshold tuning -> feature importance.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve,
                             precision_recall_curve, classification_report)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
RS = 42

# ------------------------------------------------------------------ 1. LOAD
df = pd.read_csv("transactions.csv")
print("Shape:", df.shape)
print("Class distribution:\n", df.is_fraud.value_counts(normalize=True).round(4))

# ------------------------------------------------------------------ 2. EDA
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Exploratory Data Analysis - Fraud Patterns", fontsize=16, fontweight="bold")

ax[0, 0].pie(df.is_fraud.value_counts(), labels=["Legit", "Fraud"], autopct="%1.2f%%",
             colors=["#4C9F70", "#D64045"], explode=(0, .12), startangle=90)
ax[0, 0].set_title("Class Imbalance")

sns.histplot(data=df, x=np.log1p(df.amount), hue="is_fraud", bins=40, ax=ax[0, 1],
             stat="density", common_norm=False, palette=["#4C9F70", "#D64045"])
ax[0, 1].set_title("Transaction Amount (log scale)")

hr = df.groupby("hour").is_fraud.mean() * 100
ax[0, 2].bar(hr.index, hr.values, color="#D64045")
ax[0, 2].set_title("Fraud Rate by Hour of Day (%)")
ax[0, 2].set_xlabel("Hour")

cat = df.groupby("merchant_category").is_fraud.mean().sort_values() * 100
ax[1, 0].barh(cat.index, cat.values, color="#2E6F95")
ax[1, 0].set_title("Fraud Rate by Merchant Category (%)")

ch = df.groupby("payment_channel").is_fraud.mean() * 100
ax[1, 1].bar(ch.index, ch.values, color=["#E8A33D", "#2E6F95", "#4C9F70"])
ax[1, 1].set_title("Fraud Rate by Payment Channel (%)")

sns.boxplot(data=df, x="is_fraud", y="distance_from_home_km", ax=ax[1, 2],
            palette=["#4C9F70", "#D64045"], showfliers=False)
ax[1, 2].set_title("Distance From Home (km)")
plt.tight_layout()
plt.savefig(f"{OUT}/01_eda.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ 3. FEATURES
df["is_night"] = df.hour.isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
df["log_amount"] = np.log1p(df.amount)
df["amount_per_txn_24h"] = df.amount / (df.txns_last_24h + 1)
df["new_card_flag"] = (df.card_age_days < 90).astype(int)

target = "is_fraud"
num_cols = ["amount", "log_amount", "hour", "distance_from_home_km", "txns_last_24h",
            "card_age_days", "customer_age", "amount_per_txn_24h"]
bin_cols = ["is_foreign", "device_new", "is_night", "new_card_flag"]
cat_cols = ["merchant_category", "payment_channel"]

X, y = df[num_cols + bin_cols + cat_cols], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("bin", "passthrough", bin_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)
feat_names = (num_cols + bin_cols + list(pre.named_transformers_["cat"]["oh"].get_feature_names_out(cat_cols)))

# ------------------------------------------------------------------ 4. IMBALANCE (train only!)
def random_oversample(X_, y_, ratio=0.5):
    """Oversample minority class up to `ratio` of majority (applied ONLY to training data)."""
    y_ = np.asarray(y_)
    Xmaj, Xmin = X_[y_ == 0], X_[y_ == 1]
    n_target = int(len(Xmaj) * ratio)
    Xmin_up = resample(Xmin, replace=True, n_samples=n_target, random_state=RS)
    Xb = np.vstack([Xmaj, Xmin_up])
    yb = np.hstack([np.zeros(len(Xmaj)), np.ones(len(Xmin_up))])
    idx = np.random.RandomState(RS).permutation(len(yb))
    return Xb[idx], yb[idx]

Xtr_bal, ytr_bal = random_oversample(Xtr, y_train, ratio=0.5)
print(f"\nTrain before: {np.bincount(y_train)}  | after resampling: {np.bincount(ytr_bal.astype(int))}")

# ------------------------------------------------------------------ 5. MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RS),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                            n_jobs=-1, random_state=RS),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                    max_depth=3, random_state=RS),
}

results, probs = [], {}
for name, m in models.items():
    m.fit(Xtr_bal, ytr_bal)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    probs[name] = p
    results.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, p),
        "PR-AUC": average_precision_score(y_test, p),
    })
res = pd.DataFrame(results).set_index("Model").round(4).sort_values("F1-Score", ascending=False)
print("\n=== Model Comparison (threshold = 0.5) ===\n", res)
res.to_csv(f"{OUT}/model_comparison.csv")

best = res.index[0]
print("\nBest model:", best)

# ------------------------------------------------------------------ 6. PLOTS
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
res[["Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax[0], color=["#2E6F95", "#E8A33D", "#4C9F70"])
ax[0].set_title("Model Comparison"); ax[0].set_ylim(0, 1.05); ax[0].tick_params(axis="x", rotation=20)
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[1].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax[2].plot(rc, pr, label=f"{name} ({average_precision_score(y_test, p):.3f})")
ax[1].plot([0, 1], [0, 1], "k--", alpha=.4); ax[1].set_title("ROC Curves"); ax[1].legend(fontsize=8)
ax[1].set_xlabel("False Positive Rate"); ax[1].set_ylabel("True Positive Rate")
ax[2].set_title("Precision-Recall Curves"); ax[2].legend(fontsize=8)
ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision")
plt.tight_layout(); plt.savefig(f"{OUT}/02_model_comparison.png", dpi=130); plt.close()

# ------------------------------------------------------------------ 7. THRESHOLD TUNING
p_best = probs[best]
ths = np.arange(0.05, 0.96, 0.05)
rows = [(t, precision_score(y_test, p_best >= t), recall_score(y_test, p_best >= t),
         f1_score(y_test, p_best >= t)) for t in ths]
th = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])
best_t = th.loc[th.f1.idxmax(), "threshold"]
print(f"\nOptimal threshold by F1: {best_t:.2f}")

plt.figure(figsize=(8, 5))
for c, col in zip(["precision", "recall", "f1"], ["#2E6F95", "#E8A33D", "#4C9F70"]):
    plt.plot(th.threshold, th[c], marker="o", label=c.title(), color=col)
plt.axvline(best_t, color="#D64045", ls="--", label=f"Best threshold = {best_t:.2f}")
plt.title(f"Threshold Tuning - {best}"); plt.xlabel("Decision Threshold"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/03_threshold_tuning.png", dpi=130); plt.close()

final_pred = (p_best >= best_t).astype(int)
print("\n=== Final Classification Report ===")
print(classification_report(y_test, final_pred, target_names=["Legit", "Fraud"], digits=4))

cm = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.title(f"Confusion Matrix - {best}\n(threshold {best_t:.2f})")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig(f"{OUT}/04_confusion_matrix.png", dpi=130); plt.close()

# ------------------------------------------------------------------ 8. FEATURE IMPORTANCE
m = models[best]
imp = getattr(m, "feature_importances_", None)
if imp is None:
    imp = np.abs(m.coef_[0])
fi = pd.Series(imp, index=feat_names).sort_values().tail(12)
plt.figure(figsize=(8, 5.5))
fi.plot(kind="barh", color="#2E6F95")
plt.title(f"Top Fraud Indicators - {best}")
plt.tight_layout(); plt.savefig(f"{OUT}/05_feature_importance.png", dpi=130); plt.close()
print("\nTop 5 features:\n", fi.tail(5)[::-1].round(4))

# Business impact (illustrative)
tn, fp, fn, tp = cm.ravel()
print(f"\nCaught {tp} of {tp + fn} fraud cases ({tp / (tp + fn):.1%}); "
      f"{fp} legit transactions flagged ({fp / (fp + tn):.2%} of legit).")
