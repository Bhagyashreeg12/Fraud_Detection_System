"""Generate a realistic synthetic credit-card transaction dataset (~1.5% fraud).

Replace the output CSV with a real dataset (e.g. Kaggle "Credit Card Fraud
Detection") and the rest of the pipeline works with minimal changes.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 60_000
FRAUD_RATE = 0.015

is_fraud = rng.random(N) < FRAUD_RATE
n_f, n_l = is_fraud.sum(), (~is_fraud).sum()

df = pd.DataFrame({"is_fraud": is_fraud.astype(int)})

# Amount: fraud skews higher
df.loc[~is_fraud, "amount"] = rng.lognormal(3.4, 1.0, n_l)
df.loc[is_fraud, "amount"] = rng.lognormal(5.0, 1.2, n_f)

# Hour of day: legit = daytime, fraud = night-heavy
legit_hours = rng.choice(24, n_l, p=np.array([1,1,1,1,1,2,4,6,8,8,8,8,8,8,8,8,8,8,8,7,5,3,2,1]) / 
                         np.array([1,1,1,1,1,2,4,6,8,8,8,8,8,8,8,8,8,8,8,7,5,3,2,1]).sum())
fraud_hours = rng.choice(24, n_f, p=np.array([9,9,9,9,8,4,2,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4,6,8]) /
                         np.array([9,9,9,9,8,4,2,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4,6,8]).sum())
df.loc[~is_fraud, "hour"] = legit_hours
df.loc[is_fraud, "hour"] = fraud_hours

cats = ["grocery", "fuel", "restaurant", "travel", "electronics", "clothing", "online_services", "jewelry"]
df.loc[~is_fraud, "merchant_category"] = rng.choice(cats, n_l, p=[.28,.18,.2,.07,.07,.1,.08,.02])
df.loc[is_fraud, "merchant_category"] = rng.choice(cats, n_f, p=[.03,.03,.05,.14,.28,.1,.2,.17])

df["payment_channel"] = np.where(
    is_fraud, rng.choice(["online", "pos", "atm"], N, p=[.72, .18, .10]),
    rng.choice(["online", "pos", "atm"], N, p=[.35, .55, .10]))

df["distance_from_home_km"] = np.where(is_fraud, rng.exponential(250, N), rng.exponential(12, N)).round(1)
df["txns_last_24h"] = np.where(is_fraud, rng.poisson(5, N), rng.poisson(1.6, N))
df["card_age_days"] = np.where(is_fraud, rng.integers(1, 400, N), rng.integers(30, 3000, N))
df["is_foreign"] = np.where(is_fraud, rng.random(N) < .45, rng.random(N) < .04).astype(int)
df["device_new"] = np.where(is_fraud, rng.random(N) < .6, rng.random(N) < .07).astype(int)
df["customer_age"] = rng.integers(18, 80, N)

# Make ~12% of fraud "subtle": it mimics normal behaviour (hard cases, like real life)
feat = ["amount", "hour", "merchant_category", "payment_channel", "distance_from_home_km",
        "txns_last_24h", "card_age_days", "is_foreign", "device_new"]
fraud_idx = df.index[df.is_fraud == 1]
subtle = rng.choice(fraud_idx, int(len(fraud_idx) * 0.12), replace=False)
legit_pool = df.index[df.is_fraud == 0]
donors = rng.choice(legit_pool, len(subtle), replace=False)
df.loc[subtle, feat] = df.loc[donors, feat].values
# ...and ~0.1% of legit transactions look suspicious (false-alarm cases)
odd = rng.choice(legit_pool, int(len(legit_pool) * 0.001), replace=False)
fdon = rng.choice(fraud_idx, len(odd), replace=True)
df.loc[odd, feat] = df.loc[fdon, feat].values

# Inject realistic missing values / noise
df.loc[rng.random(N) < .02, "distance_from_home_km"] = np.nan
df.loc[rng.random(N) < .01, "merchant_category"] = np.nan
df["amount"] = df["amount"].round(2)
df["hour"] = df["hour"].astype(int)

df.insert(0, "transaction_id", np.arange(1, N + 1))
df.to_csv("transactions.csv", index=False)
print(df.shape, "fraud rate:", df.is_fraud.mean().round(4))
