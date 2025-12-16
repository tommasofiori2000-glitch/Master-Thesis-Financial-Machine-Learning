# ============================================================
# MARKET PIPELINE — LOCALE (CLEAN, KAGGLE-COMPATIBLE)
# ============================================================
# - Usa SOLO feature disponibili nel test Kaggle:
#       * lagged_forward_returns
#       * lagged_risk_free_rate
#       * lagged_market_forward_excess_returns
#       * D*, E*, I*, M*, P*, S*, V*
# - NO risk_free_rate come feature
# - NO market_forward_excess_returns
# - NO rolling sul target / mercato
# - Include:
#       * Training base XGB
#       * Residual ElasticNet (NON piatto)
#       * Walk-forward locale (Kaggle-like)
#       * Predizioni su test.csv
#       * Salvataggio artifact per Kaggle
#       * Diagnostica avanzata
# ============================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import xgboost as xgb


# ============================================================
# 0) PATHS & LOADING
# ============================================================

TRAIN_PATH = r"C:/Users/tommaso/Desktop/DATA SCIENCE/TESI/train.csv"
TEST_PATH  = r"C:/Users/tommaso/Desktop/DATA SCIENCE/TESI/test.csv"

print("=== LOADING DATA ===")
train_df = pd.read_csv(TRAIN_PATH, delimiter=";")
test_df  = pd.read_csv(TEST_PATH,  delimiter=";")

print("Train:", train_df.shape)
print("Test :", test_df.shape)

# Convert object → numeric
for c in train_df.select_dtypes(include=["object"]).columns:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

for c in test_df.select_dtypes(include=["object"]).columns:
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")


# ============================================================
# 1) FEATURE ENGINEERING COERENTE TRAIN/TEST
# ============================================================

# Train: forward_returns (target), risk_free_rate, market_forward_excess_returns
# Test: NO forward_returns, ha già lagged_* + is_scored

# 1a) costruiamo lagged_market_forward_excess_returns nel TRAIN se manca
if "lagged_market_forward_excess_returns" not in train_df.columns:
    if "market_forward_excess_returns" not in train_df.columns:
        raise ValueError("In train non trovo 'market_forward_excess_returns' per creare il lag.")
    train_df["lagged_market_forward_excess_returns"] = (
        train_df["market_forward_excess_returns"].shift(1)
    )

# 1b) togliamo righe con target mancante
train_df = train_df.dropna(subset=["forward_returns"]).reset_index(drop=True)
print("Train dopo drop di forward_returns NaN:", train_df.shape)

# 1c) costruiamo lag nel TRAIN come esistono nel TEST
if "lagged_forward_returns" not in train_df.columns:
    train_df["lagged_forward_returns"] = train_df["forward_returns"].shift(1)

if "lagged_risk_free_rate" not in train_df.columns:
    if "risk_free_rate" not in train_df.columns:
        raise ValueError("risk_free_rate mancante nel train: impossibile creare lagged_risk_free_rate.")
    train_df["lagged_risk_free_rate"] = train_df["risk_free_rate"].shift(1)

if "lagged_market_forward_excess_returns" not in train_df.columns:
    if "market_forward_excess_returns" in train_df.columns:
        train_df["lagged_market_forward_excess_returns"] = \
            train_df["market_forward_excess_returns"].shift(1)
    else:
        raise ValueError("market_forward_excess_returns mancante nel train: impossibile creare lag.")

# 1d) definizione feature lecite: LAG + D/E/I/M/P/S/V
lag_features = [
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns"
]

D_cols = [c for c in train_df.columns if c.startswith("D")]
E_cols = [c for c in train_df.columns if c.startswith("E")]
I_cols = [c for c in train_df.columns if c.startswith("I")]
M_cols = [c for c in train_df.columns if c.startswith("M")]
P_cols = [c for c in train_df.columns if c.startswith("P")]
S_cols = [c for c in train_df.columns if c.startswith("S")]
V_cols = [c for c in train_df.columns if c.startswith("V")]

feature_cols = lag_features + D_cols + E_cols + I_cols + M_cols + P_cols + S_cols + V_cols
feature_resid = lag_features.copy()  # modello residuo molto parsimonioso

print("\n=== FEATURE SET ===")
print("Numero totale feature:", len(feature_cols))
print("Prime 10 feature:", feature_cols[:10])
print("Resid features:", feature_resid)


# ============================================================
# 2) PREPROCESSING GLOBAL (IMPUTER + SCALER)
# ============================================================

imputer_base = SimpleImputer(strategy="median")
scaler_base  = StandardScaler()

imputer_resid = SimpleImputer(strategy="median")
scaler_resid  = StandardScaler()


# ============================================================
# 3) TRAIN BASE MODEL (XGB) SU TUTTO IL TRAIN
# ============================================================

print("\n=== TRAINING BASE MODEL (XGB) SU TUTTO IL TRAIN ===")

X_train = train_df[feature_cols]
X_train_imp = imputer_base.fit_transform(X_train)
X_train_sc  = scaler_base.fit_transform(X_train_imp)

y_train = train_df["forward_returns"].values.astype(float)

xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)

xgb_model.fit(X_train_sc, y_train)
base_pred_train = xgb_model.predict(X_train_sc)

print("Base train pred stats: mean", base_pred_train.mean(), "std", base_pred_train.std())


# ============================================================
# 4) TRAIN RESIDUAL MODEL (ElasticNet "vivo")
# ============================================================

print("\n=== TRAINING RESIDUAL MODEL (ElasticNet) ===")

residuals = y_train - base_pred_train
print("Residuals stats: mean", residuals.mean(), "std", residuals.std())

X_res_train = train_df[feature_resid]
X_res_imp   = imputer_resid.fit_transform(X_res_train)
X_res_sc    = scaler_resid.fit_transform(X_res_imp)

# Parametri scelti più "morbidi" per evitare soluzione nulla
resid_model = ElasticNet(alpha=1e-4, l1_ratio=0.1, max_iter=10000)
resid_model.fit(X_res_sc, residuals)

resid_pred_train = resid_model.predict(X_res_sc)
final_pred_train = base_pred_train + 0.5 * resid_pred_train  # 0.5 = attenuazione

print("\nResidual model prediction stats:")
print("  mean:", resid_pred_train.mean())
print("  std :", resid_pred_train.std())
if abs(resid_pred_train.std()) < 1e-6:
    print("  ⚠ WARNING: residual std ~ 0 (modello residuo quasi piatto).")
else:
    print("  ✅ residual model NON piatto.")

print("Final train pred stats: mean", final_pred_train.mean(), "std", final_pred_train.std())


# ============================================================
# 5) WALK-FORWARD EVALUATION (KAGGLE-LIKE)
# ============================================================

def sharpe_annualized(returns, td=252):
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    return returns.mean() / std * np.sqrt(td)


def walk_forward_kaggle_like(df, feature_cols, feature_resid, batch=20, min_train=1000):
    """
    Valutazione walk-forward stile Kaggle:
    - Allena da 0 su df[:start]
    - Valida su df[start:end]
    - Usa XGB + ElasticNet residual + mapping tanh → alloc 0-2
    """

    n = len(df)
    all_alloc = []
    all_rets  = []

    print(f"\n=== WALK-FORWARD: total={n}, min_train={min_train}, batch={batch} ===")

    for start in range(min_train, n, batch):
        end = min(start + batch, n)
        print(f"  -> Train up to {start}, validate {start}:{end}")

        df_tr = df.iloc[:start]
        df_va = df.iloc[start:end]

        X_tr = df_tr[feature_cols]
        y_tr = df_tr["forward_returns"].values

        imp_b = SimpleImputer(strategy="median")
        sc_b  = StandardScaler()
        X_tr_imp = imp_b.fit_transform(X_tr)
        X_tr_sc  = sc_b.fit_transform(X_tr_imp)

        model_b = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist"
        )
        model_b.fit(X_tr_sc, y_tr)

        # Residual model per fold
        X_res_tr = df_tr[feature_resid]
        imp_r = SimpleImputer(strategy="median")
        sc_r  = StandardScaler()
        X_res_tr_imp = imp_r.fit_transform(X_res_tr)
        X_res_tr_sc  = sc_r.fit_transform(X_res_tr_imp)

        base_tr_pred = model_b.predict(X_tr_sc)
        resid_tr     = y_tr - base_tr_pred

        model_r = ElasticNet(alpha=1e-4, l1_ratio=0.1, max_iter=5000)
        model_r.fit(X_res_tr_sc, resid_tr)

        # Validation
        X_va = df_va[feature_cols]
        X_va_sc = sc_b.transform(imp_b.transform(X_va))
        base_va_pred = model_b.predict(X_va_sc)

        X_res_va = df_va[feature_resid]
        X_res_va_sc = sc_r.transform(imp_r.transform(X_res_va))
        resid_va_pred = model_r.predict(X_res_va_sc)

        final_va_pred = base_va_pred + 0.5 * resid_va_pred

        # Allocation mapping
        z = (final_va_pred - final_va_pred.mean()) / (final_va_pred.std() + 1e-8)
        alloc = 1.0 + np.tanh(1.5 * z)
        alloc = np.clip(alloc, 0, 2)

        all_alloc.append(alloc)
        all_rets.append(df_va["forward_returns"].values)

    all_alloc = np.concatenate(all_alloc)
    all_rets  = np.concatenate(all_rets)

    strat = all_alloc * all_rets
    return sharpe_annualized(strat)


print("\n=== WALK-FORWARD (KAGGLE-LIKE) SU TRAIN ===")
sh_kagg = walk_forward_kaggle_like(
    train_df,
    feature_cols,
    feature_resid,
    batch=20,
    min_train=1000
)
print("\nSharpe Kaggle-like (train, WF):", sh_kagg)
print("===========================================================\n")


# ============================================================
# 6) PREVISIONI SU TEST.CSV (LOCALE)
# ============================================================

print("\n=== PREDIZIONI SU TEST.CSV (LOCALE) ===")

def predict_local(df):
    """
    Replica la pipeline: XGB base + ElasticNet residual + mapping tanh → [0,2]
    """

    # Controlliamo che tutte le feature ci siano
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("⚠ Missing columns in test:", missing)
        for c in missing:
            df[c] = 0.0  # fallback neutro

    X_base = df[feature_cols].copy()
    X_res  = df[feature_resid].copy()

    Xb_imp = imputer_base.transform(X_base)
    Xb_sc  = scaler_base.transform(Xb_imp)
    base_pred = xgb_model.predict(Xb_sc)

    Xr_imp = imputer_resid.transform(X_res)
    Xr_sc  = scaler_resid.transform(Xr_imp)
    resid_pred = resid_model.predict(Xr_sc)

    final_pred = base_pred + 0.5 * resid_pred

    z = (final_pred - final_pred.mean()) / (final_pred.std() + 1e-8)
    alloc = 1.0 + np.tanh(1.5 * z)
    alloc = np.clip(alloc, 0, 2)

    out = pd.DataFrame({
        "date_id": df["date_id"].astype(int) if "date_id" in df.columns else np.arange(len(df)),
        "prediction": alloc
    })
    return out, final_pred, base_pred, resid_pred


# garantiamo che ci sia date_id nel test locale
if "date_id" not in test_df.columns:
    test_df["date_id"] = np.arange(len(test_df))

local_pred, final_loc, base_loc, resid_loc = predict_local(test_df.copy())

print("\n=== HEAD LOCAL PRED ===")
print(local_pred.head(10))

print("\nSTATISTICHE LOCAL PRED:")
print("Min :", local_pred["prediction"].min())
print("Max :", local_pred["prediction"].max())
print("Mean:", local_pred["prediction"].mean())
print("Std :", local_pred["prediction"].std())

print("\n=== DISTRIBUZIONE (10 BINS) ===")
print(local_pred["prediction"].value_counts(bins=10).sort_index())


# ============================================================
# 7) DIAGNOSTICA AVANZATA SU TRAIN
# ============================================================

print("\n==============================================================")
print("                DIAGNOSTICA AVANZATA — TRAIN")
print("==============================================================\n")

y = y_train
true_sign  = np.sign(y)
base_sign  = np.sign(base_pred_train)
final_sign = np.sign(final_pred_train)
mask_nz = true_sign != 0

acc_base  = (true_sign[mask_nz] == base_sign[mask_nz]).mean() * 100
acc_final = (true_sign[mask_nz] == final_sign[mask_nz]).mean() * 100

print("1) Directional Accuracy")
print(f"  Base  : {acc_base:.2f}%")
print(f"  Final : {acc_final:.2f}%\n")

print("2) Correlazioni")
corr_base  = np.corrcoef(base_pred_train,  y)[0,1]
corr_resid = np.corrcoef(resid_pred_train, y)[0,1]
corr_final = np.corrcoef(final_pred_train, y)[0,1]
corr_br    = np.corrcoef(base_pred_train, resid_pred_train)[0,1]

print(f"  Corr(BASE, target):  {corr_base:.4f}")
print(f"  Corr(RESID, target): {corr_resid:.4f}")
print(f"  Corr(FINAL, target): {corr_final:.4f}")
print(f"  Corr(BASE, RESID):   {corr_br:.4f}\n")

print("3) Decile monotonicity su FINAL")
df_dec = pd.DataFrame({"y": y, "pred": final_pred_train})
df_dec["decile"] = pd.qcut(df_dec["pred"], 10, labels=False, duplicates="drop")
dec_stats = df_dec.groupby("decile").agg(
    mean_y=("y", "mean"),
    std_y=("y", "std"),
    count=("y", "count"),
    mean_pred=("pred", "mean")
)
print(dec_stats, "\n")

def dist_summary(arr, name):
    arr = np.asarray(arr)
    q = np.percentile(arr, [1,5,50,95,99])
    print(f"{name}:")
    print(f"  mean {arr.mean(): .6f}")
    print(f"  std  {arr.std(): .6f}")
    print(f"  min  {arr.min(): .6f}")
    print(f"  1%   {q[0]: .6f}")
    print(f"  5%   {q[1]: .6f}")
    print(f"  50%  {q[2]: .6f}")
    print(f"  95%  {q[3]: .6f}")
    print(f"  99%  {q[4]: .6f}")
    print(f"  max  {arr.max(): .6f}\n")

print("4) Distribution summary\n")
dist_summary(base_pred_train,  "Base pred")
dist_summary(resid_pred_train, "Residual pred")
dist_summary(final_pred_train, "Final pred")
dist_summary(y,                "Target")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(base_pred_train, bins=40, alpha=0.6, label="Base")
plt.hist(final_pred_train, bins=40, alpha=0.6, label="Final")
plt.legend()
plt.title("Distribuzione BASE vs FINAL")

plt.subplot(1,2,2)
plt.scatter(y, final_pred_train, alpha=0.3, s=10)
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.title("True vs FINAL pred")
plt.xlabel("forward_returns")
plt.ylabel("final_pred")
plt.tight_layout()
plt.show()

print("\n==============================================================")
print("        DIAGNOSTICA AVANZATA — COMPLETATA")
print("==============================================================\n")


# ============================================================
# 8) SALVATAGGIO ARTIFACT PER KAGGLE
# ============================================================

EXPORT_DIR = "export_model_kaggle/"
os.makedirs(EXPORT_DIR, exist_ok=True)

print("\n=== SALVATAGGIO ARTIFACT IN", EXPORT_DIR, "===\n")

# 1) feature lists
with open(os.path.join(EXPORT_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)

with open(os.path.join(EXPORT_DIR, "feature_resid.json"), "w") as f:
    json.dump(feature_resid, f)

# 2) imputers & scalers
pickle.dump(imputer_base,  open(os.path.join(EXPORT_DIR, "imputer_base.pkl"),  "wb"))
pickle.dump(scaler_base,   open(os.path.join(EXPORT_DIR, "scaler_base.pkl"),   "wb"))
pickle.dump(imputer_resid, open(os.path.join(EXPORT_DIR, "imputer_resid.pkl"), "wb"))
pickle.dump(scaler_resid,  open(os.path.join(EXPORT_DIR, "scaler_resid.pkl"),  "wb"))

# 3) residual model
pickle.dump(resid_model, open(os.path.join(EXPORT_DIR, "elastic_residual.pkl"), "wb"))

# 4) XGB model
xgb_model.save_model(os.path.join(EXPORT_DIR, "xgb_base.json"))

print("Files creati in", EXPORT_DIR, ":")
for fname in os.listdir(EXPORT_DIR):
    print("  -", fname)

print("\n✅ PIPELINE LOCALE COMPLETATA.")
print("   Sharpe WF locale (Kaggle-like):", sh_kagg)
print("   Pronto per esportare il modello su Kaggle.")
