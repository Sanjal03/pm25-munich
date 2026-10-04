"""
model_factory.py — Shared Model Factory & Evaluation Logic
=============================================================
This is the single source of truth for every model used in the
Champion/Challenger comparison. Phase 1 (benchmark), Phase 2 (RQ1
ablation) and Phase 3 (RQ2 counterfactual) all import `train_and_eval()`
from this file — never duplicate model code in a phase script. That is
the point of the "Model Factory" pattern you need to defend in your
Methodology chapter: performance differences between models can only be
attributed to the model itself if every model saw an identical
train/test split, identical preprocessing, and identical metric code.

Two families of model, because they consume different input shapes:

  TABULAR family  (lightgbm, xgboost, catboost)
    Consume the full ~36-column engineered feature table from
    03_feature_engineering.py (lags, rolling stats, co-pollutants,
    meteorology, temporal encodings).

  SEQUENCE family  (dlinear, patchtst)
    Consume a contiguous 168-hour raw PM2.5 lookback window built
    directly from the `target` column's continuous time series.
    Reason: the engineered lag features are intentionally sparse
    (1,2,3,6,12,24,48,168h) for tree-model efficiency — that is NOT a
    valid lookback tensor for a sequence model. Rebuilding a dense
    168h window from the same underlying (already-cleaned, already
    gap-flagged) target series keeps the comparison fair: every model
    sees the same information and is evaluated on the exact same set
    of timestamps.

TASK DEFINITION
  Every model estimates PM2.5 at hour t from PM2.5 up to t-1 plus the
  covariates observed AT hour t: co-pollutants (PM10, NO2, O3),
  pm25_background (mean PM2.5 of the other 6 stations), DWD weather and
  calendar features. This is one-hour-ahead estimation with concurrently
  observed network covariates, not a pure forecast from information
  available before t.

  BASELINE
    persistence — predicts y(t) = y(t-1). Mandatory sanity floor: if a
    "real" model can't beat this, it isn't learning anything.

Metrics: R², MAE, RMSE, Bias (mean(pred - actual); sign matters for the
RQ2 counterfactual gap and is worth reporting for every model, not just
the DL ones).
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

LOOKBACK = 168  # 7 days, hourly

# Study windows — the single definition every script imports. End points are
# explicit last hours: a bare date like "2020-05-04" compares as 00:00, which
# would silently drop the rest of that day from a window.
TRAIN_END                    = "2023-12-31 23:00"                       # train 2019-2023, test 2024
GATE_START, GATE_END         = "2020-01-15 00:00", "2020-03-21 23:00"   # Validity Gate, 1608 h
LOCKDOWN_START, LOCKDOWN_END = "2020-03-22 00:00", "2020-05-04 23:00"   # lockdown, 1056 h

TABULAR_MODELS  = ["lightgbm", "xgboost", "catboost"]
SEQUENCE_MODELS = ["dlinear", "patchtst"]
ALL_MODELS      = TABULAR_MODELS + SEQUENCE_MODELS
NON_FEATURE_COLS = ["target", "station", "split"]

FEATURE_GROUPS = {
    "PM2.5_lags":    [f"pm25_lag{l}h" for l in [1, 2, 3, 6, 12, 24, 48, 168]],
    "PM2.5_rolling": ["pm25_roll6h", "pm25_roll24h", "pm25_roll168h", "pm25_roll24h_std"],
    "Co-pollutants": ["pm10", "no2", "o3", "pm25_background"],
    "Meteorology":   ["temp_c", "humidity_pct", "wind_speed", "wind_u", "wind_v",
                       "pressure", "pressure_diff_24h", "temp_diff_1h"],
    "Temporal":      ["hour_sin", "hour_cos", "month_sin", "month_cos",
                       "dow_sin", "dow_cos", "is_weekend", "is_heating", "year"],
}

# Exogenous covariates for the SEQUENCE family (dlinear, patchtst) — matched
# to what the tabular models see, minus PM2.5 lags/rolling (the raw lookback
# window already covers that for sequence models). Shared by build_sequence_frame
# (training/eval) and recursive_counterfactual_forecast (RQ2 rollout) so the two
# code paths can never drift out of sync with each other.
SEQUENCE_EXO_COLS = [
    "pm10", "no2", "o3", "pm25_background",
    "temp_c", "humidity_pct", "wind_speed", "wind_u", "wind_v",
    "pressure", "pressure_diff_24h", "temp_diff_1h",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "is_weekend", "is_heating",
]


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_station_data(station: str) -> pd.DataFrame:
    path = Path("features") / f"{station}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 03_feature_engineering.py first — {path} missing")
    return pd.read_csv(path, index_col="datetime", parse_dates=True)


def tabular_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_sequence_frame(df: pd.DataFrame, lookback: int = LOOKBACK,
                          exo_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Builds a lookback-window table indexed the same way as `df`, so it can
    be filtered by date range or by df['split'] exactly like the tabular
    features. Each row i holds target[i-lookback : i] as columns t-168..t-1
    plus the hour-t exogenous covariates, which DLinear and PatchTST both
    consume (DLinear as extra regressors, PatchTST before its linear head).

    exo_cols overrides which exogenous columns get attached — defaults to
    SEQUENCE_EXO_COLS (the full set). This is what makes the RQ1 leave-
    one-group-out ablation meaningful for the sequence family: pass a
    restricted list (e.g. SEQUENCE_EXO_COLS minus the Meteorology columns)
    to actually remove that group's information from what DLinear/PatchTST
    see, the same way split_tabular's `feature_cols` does for the tree
    models. (The PM2.5 lookback window itself can't be restricted this way
    — see split_sequence's `shuffle_lookback` for how that group is
    ablated instead.)
    """
    target = df["target"].values
    n = len(df)
    if n <= lookback:
        raise ValueError("Not enough rows to build a lookback window")

    seq = np.full((n, lookback), np.nan, dtype=float)
    for i in range(lookback, n):
        seq[i] = target[i - lookback:i]

    seq_df = pd.DataFrame(seq, index=df.index,
                           columns=[f"t-{lookback-k}" for k in range(lookback)])
    seq_df["target"] = df["target"]
    seq_df["split"]  = df["split"]
    # Exogenous covariates — deliberately matched to what the tabular models
    # see (everything except PM2.5 lags/rolling, which the lookback window
    # already covers for sequence models). This gives DLinear/PatchTST a
    # fair comparison rather than an artificial univariate handicap.
    exo_cols = SEQUENCE_EXO_COLS if exo_cols is None else exo_cols
    for col in exo_cols:
        if col in df.columns:
            seq_df[col] = df[col]
    # A handful of exogenous columns retain NaNs from long-gap-flagged
    # periods (deliberately left unfilled in 02_preprocess.py). Forward-fill
    # is applied here only because DLinear/PatchTST cannot natively handle
    # missing values the way the tree models can; this affects a small
    # fraction of rows and is a pragmatic engineering step, not a change to
    # the missing-value policy itself.
    exo_present = [c for c in exo_cols if c in seq_df.columns]
    seq_df[exo_present] = seq_df[exo_present].ffill().bfill()
    return seq_df.dropna(subset=[f"t-{lookback}"] if lookback > 0 else None)


def split_tabular(df: pd.DataFrame, feature_cols: list[str],
                   date_range: tuple[str, str] | None = None,
                   exclude_range: tuple[str, str] | None = None):
    """
    Standard chronological split: train=2019-2023, test=2024 (via df['split']).
    Optionally restrict to a date_range (used by Phase 1/3 validity gate),
    or exclude a date_range from training (used by the gate & RQ2).
    """
    train = df[df["split"] == "train"]
    if exclude_range:
        lo, hi = exclude_range
        train = train[(train.index < lo) | (train.index > hi)]
    if date_range:
        lo, hi = date_range
        test = df[(df.index >= lo) & (df.index <= hi)]
    else:
        test = df[df["split"] == "test"]
    return (train[feature_cols], train["target"],
            test[feature_cols],  test["target"])


def split_sequence(seq_df: pd.DataFrame,
                    date_range: tuple[str, str] | None = None,
                    exclude_range: tuple[str, str] | None = None,
                    shuffle_lookback: bool = False, seed: int = 42):
    """
    shuffle_lookback=True is how the RQ1 ablation removes "PM2.5 history"
    for the sequence family. The raw 168h lookback window can't be
    partially zeroed the way tabular lag/rolling columns can be dropped —
    every value in it is real PM2.5 history, and zeroing it outright would
    collapse the array's variance to 0 and break DLinear/PatchTST's
    internal normalisation (mu/sigma computed from X.std()). Instead, each
    row's lookback window is randomly reassigned to a DIFFERENT row's
    window from the same split (train shuffled among train, test among
    test — never across the split boundary, so this never leaks 2024 test
    history into a 2019-2023 training row or vice versa). That keeps the
    array's realistic value distribution intact while destroying its
    correlation with that row's own target — the sequence-model equivalent
    of leave-one-group-out.
    """
    lookback_cols = [c for c in seq_df.columns if c.startswith("t-")]
    train = seq_df[seq_df["split"] == "train"]
    if exclude_range:
        lo, hi = exclude_range
        train = train[(train.index < lo) | (train.index > hi)]
    if date_range:
        lo, hi = date_range
        test = seq_df[(seq_df.index >= lo) & (seq_df.index <= hi)]
    else:
        test = seq_df[seq_df["split"] == "test"]
    extra_cols = [c for c in seq_df.columns if c not in lookback_cols + ["target", "split"]]

    Xtr, Xte = train[lookback_cols], test[lookback_cols]
    if shuffle_lookback:
        rng = np.random.default_rng(seed)
        Xtr = pd.DataFrame(rng.permutation(Xtr.values, axis=0),
                            index=Xtr.index, columns=Xtr.columns)
        Xte = pd.DataFrame(rng.permutation(Xte.values, axis=0),
                            index=Xte.index, columns=Xte.columns)

    return (Xtr, train[extra_cols], train["target"],
            Xte, test[extra_cols],  test["target"])


# ══════════════════════════════════════════════════════════════════════════
#  SEQUENCE MODELS (implemented as sklearn-style fit/predict estimators
#  so they slot into the exact same train_and_eval() call as the GBMs)
# ══════════════════════════════════════════════════════════════════════════

class DLinear:
    """DLinear (Zeng et al., 2023) — moving-average trend/residual decomposition
    + independent linear layers on each component. Closed-form least squares,
    no gradient descent needed."""

    def __init__(self, kernel_size=25):
        self.kernel_size = kernel_size

    def _decompose(self, X: np.ndarray):
        pad = self.kernel_size // 2
        padded = np.pad(X, ((0, 0), (pad, pad)), mode="edge")
        trend = np.array([
            padded[:, i:i + self.kernel_size].mean(axis=1) for i in range(X.shape[1])
        ]).T
        return trend, X - trend

    def fit(self, X, y, exo=None):
        # IMPORTANT: trend and residual branches must be fit JOINTLY (one
        # regression on the concatenated design matrix), not as two
        # independent regressions each targeting the full y — fitting them
        # separately makes both branches converge toward the same target
        # mean and the summed prediction ends up roughly double-biased.
        X, y = np.asarray(X, float), np.asarray(y, float)
        trend, resid = self._decompose(X)
        pieces = [trend, resid]
        if exo is not None:
            pieces.append(np.asarray(exo, float))
        design = np.hstack(pieces + [np.ones((len(trend), 1))])
        w = np.linalg.lstsq(design, y, rcond=None)[0]
        k = trend.shape[1]
        self.w_trend, self.w_resid = w[:k], w[k:2*k]
        self.w_exo = w[2*k:-1] if exo is not None else None
        self.bias = w[-1]
        return self

    def predict(self, X, exo=None):
        X = np.asarray(X, float)
        trend, resid = self._decompose(X)
        pred = trend @ self.w_trend + resid @ self.w_resid + self.bias
        if exo is not None and self.w_exo is not None:
            pred = pred + np.asarray(exo, float) @ self.w_exo
        return pred


def _require_torch():
    try:
        import torch
        torch.set_num_threads(1)
        return torch
    except ImportError as e:
        raise ImportError(
            "PatchTST requires PyTorch. Install with `pip install torch` "
            "(see requirements.txt)."
        ) from e


class _TorchSeqModel:
    """Shared training loop for PatchTST-lite so it reduces to the same
    fit()/predict() contract as the GBMs and DLinear.

    Exogenous covariates (when enabled) are handled by plain
    concatenation to the flattened patch embedding before the final
    linear head — deliberately NOT the attention-based "exogenous variate
    token" mechanism TimeXer uses. This is a simpler, cheaper way to give
    PatchTST fair access to the same weather/co-pollutant/temporal
    information the tabular models see, without reintroducing the added
    architectural complexity that was the reason TimeXer was scoped out.
    It is a modification of vanilla PatchTST (which is purely univariate)
    and should be described as such in the thesis, not cited as the
    unmodified Nie et al. (2023) architecture.
    """

    def __init__(self, lookback=LOOKBACK, patch_len=24, stride=24,
                 d_model=32, n_heads=4, n_layers=2, n_epochs=30,
                 batch_size=256, lr=1e-3, use_exogenous=True, seed=42):
        self.lookback, self.patch_len, self.stride = lookback, patch_len, stride
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.n_epochs, self.batch_size, self.lr = n_epochs, batch_size, lr
        self.use_exogenous = use_exogenous
        self.seed = seed
        self.model = None
        self.mu = self.sigma = None
        self.exo_mu = self.exo_sigma = None

    def _build_net(self, n_exo_features: int):
        torch = _require_torch()
        import torch.nn as nn

        n_patches = (self.lookback - self.patch_len) // self.stride + 1
        use_exo = self.use_exogenous and n_exo_features > 0

        class SeqNet(nn.Module):
            def __init__(self, patch_len, stride, n_patches, d_model, n_heads,
                         n_layers, n_exo, use_exo):
                super().__init__()
                self.patch_len, self.stride = patch_len, stride
                self.use_exo = use_exo
                self.input_proj = nn.Linear(patch_len, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                    batch_first=True, dropout=0.1,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                exo_dim = n_exo if use_exo else 0
                self.head = nn.Linear(n_patches * d_model + exo_dim, 1)

            def forward(self, x, exo=None):
                patches = x.unfold(1, self.patch_len, self.stride)   # (batch, n_patches, patch_len)
                z = self.input_proj(patches)
                z = self.encoder(z)
                z = z.flatten(1)                                     # (batch, n_patches * d_model)
                if self.use_exo and exo is not None:
                    z = torch.cat([z, exo], dim=1)                   # plain concatenation, no attention
                return self.head(z).squeeze(-1)

        torch.manual_seed(self.seed)
        return SeqNet(self.patch_len, self.stride, n_patches, self.d_model,
                       self.n_heads, self.n_layers, n_exo_features, use_exo)

    def fit(self, X, y, exo=None):
        torch = _require_torch()
        import torch.nn as nn
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mu, self.sigma = float(X.mean()), float(X.std() + 1e-8)
        Xn = (X - self.mu) / self.sigma
        yn = (y - self.mu) / self.sigma

        n_exo = 0
        exo_n = None
        if self.use_exogenous and exo is not None:
            exo = np.asarray(exo, dtype=np.float32)
            n_exo = exo.shape[1]
            self.exo_mu = exo.mean(axis=0)
            self.exo_sigma = exo.std(axis=0) + 1e-8
            exo_n = (exo - self.exo_mu) / self.exo_sigma

        self.model = self._build_net(n_exo)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(Xn)
        yt = torch.tensor(yn)
        Et = torch.tensor(exo_n) if exo_n is not None else None

        self.model.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                exo_batch = Et[idx] if Et is not None else None
                pred = self.model(Xt[idx], exo_batch)
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X, exo=None):
        torch = _require_torch()
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.mu) / self.sigma
        Xt = torch.tensor(Xn)
        Et = None
        if self.use_exogenous and exo is not None and self.exo_mu is not None:
            exo = np.asarray(exo, dtype=np.float32)
            exo_n = (exo - self.exo_mu) / self.exo_sigma
            Et = torch.tensor(exo_n)
        self.model.eval()
        with torch.no_grad():
            pred_n = self.model(Xt, Et).numpy()
        return pred_n * self.sigma + self.mu


class PatchTSTLite(_TorchSeqModel):
    """Simplified PatchTST-style patch Transformer, after Nie et al. (2023).
    Patches the raw PM2.5 lookback window (7 patches of 24 h), applies a
    small Transformer encoder, then concatenates weather/co-pollutant/
    temporal covariates before the final linear head (see _TorchSeqModel).

    Differences from the original, to state wherever this is described:
      - not univariate: exogenous covariates enter before the head;
      - no positional encoding: attention itself is order-blind across
        patches; patch order is only recovered by the flattened linear head;
      - one global mean/std normalisation instead of per-instance RevIN."""
    def __init__(self, **kw):
        super().__init__(use_exogenous=True, **kw)


class PersistenceModel:
    """Naive baseline: predict y(t) = y(t-1)."""
    def fit(self, X, y): return self
    def predict(self, X):
        X = np.asarray(X)
        return X[:, -1] if X.ndim == 2 else X   # last lookback value = t-1


# ══════════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════

def recursive_counterfactual_forecast(model, df: pd.DataFrame, feature_cols: list[str],
                                       start: str, end: str, model_type: str,
                                       lags=(1, 2, 3, 6, 12, 24, 48, 168)) -> pd.Series:
    """
    Rolls a trained model forward hour-by-hour through [start, end]
    WITHOUT letting the model see the station's own real lockdown-period
    PM2.5 values. (pm25_background — the other stations' observed PM2.5 —
    is a co-pollutant and stays observed; see the caveat below.)

    This exists because of a real methodological flaw in a naive
    approach: if you evaluate the model on the lockdown window using the
    already-computed feature table, the PM2.5 lag/rolling columns are
    built from the ACTUAL observed lockdown-period PM2.5 series. Since
    PM2.5 lags are by far the strongest feature group (see the RQ1
    ablation), feeding real, already-lockdown-affected lag values anchors
    every prediction close to the real value — shrinking any genuine
    counterfactual gap toward zero regardless of whether a real effect
    exists. That makes "no lockdown effect" an artifact of the evaluation
    method, not necessarily a finding about PM2.5.

    The fix: only the pre-lockdown history is real. From the first
    lockdown hour onward, every PM2.5 lag/rolling feature (tabular family)
    or lookback-window entry (sequence family) is rebuilt from the model's
    OWN previous predictions, exactly as a genuine "what would have
    happened without lockdown" simulation requires. Weather, co-pollutants,
    and temporal features are kept at their true observed values
    throughout — only the autoregressive PM2.5 signal is counterfactual.
    Because co-pollutants and pm25_background were themselves lowered by
    the lockdown, this partly anchors the counterfactual to the observed
    outcome, so the resulting gap is conservative (a lower bound).
    Prediction errors compound hour over hour (this is
    unavoidable in any counterfactual rollout, not a bug), so the
    resulting gap should be read as a trend-level signal, not an
    hour-by-hour point estimate.

    model_type must be passed explicitly (rather than inferred from the
    model instance) because the tabular and sequence families need
    entirely different input shapes at each rollout step — tabular models
    take the full engineered feature row, sequence models take a raw
    168h lookback array plus a separately-passed exogenous array (exactly
    matching how train_and_eval/build_sequence_frame trained them). Mixing
    these up produces a matrix-shape crash, not a silently wrong number,
    so getting model_type right here is load-bearing, not cosmetic.
    """
    history = df["target"].copy()  # real series; only entries >= start get overwritten below
    horizon = df.loc[start:end].index
    preds = {}

    if model_type in SEQUENCE_MODELS:
        exo_df = df[[c for c in SEQUENCE_EXO_COLS if c in df.columns]].ffill().bfill()
        for t in horizon:
            lookback_vals = [history.get(t - pd.Timedelta(hours=h), np.nan)
                              for h in range(LOOKBACK, 0, -1)]
            X_row = np.array([lookback_vals], dtype=float)
            exo_row = exo_df.loc[t].values.reshape(1, -1) if len(exo_df.columns) else None
            pred = float(model.predict(X_row, exo=exo_row)[0])
            preds[t] = pred
            history.loc[t] = pred  # feed forward — this IS the counterfactual assumption
        return pd.Series(preds)

    for t in horizon:
        row = df.loc[t, feature_cols].copy()

        for lag in lags:
            col = f"pm25_lag{lag}h"
            if col in row.index:
                lag_time = t - pd.Timedelta(hours=lag)
                row[col] = history.get(lag_time, np.nan)

        prior = history.loc[:t - pd.Timedelta(hours=1)]
        if "pm25_roll6h" in row.index:
            row["pm25_roll6h"] = prior.tail(6).mean()
        if "pm25_roll24h" in row.index:
            row["pm25_roll24h"] = prior.tail(24).mean()
        if "pm25_roll168h" in row.index:
            row["pm25_roll168h"] = prior.tail(168).mean()
        if "pm25_roll24h_std" in row.index:
            row["pm25_roll24h_std"] = prior.tail(24).std()

        X_row = pd.DataFrame([row[feature_cols].values], columns=feature_cols, index=[t])
        pred = float(model.predict(X_row)[0])
        preds[t] = pred
        history.loc[t] = pred  # feed forward — this IS the counterfactual assumption

    return pd.Series(preds)


def get_model(model_type: str):
    """Returns an untrained model instance with fixed hyperparameters —
    identical across every experiment in every phase, for fair comparison."""
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
        )
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0,
        )
    if model_type == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=600, depth=6, learning_rate=0.05,
            subsample=0.8, l2_leaf_reg=1.0, random_seed=42, thread_count=-1, verbose=0,
        )
    if model_type == "dlinear":
        return DLinear(kernel_size=25)
    if model_type == "patchtst":
        return PatchTSTLite(lookback=LOOKBACK)
    if model_type == "persistence":
        return PersistenceModel()
    raise ValueError(f"Unknown model_type: {model_type}")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "R2":   round(r2_score(y_true, y_pred), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "Bias": round(float(np.mean(np.asarray(y_pred) - np.asarray(y_true))), 3),
    }


def train_and_eval(model_type: str, df: pd.DataFrame, feature_cols: list[str] | None = None,
                    date_range=None, exclude_range=None, return_model=False,
                    sequence_exo_cols: list[str] | None = None,
                    shuffle_lookback: bool = False) -> dict:
    """
    THE single shared training/evaluation entry point. Every phase script
    calls this — never trains a model any other way. Handles both model
    families transparently so the caller doesn't need to know whether it's
    tabular or sequence-based.

    date_range / exclude_range let the same function serve: the standard
    2019-2023 / 2024 split, the Validity Gate window, and the pre-2020 /
    lockdown counterfactual split — all through one code path.

    sequence_exo_cols / shuffle_lookback are the SEQUENCE-family equivalent
    of `feature_cols` — plain `feature_cols` is silently ignored for
    dlinear/patchtst (they don't consume the tabular engineered columns at
    all), so RQ1's leave-one-group-out ablation needs its own hook into
    build_sequence_frame's exo-column restriction and split_sequence's
    lookback-shuffle to actually mean something for this family. Both
    default to "no ablation" (full exo set, real lookback window).
    """
    if model_type in TABULAR_MODELS:
        cols = feature_cols or tabular_feature_cols(df)
        X_train, y_train, X_test, y_test = split_tabular(
            df, cols, date_range=date_range, exclude_range=exclude_range)
        model = get_model(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type in SEQUENCE_MODELS:
        seq_df = build_sequence_frame(df, exo_cols=sequence_exo_cols)
        Xtr, Etr, ytr, Xte, Ete, yte = split_sequence(
            seq_df, date_range=date_range, exclude_range=exclude_range,
            shuffle_lookback=shuffle_lookback)
        model = get_model(model_type)
        model.fit(Xtr.values, ytr.values, exo=Etr.values)
        y_pred = model.predict(Xte.values, exo=Ete.values)
        y_test = yte

    elif model_type == "persistence":
        cols = feature_cols or tabular_feature_cols(df)
        X_train, y_train, X_test, y_test = split_tabular(
            df, cols, date_range=date_range, exclude_range=exclude_range)
        lag1_col = "pm25_lag1h"
        model = PersistenceModel()
        y_pred = X_test[lag1_col].values if lag1_col in X_test.columns else model.predict(X_test.values)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    metrics = compute_metrics(y_test, y_pred)
    metrics["n_test"] = len(y_test)
    if return_model:
        metrics["_model"] = model
        metrics["_y_pred"] = y_pred
        metrics["_y_test"] = y_test
        metrics["_test_index"] = y_test.index if hasattr(y_test, "index") else None
    return metrics
