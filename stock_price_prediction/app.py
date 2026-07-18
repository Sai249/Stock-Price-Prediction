
# app.py — Stock Price Prediction Dashboard
# Run: streamlit run app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# CUSTOM_CSS = '''
# <style>
# section[data-testid="stSidebar"] > div { background:#111827; padding:20px; color:#fff; }
# section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 { color:#fff !important; font-weight:700 !important; }
# .step-pill { display:flex; align-items:center; gap:.6rem; padding:.55rem .8rem; margin:.3rem 0; border-radius:10px; font-size:15px; font-weight:500; background:#1f2937; color:#e5e7eb; border:1px solid #374151; cursor:pointer; transition:all .15s; }
# .step-pill .idx { width:1.4rem; height:1.4rem; border-radius:999px; background:#4b5563; color:#fff; display:flex; align-items:center; justify-content:center; font-size:.75rem; }
# .step-pill.active { background:#7c3aed; border-color:#a78bfa; color:#fff; box-shadow:0 0 10px rgba(124,58,237,.35); }
# .step-pill.active .idx { background:#fff; color:#7c3aed; }
# .step-pill:hover { transform:translateX(3px); background:#334155; }
# </style>
# '''
CUSTOM_CSS = '''
<style>
/* Sidebar background */
section[data-testid="stSidebar"] > div {
  background: #111827; /* solid dark navy */
  padding: 20px;
  color: #ffffff;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* Step pill container */
.step-pill {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: .55rem .8rem;
  margin: .3rem 0;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 500;
  background: #1f2937;   /* slate-800 */
  color: #e5e7eb;        /* gray-200 */
  border: 1px solid #374151;
  cursor: pointer;
  transition: all 0.15s ease-in-out;
}

/* Number circle */
.step-pill .idx {
  width: 1.4rem;
  height: 1.4rem;
  border-radius: 999px;
  background: #4b5563; 
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: .75rem;
}

/* Active step */
.step-pill.active {
  background: #2563eb;   /* blue-600 */
  border-color: #3b82f6;
  color: #ffffff;
  box-shadow: 0 0 10px rgba(37, 99, 235, 0.4);
}
.step-pill.active .idx {
  background: white;
  color: #2563eb;
}

/* Hover effect */
.step-pill:hover {
  transform: translateX(3px);
  background: #334155;
}
</style>
'''

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def readable_int(n):
    try: return f"{int(n):,}"
    except: return str(n)

if "df" not in st.session_state: st.session_state["df"] = None
if "features_df" not in st.session_state: st.session_state["features_df"] = None

DATE_CANDIDATES = ["Date","date","Datetime","datetime","timestamp","Timestamp"]
TARGET_CANDIDATES = ["Close","Adj Close","close","adj_close"]

@st.cache_data(show_spinner=False)
def load_any(file_bytes: bytes, name: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if name.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(bio)
    else:
        df = pd.read_csv(bio, low_memory=False)
    for c in df.columns:
        if any(k in c.lower() for k in ["date","time","timestamp"]):
            try: df[c] = pd.to_datetime(df[c], errors="coerce")
            except: pass
    return df

@st.cache_data(show_spinner=False)
def compute_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    return miss[miss>0].to_frame("MissingCount")

def detect_first_present(cols, candidates):
    for c in candidates:
        if c in cols: return c
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

def make_lag_features(df: pd.DataFrame, tgt: str, lags=(1,2,3,5,10), windows=(5,10,20)):
    out = pd.DataFrame(index=df.index)
    out[tgt] = df[tgt]
    for L in lags:
        out[f"{tgt}_lag{L}"] = df[tgt].shift(L)
    for W in windows:
        out[f"{tgt}_sma{W}"] = df[tgt].rolling(W).mean().shift(1)
        out[f"{tgt}_ema{W}"] = df[tgt].ewm(span=W, adjust=False).mean().shift(1)
        out[f"{tgt}_rstd{W}"] = df[tgt].rolling(W).std().shift(1)
        out[f"{tgt}_mom{W}"] = df[tgt]/df[tgt].shift(W) - 1.0
    out["return_1"] = df[tgt].pct_change(1)
    return out

def to_csv_download(df, fname):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name=fname, mime="text/csv")

steps = [("Upload Data","📤"),("Preview & Columns","👀"),("Missing Values","🧩"),("Feature Engineering","🧪"),("EDA","📊"),("Train & Evaluate","🎯"),("Forecast","📈"),("Export","💾")]
step_titles = [s[0] for s in steps]
with st.sidebar:
    st.title("Stock Dashboard")
    current = st.radio("Navigate", step_titles, index=0, label_visibility="collapsed")
    st.markdown("### Steps")
    for idx,(title,icon) in enumerate(steps, start=1):
        cls = "step-pill active" if title==current else "step-pill"
        st.markdown(f'<div class="{cls}"><span class="idx">{idx}</span> {icon} {title}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Use time-based split for realistic evaluation.")

st.title("📈 Stock Price Prediction – E2E Dashboard")
st.caption("Upload → Preview → Missing → Feature Engineering → EDA → Train → Forecast → Export")

if current=="Upload Data":
    st.header("1) Upload Data")
    up = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"])
    if up is not None:
        df = load_any(up.getvalue(), up.name)
        st.session_state["df"] = df
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))
        to_csv_download(df, "uploaded_copy.csv")
    else:
        st.info("Upload a file to start.")

if current=="Preview & Columns":
    st.header("2) Preview & Columns")
    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a file first.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Rows", readable_int(df.shape[0]))
        with c2: st.metric("Columns", readable_int(df.shape[1]))
        with c3: st.metric("Numeric Cols", readable_int(df.select_dtypes(include=[np.number]).shape[1]))
        st.subheader("Top 10 rows"); st.dataframe(df.head(10))
        st.subheader("Columns & Dtypes"); st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}))

if current=="Missing Values":
    st.header("3) Missing Values")
    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a file first.")
    else:
        miss = compute_missing(df)
        if miss.empty: st.success("No missing values detected 🎉")
        else:
            st.info("Per-column missing values"); st.dataframe(miss)

if current=="Feature Engineering":
    st.header("4) Feature Engineering (lags, rolling stats)")
    df = st.session_state["df"]
    if df is None:
        st.warning("Please upload a file first.")
    else:
        cols = list(df.columns)
        tgt_default = detect_first_present(cols, TARGET_CANDIDATES) or (cols[0] if cols else None)
        tgt = st.selectbox("Target (e.g., Close)", cols, index=cols.index(tgt_default) if tgt_default in cols else 0)
        lags = st.multiselect("Lags", [1,2,3,5,10,15,20,30], default=[1,2,3,5,10])
        windows = st.multiselect("Rolling Windows", [5,10,20,30,50,100], default=[5,10,20])
        if st.button("🧪 Build Features"):
            fe = make_lag_features(df, tgt, lags=tuple(lags), windows=tuple(windows))
            extras = [c for c in df.columns if c not in fe.columns and df[c].dtype.kind in "if"]
            fe = pd.concat([fe, df[extras]], axis=1)
            fe = fe.dropna().reset_index(drop=True)
            st.session_state["features_df"] = fe
            st.success(f"Feature matrix ready: {fe.shape[0]} rows × {fe.shape[1]} columns")
            st.dataframe(fe.head(10))
            to_csv_download(fe, "features.csv")

if current=="EDA":
    st.header("5) EDA")
    df = st.session_state["df"]
    if df is None: st.warning("Please upload a file first.")
    else:
        cols = list(df.columns)
        date_col = detect_first_present(cols, DATE_CANDIDATES)
        price_col = detect_first_present(cols, TARGET_CANDIDATES) or (cols[0] if cols else None)
        c1,c2 = st.columns(2)
        with c1: tcol = st.selectbox("Time axis", [None]+cols, index=(cols.index(date_col)+1) if date_col in cols else 0)
        with c2: pcol = st.selectbox("Price column", cols, index=(cols.index(price_col) if price_col in cols else 0))
        if pcol:
            if tcol:
                fig,ax = plt.subplots(); ax.plot(df[tcol], df[pcol]); ax.set_xlabel(tcol); ax.set_ylabel(pcol); ax.set_title("Price over Time"); st.pyplot(fig)
            else:
                fig2,ax2 = plt.subplots(); ax2.plot(df[pcol].values); ax2.set_xlabel("Index"); ax2.set_ylabel(pcol); ax2.set_title("Price (indexed)"); st.pyplot(fig2)
        vol = detect_first_present(cols, ["Volume","volume","Vol","vol"])
        if vol:
            fig3,ax3 = plt.subplots(); ax3.plot(df[vol]); ax3.set_title("Volume"); st.pyplot(fig3)

if current=="Train & Evaluate":
    st.header("6) Train & Evaluate")
    fe = st.session_state["features_df"]
    if fe is None or "return_1" not in fe.columns:
        st.warning("Please build features first (previous step).")
    else:
        cols = list(fe.columns)
        tgt = st.selectbox("Target", cols, index=0, key="tgt_model")
        features = [c for c in cols if c != tgt]
        model_name = st.selectbox("Model", ["LinearRegression","RandomForestRegressor","GradientBoostingRegressor"], index=1)
        test_size = st.slider("Holdout Test size (%)", 10, 40, 20, step=5) / 100.0
        use_tscv = st.checkbox("Use TimeSeriesSplit CV (averages shown)", value=False)
        splits = st.slider("TimeSeriesSplit folds", 3, 10, 5) if use_tscv else None
        n_estimators = st.slider("n_estimators (RF/GB)", 50, 500, 200, step=50)
        max_depth = st.slider("max_depth (RF only, 0=None)", 0, 30, 10, step=2); max_depth = None if max_depth==0 else max_depth
        lr_gb = st.slider("learning_rate (GB)", 0.01, 0.5, 0.1, step=0.01)

        def make_estimator():
            if model_name=="LinearRegression": return LinearRegression()
            elif model_name=="RandomForestRegressor": return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            else: return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr_gb, random_state=42)

        if st.button("🎯 Train"):
            X = fe[features]; y = fe[tgt].values
            split_idx = int(len(X)*(1-test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False)), ("model", make_estimator())])
            if use_tscv:
                tscv = TimeSeriesSplit(n_splits=splits); maes=[]; mses=[]; r2s=[]
                for tr,te in tscv.split(X):
                    pipe.fit(X.iloc[tr], y[tr]); pr = pipe.predict(X.iloc[te])
                    maes.append(mean_absolute_error(y[te], pr)); mses.append(mean_squared_error(y[te], pr)); r2s.append(r2_score(y[te], pr))
                st.info(f"CV (TS split={splits}) — MAE={np.mean(maes):.4f}, MSE={np.mean(mses):.4f}, R²={np.mean(r2s):.4f}")
            pipe.fit(X_train, y_train); pred = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, pred); mse = mean_squared_error(y_test, pred); r2 = r2_score(y_test, pred)
            c1,c2,c3 = st.columns(3); 
            with c1: st.metric("MAE", f"{mae:.4f}")
            with c2: st.metric("MSE", f"{mse:.4f}")
            with c3: st.metric("R²", f"{r2:.4f}")
            fig,ax = plt.subplots(); ax.plot(y_test, label="Actual"); ax.plot(pred, label="Predicted", linestyle="--"); ax.legend(); ax.set_title("Test segment"); st.pyplot(fig)
            fig2,ax2 = plt.subplots(); ax2.plot(y_test - pred); ax2.set_title("Residuals"); st.pyplot(fig2)
            if model_name in ["RandomForestRegressor","GradientBoostingRegressor"]:
                try:
                    importances = pipe.named_steps["model"].feature_importances_
                    imp_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False).head(25)
                    st.subheader("Top Feature Importances"); st.dataframe(imp_df)
                except Exception: pass
            st.session_state["last_predictions"] = pd.DataFrame({"y_true": y_test, "y_pred": pred})

if current=="Forecast":
    st.header("7) Forecast next N steps (recursive)")
    fe = st.session_state.get("features_df")
    if fe is None: st.warning("Please build features first.")
    else:
        cols = list(fe.columns)
        tgt = st.selectbox("Target", cols, index=0, key="tgt_fore")
        features = [c for c in cols if c != tgt]
        model_name = st.selectbox("Model", ["LinearRegression","RandomForestRegressor","GradientBoostingRegressor"], index=1, key="model_fore")
        horizon = st.slider("Forecast horizon (steps)", 1, 60, 10, step=1)
        n_estimators = st.slider("n_estimators (RF/GB)", 50, 500, 200, step=50, key="nest_fore")
        max_depth = st.slider("max_depth (RF only, 0=None)", 0, 30, 10, step=2, key="md_fore"); max_depth = None if max_depth==0 else max_depth
        lr_gb = st.slider("learning_rate (GB)", 0.01, 0.5, 0.1, step=0.01, key="lrf_fore")

        def make_estimator():
            if model_name=="LinearRegression": return LinearRegression()
            elif model_name=="RandomForestRegressor": return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            else: return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr_gb, random_state=42)

        if st.button("📈 Forecast"):
            X = fe[features]; y = fe[tgt].values
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False)), ("model", make_estimator())]).fit(X, y)
            future = []
            current_row = X.iloc[-1].copy()
            for step in range(horizon):
                yhat = pipe.predict(pd.DataFrame([current_row]))[0]
                future.append(yhat)
                for c in X.columns:
                    if c.startswith(f"{tgt}_lag"):
                        try:
                            lag_n = int(c.split("lag")[1])
                            if lag_n == 1: current_row[c] = yhat
                            else:
                                prev = f"{tgt}_lag{lag_n-1}"
                                if prev in current_row.index: current_row[c] = current_row[prev]
                        except: pass
            st.success(f"Generated {len(future)} forecasted steps")
            # import numpy as np
            # fig,ax = plt.subplots(); ax.plot(np.arange(len(y))[-100:], y[-100:], label="History (last 100)"); ax.plot(np.arange(len(y)-1, len(y)+len(future)-1), [y[-1]]+future, linestyle="--", label="Forecast"); ax.legend(); ax.set_title("Recursive Forecast"); st.pyplot(fig)
            # st.session_state["future_forecast"] = pd.DataFrame({"forecast": future})
            # --- Plot history + forecast (fixed lengths) ---
            import numpy as np

            hist_tail = min(100, len(y))  # show up to last 100 points
            fig, ax = plt.subplots()

            # history
            ax.plot(np.arange(len(y))[-hist_tail:], y[-hist_tail:], label="History (last 100)")

            # forecast: N+1 y points (last actual + N forecasts) and N+1 x points
            future = [float(v) for v in future]  # ensure numeric
            x_fore = np.arange(len(y)-1, len(y)-1 + len(future) + 1)
            y_fore = np.concatenate([[y[-1]], np.asarray(future)])

            ax.plot(x_fore, y_fore, linestyle="--", label="Forecast")
            ax.legend()
            ax.set_title("Recursive Forecast")
            st.pyplot(fig)


if current=="Export":
    st.header("8) Export")
    preds = st.session_state.get("last_predictions")
    fut = st.session_state.get("future_forecast")
    if preds is not None:
        st.subheader("Test Predictions"); st.dataframe(preds.head(25)); to_csv_download(preds, "test_predictions.csv")
    if fut is not None:
        st.subheader("Future Forecast"); st.dataframe(fut.head(25)); to_csv_download(fut, "future_forecast.csv")
    if preds is None and fut is None:
        st.info("Run Train/Forecast steps to export CSVs.")
