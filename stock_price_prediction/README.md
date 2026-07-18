
# 📈 Stock Price Prediction Dashboard

Upload → Preview → Missing → Feature Engineering → EDA → Train → Forecast → Export.

## Run
mkdir stock_dashboard && cd stock_dashboard
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
# Copy app.py & requirements.txt here
pip install -r requirements.txt
streamlit run app.py

# For Windows

mkdir stock_dashboard
cd stock_dashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py