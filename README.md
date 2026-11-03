# HeatX AI — Heat Exchanger Simulation Web App

A runnable educational Streamlit application with counter-current effectiveness–NTU heat exchanger physics, interactive sensitivity plots, a trained random-forest surrogate, operating point search, and illustrative fouling scenarios.

## Windows setup

1. Install Python 3.11+ and VS Code.
2. Extract this ZIP, then open the `HeatX_AI_App` folder in VS Code.
3. Open Terminal > New Terminal, and enter:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

4. Visit the URL displayed in the terminal (usually http://localhost:8501).

On macOS/Linux, use `python3 -m venv .venv`, `./.venv/bin/python -m pip install -r requirements.txt`, and `./.venv/bin/python -m streamlit run app.py`.

## Publish online

Create a GitHub repository, upload `app.py`, `engine.py`, and `requirements.txt` to its root, then connect that repository to https://share.streamlit.io and choose `app.py` as the entrypoint. Do not upload `.venv`.

## Scientific scope

The model assumes steady-state counter-current flow, water on both sides, constant heat capacity 4180 J/(kg K), constant overall UA, no phase change, and no heat loss. Pressure drop and pump energy are not included. Fouling is simulated as a user-selected decrease in UA. Machine learning is trained on synthetic model outputs and is **not** real-world validated. The optimization feature searches cold-side flow for a desired temperature; it does not minimize energy consumption.
