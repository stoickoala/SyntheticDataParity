# Synthetic Data Parity Dashboard

A Gradio-powered web app for quickly assessing how closely a synthetic tabular dataset matches its real counterpart. Compare distributions, summary statistics, principal-component structure, correlations, contingency tables and more — all in one interactive interface.

---

## 🔍 Features

1. **General Stats**  
    - Side-by-side view of reference vs. comparison tables (common columns only)  
    - Univariate metrics per column:  
        - Missing-rate breakdown  
            - Numeric: KS similarity, Wasserstein distance (raw & normalised), mean/median/std differences, range coverage  
            - Categorical: TVD similarity, category coverage  
            - Text: token-JS divergence, TF-IDF cosine, Jaccard, OOV rate, topic-distribution cosine, embedding metrics, sample snippets  

2. **Multivariate Analysis**  
    - PCA eigenvector cosines & weighted cosine  
    - 2D PCA projection plot  
    - Correlation-matrix similarity  
    - Contingency (joint-frequency) similarity for categorical pairs  
    - Distinguishability AUC via a small logistic classifier  

3. **Global Summary**  
    - Aggregated per-column table of every metric  
    - “Primary divergence” ranking and top-5 bar chart  

---

## 🚀 Installation

1. **Clone the repo**  
    ```bash
    git clone https://github.com/your-org/SyntheticDataParity.git
    cd SyntheticDataParity
    ```

2. **Create a virtual environment (optional, but recommended)

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # on Windows use `.venv\Scripts\activate`
    ```

3. **Install dependencies

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## 📁 Project Structure

SyntheticDataParity/
├── requirements.txt           ← project dependencies
├── data/                      ← example datasets & schemas
├── analysis/                  ← ad-hoc scripts (e.g. univariate.py)
├── notebooks/                 ← exploratory notebooks
└── src/
    ├── app/
    │   └── app.py             ← entry-point to launch Gradio UI
    ├── data_management/
    │   ├── data_loader.py
    │   ├── schema_enforcer.py
    │   └── data_manager.py    ← reads CSV/YAML + enforces schema
    └── report/
        ├── report.py          ← all metric-computing logic
        └── tab/
            ├── abstract_tab.py
            ├── general_stats_tab.py
            ├── multivariate_tab.py
            └── global_summary_tab.py

## ▶️ Usage

1. **Launch the app

    ```bash
    python -m src.app.app
    ```

2. **General Stats
    - Enter paths to your reference CSV and comparison CSV
    - Enter paths to their JSON/YAML schemas
    - Click Load Data
    - Select any common column to see detailed univariate comparisons

3. **Multivariate Analysis
    - Once data is loaded, choose the number of PCA components
    - Click Compute PCA & Multivariate Metrics

4. **Global Summary
    - Click Compute Global Summary to get a full metrics table and top-5 divergence chart

## 💡 Tips & Notes
- Schema files must map each column name to one of: `int`, `float`, `string/text`, `boolean`, `datetime`, `decimal`, or `category`.
- Text columns are auto-detected as “text” vs “categorical” based on unique-value ratio and token counts.
- All plots use Plotly for interactivity.
- No GPU or cloud infra needed — everything runs locally on CPU.
- For large tables, consider subsampling or limiting to ~10 000 rows to improve responsiveness.

## 🛠 Development

- Adding a new metric
    1. Implement computation in src/report/report.py
    2. Expose it in the appropriate tab under src/report/tab/
    3. Update the README & UI labels

- Customising UI
    - All Gradio components live under src/report/tab/*.py
    - Feel free to reorganise accordions, tweak colours or add tooltips