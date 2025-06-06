import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.report.report import Report
from src.report.tab.abstract_tab import Tab

import logging

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class GlobalSummaryTab(Tab):
    """
    “Global Summary” tab (Step 5):
      - Button to “Compute Global Summary”
      - Dataframe: one row per common column, listing all metrics
      - Bar‐chart: top-5 most divergent columns by primary_divergence
    """
    def __init__(self, report_state, common_cols_state):
        super().__init__(report_state, common_cols_state)

    def build_ui(self):
        with gr.TabItem("Global Summary"):
            self.compute_summary_btn = gr.Button("Compute Global Summary")

            # Expanded set of headers to include all metrics + primary_divergence
            self.summary_table = gr.Dataframe(
                headers=[
                    "column",
                    "dtype",

                    # Missing-rate (real, syn, Δ)
                    "missing_rate_real",
                    "missing_rate_syn",
                    "missing_rate_diff",

                    # Numeric: KS & Wasserstein
                    "ks_similarity",
                    "wasserstein_raw",
                    "wasserstein_norm",

                    # Numeric: mean/median/std
                    "mean_real",
                    "mean_syn",
                    "mean_diff",
                    "mean_pct_of_range",
                    "median_real",
                    "median_syn",
                    "median_diff",
                    "median_pct_of_range",
                    "std_real",
                    "std_syn",
                    "std_diff",
                    "std_pct_of_std",

                    # Numeric: range coverage
                    "range_coverage",

                    # Categorical: TVD & coverage
                    "tvd_similarity",
                    "category_coverage",

                    # Text: token-JS, bigram Jaccard/JS
                    "text_tok_js",
                    "text_bigram_jaccard",
                    "text_bigram_js",

                    # Text: TF–IDF cosine, vocab Jaccard, OOV rate
                    "text_tfidf_cosine",
                    "text_vocab_jaccard",
                    "text_oov_rate",

                    # Text: topic cosine
                    "text_topic_cosine",

                    # Text: length (real, syn, Δ)
                    "len_real",
                    "len_syn",
                    "len_diff",

                    # Text: embedding metrics
                    "text_emb_cosine",
                    "text_emb_mmd",

                    # Global multivariate (same value for every row)
                    "correlation_similarity",
                    "contingency_similarity",
                    "distinguishability_auc",

                    # The chosen “primary divergence” for ranking
                    "primary_divergence"
                ],
                row_count="dynamic",
                interactive=False,
                label="Per-Column Metrics & Divergence"
            )

            # Bar chart for top-5 columns by primary_divergence
            self.divergence_bar = gr.Plot(visible=False)

    def register_callbacks(self):
        # Enable the “Compute” button only once report is loaded
        def enable_button(report):
            logger.info("GlobalSummaryTab: enable_button called.")
            return gr.update(interactive=report is not None)

        self.report_state.change(
            enable_button,
            inputs=[self.report_state],
            outputs=[self.compute_summary_btn]
        )

        # When “Compute Global Summary” is clicked:
        def compute_global_summary(report):
            logger.info("GlobalSummaryTab: compute_global_summary called.")
            if report is None:
                logger.info("  → No report: clearing table & hiding bar chart.")
                return ([], None, gr.update(visible=False))

            rows = []
            # 1) Gather univariate metrics per column
            for col in report.common_columns:
                logger.info(f"  → Computing univariate for column '{col}'.")
                col_info = report.compute_column(col)
                dtype = col_info.pop("dtype")

                # Initialize all possible keys to NaN
                base_row = {
                    "column": col,
                    "dtype": dtype,

                    # Missing-rate
                    "missing_rate_real":  np.nan,
                    "missing_rate_syn":   np.nan,
                    "missing_rate_diff":  np.nan,

                    # Numeric
                    "ks_similarity":      np.nan,
                    "wasserstein_raw":    np.nan,
                    "wasserstein_norm":   np.nan,
                    "mean_real":          np.nan,
                    "mean_syn":           np.nan,
                    "mean_diff":          np.nan,
                    "mean_pct_of_range":  np.nan,
                    "median_real":        np.nan,
                    "median_syn":         np.nan,
                    "median_diff":        np.nan,
                    "median_pct_of_range": np.nan,
                    "std_real":           np.nan,
                    "std_syn":            np.nan,
                    "std_diff":           np.nan,
                    "std_pct_of_std":     np.nan,
                    "range_coverage":     np.nan,

                    # Categorical
                    "tvd_similarity":     np.nan,
                    "category_coverage":  np.nan,

                    # Text
                    "text_tok_js":           np.nan,
                    "text_bigram_jaccard":   np.nan,
                    "text_bigram_js":        np.nan,
                    "text_tfidf_cosine":     np.nan,
                    "text_vocab_jaccard":    np.nan,
                    "text_oov_rate":         np.nan,
                    "text_topic_cosine":     np.nan,
                    "len_real":              np.nan,
                    "len_syn":               np.nan,
                    "len_diff":              np.nan,
                    "text_emb_cosine":       np.nan,
                    "text_emb_mmd":          np.nan,

                    # Global multivariate (filled below)
                    "correlation_similarity": np.nan,
                    "contingency_similarity": np.nan,
                    "distinguishability_auc": np.nan,

                    # To be computed per row
                    "primary_divergence":    np.nan
                }

                for key, val in col_info.items():
                    if key in base_row:
                        base_row[key] = val
                rows.append(base_row)

            df_metrics = pd.DataFrame(rows)
            logger.info("  → DataFrame of univariate metrics constructed.")

            # 2) Compute global multivariate metrics once
            corr_sim = report.compute_correlation_similarity()
            cont_sim = report.compute_contingency_similarity()
            auc_sim  = report.compute_distinguishability_auc()
            logger.info(f"  → Multivariate: corr={corr_sim}, cont={cont_sim}, auc={auc_sim}")

            # 3) Broadcast those values to every row
            df_metrics["correlation_similarity"]  = corr_sim
            df_metrics["contingency_similarity"]  = cont_sim
            df_metrics["distinguishability_auc"]  = auc_sim

            # 4) Define “primary_divergence” based on dtype
            def get_primary_divergence(r):
                dtype = r["dtype"]
                if dtype == "numeric":
                    wnorm = r["wasserstein_norm"]
                    if not pd.isna(wnorm):
                        return wnorm
                    ks = r["ks_similarity"]
                    return (1.0 - ks) if not pd.isna(ks) else np.nan

                elif dtype == "categorical":
                    tvd = r["tvd_similarity"]
                    return (1.0 - tvd) if not pd.isna(tvd) else np.nan

                elif dtype == "text":
                    tfidf = r["text_tfidf_cosine"]
                    if not pd.isna(tfidf):
                        return (1.0 - tfidf)
                    embc = r["text_emb_cosine"]
                    return (1.0 - embc) if not pd.isna(embc) else np.nan

                else:
                    return np.nan

            df_metrics["primary_divergence"] = df_metrics.apply(get_primary_divergence, axis=1)
            logger.info("  → Computed primary_divergence for each column.")

            # 5) Sort by primary_divergence (descending), then take top 5
            df_sorted = df_metrics.sort_values(
                by="primary_divergence", ascending=False, na_position="last"
            )
            top_n = min(5, len(df_sorted))
            df_top = df_sorted.iloc[:top_n, :]
            logger.info(f"  → Top {top_n} columns by primary_divergence: {df_top['column'].tolist()}")

            # 6) Build bar chart for top_n columns
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df_top["column"],
                    y=df_top["primary_divergence"],
                    text=df_top["primary_divergence"].round(4),
                    textposition="auto",
                    name="Primary Divergence"
                )
            )
            fig.update_layout(
                title=f"Top {top_n} Divergent Columns (by Primary Divergence)",
                xaxis_title="Column",
                yaxis_title="Divergence",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            logger.info("  → Bar chart for top divergences ready.")

            # 7) Return the full table and show the bar chart
            return (
                df_metrics.values.tolist(),
                fig,
                gr.update(visible=True)
            )

        self.compute_summary_btn.click(
            compute_global_summary,
            inputs=[self.report_state],
            outputs=[
                self.summary_table,   # table data (list of lists)
                self.divergence_bar,  # bar chart figure
                self.divergence_bar   # bar chart visibility toggle
            ]
        )