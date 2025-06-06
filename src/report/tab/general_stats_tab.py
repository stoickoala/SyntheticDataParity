import gradio as gr
import numpy as np

# Adjust to absolute imports under src:
from src.data_management.data_manager import DataManager
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

class GeneralStatsTab(Tab):
    """
    “General Stats” tab, reorganised into accordions for:
      • Missingness
      • Numeric Distributions
      • Categorical Frequencies
      • Text Metadata
    """
    def __init__(self, report_state, common_cols_state):
        super().__init__(report_state, common_cols_state)

    def build_ui(self):
        with gr.TabItem("General Stats"):
            # ---- Data‐Load Controls ----
            with gr.Row():
                self.ref_path_input = gr.Textbox(
                    label="Reference Dataset Path", placeholder="e.g. data/reference.csv"
                )
                self.cmp_path_input = gr.Textbox(
                    label="Comparison Dataset Path", placeholder="e.g. data/comparison.csv"
                )
            with gr.Row():
                self.ref_schema_input = gr.Textbox(
                    label="Reference Schema Path (JSON/YAML)", placeholder="e.g. data/ref_schema.json"
                )
                self.cmp_schema_input = gr.Textbox(
                    label="Comparison Schema Path (JSON/YAML)", placeholder="e.g. data/cmp_schema.json"
                )

            self.load_button = gr.Button("Load Data")
            self.load_status = gr.Markdown("Awaiting data load...")

            # ---- Side‐by‐Side Dataframes of Common Columns ----
            with gr.Row():
                self.ref_table = gr.Dataframe(
                    value=None,
                    label="Reference Dataset (Common Columns Only)",
                    interactive=False
                )
                self.cmp_table = gr.Dataframe(
                    value=None,
                    label="Comparison Dataset (Common Columns Only)",
                    interactive=False
                )

            # ---- Common‐Columns Dropdown ----
            self.column_dropdown = gr.Dropdown(
                label="Select Common Column for Univariate Comparison",
                choices=[],
                interactive=False
            )

            # ---- Accordions & Placeholders ----

            # 1) Missingness Accordion
            self.missing_accordion = gr.Accordion("Missingness", visible=False, open=True)
            with self.missing_accordion:
                self.missing_md = gr.Markdown("", visible=False)

            # 2) Numeric Distributions Accordion
            self.numeric_accordion = gr.Accordion("Numeric Distributions", visible=False, open=False)
            with self.numeric_accordion:
                self.numeric_md        = gr.Markdown("", visible=False)
                self.numeric_cdf_plot  = gr.Plot(visible=False)
                self.numeric_hist_plot = gr.Plot(visible=False)
                self.numeric_kde_plot  = gr.Plot(visible=False)

            # 3) Categorical Frequencies Accordion
            self.categorical_accordion = gr.Accordion("Categorical Frequencies", visible=False, open=False)
            with self.categorical_accordion:
                self.categorical_md       = gr.Markdown("", visible=False)
                self.categorical_bar_plot = gr.Plot(visible=False)

            # 4) Text Metadata Accordion
            self.text_accordion = gr.Accordion("Text Metadata", visible=False, open=False)
            with self.text_accordion:
                self.text_md            = gr.Markdown("", visible=False)
                self.text_token_plot    = gr.Plot(visible=False)
                self.text_len_plot      = gr.Plot(visible=False)
                self.text_samples_real  = gr.Textbox(label="Sample Real Text", visible=False, interactive=False)
                self.text_samples_synth = gr.Textbox(label="Sample Synthetic Text", visible=False, interactive=False)

    def register_callbacks(self):
        # 1. Load Data callback
        def load_data(ref_path, cmp_path, ref_schema_path, cmp_schema_path):
            try:
                ref_df, cmp_df, common_columns, ref_schema, cmp_schema = (
                    DataManager()
                    .get_ref_and_cmp_data(
                        ref_path, cmp_path, ref_schema_path, cmp_schema_path
                    )
                )
                report = Report(ref_df, cmp_df, common_columns, ref_schema, cmp_schema)

                ref_sub = ref_df.loc[:, common_columns]
                cmp_sub = cmp_df.loc[:, common_columns]

                return (
                    report,
                    common_columns,
                    ref_sub,
                    cmp_sub,
                    f"✅ Data loaded successfully. Found {len(common_columns)} common columns."
                )
            except Exception as e:
                err = f"❌ Error loading data: {e}"
                return (None, None, None, None, err)

        self.load_button.click(
            load_data,
            inputs=[
                self.ref_path_input,
                self.cmp_path_input,
                self.ref_schema_input,
                self.cmp_schema_input,
            ],
            outputs=[
                self.report_state,
                self.common_cols_state,
                self.ref_table,
                self.cmp_table,
                self.load_status,
            ]
        )

        # 2. Enable Dataframes & Dropdown after data loads
        def enable_components(report, common_columns):
            if report is not None and common_columns:
                return (
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(choices=common_columns, interactive=True),
                )
            else:
                return (
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(choices=[], interactive=False),
                )

        self.report_state.change(
            enable_components,
            inputs=[self.report_state, self.common_cols_state],
            outputs=[self.ref_table, self.cmp_table, self.column_dropdown]
        )

        # 3. Show per‐column stats & plots when dropdown changes
        def show_column_stats(report, col_name):
            # If no report or no column chosen: hide everything
            if report is None or not col_name:
                return (
                    # Missingness Accordion + content
                    gr.update(visible=False),
                    "", gr.update(visible=False),

                    # Numeric Accordion + content + plots
                    gr.update(visible=False),
                    "", gr.update(visible=False),
                    None, gr.update(visible=False),
                    None, gr.update(visible=False),
                    None, gr.update(visible=False),

                    # Categorical Accordion + content + plot
                    gr.update(visible=False),
                    "", gr.update(visible=False),
                    None, gr.update(visible=False),

                    # Text Accordion + content + plots + samples
                    gr.update(visible=False),
                    "", gr.update(visible=False),
                    None, gr.update(visible=False),
                    None, gr.update(visible=False),
                    "", gr.update(visible=False),
                    "", gr.update(visible=False),
                )

            # Compute univariate metrics
            ctype       = report.column_types.get(col_name)
            col_metrics = report.compute_column(col_name)

            #######  (A) MISSINGNESS #######
            mr_real = report.ref_df[col_name].isna().mean()
            mr_synth = report.cmp_df[col_name].isna().mean()
            mr_diff = abs(mr_real - mr_synth)
            missing_md = ""
            missing_md += "### Missingness (Real vs Synthetic)\n\n"
            missing_md += (
                f"- Real % missing = {mr_real*100:.1f}%  |  "
                f"Synth % missing = {mr_synth*100:.1f}%  |  "
                f"Δ = {mr_diff*100:+.1f} pts\n"
            )

            #######  (B) NUMERIC #######
            numeric_md   = ""
            cdf_fig      = None
            hist_fig     = None
            kde_fig      = None
            show_numeric = (ctype == "numeric")

            if show_numeric:
                # 1) KS similarity
                ks_sim = col_metrics.get("ks_similarity", np.nan)
                # 2) Wasserstein (raw + normalised)
                w_raw  = col_metrics.get("wasserstein_raw",   np.nan)
                w_norm = col_metrics.get("wasserstein_norm",  np.nan)
                # 3) Mean / Median / Std differences
                mean_r   = col_metrics.get("mean_real",    np.nan)
                mean_s   = col_metrics.get("mean_syn",     np.nan)
                mean_d   = col_metrics.get("mean_diff",    np.nan)
                mean_pct = col_metrics.get("mean_pct_of_range", np.nan)

                med_r    = col_metrics.get("median_real",  np.nan)
                med_s    = col_metrics.get("median_syn",   np.nan)
                med_d    = col_metrics.get("median_diff",  np.nan)
                med_pct  = col_metrics.get("median_pct_of_range", np.nan)

                std_r    = col_metrics.get("std_real",     np.nan)
                std_s    = col_metrics.get("std_syn",      np.nan)
                std_d    = col_metrics.get("std_diff",     np.nan)
                std_pct  = col_metrics.get("std_pct_of_std", np.nan)

                # 4) Range coverage
                range_cov = col_metrics.get("range_coverage", np.nan)

                numeric_md += "### Numeric Distributions (Real vs Synthetic)\n\n"
                if not np.isnan(ks_sim):
                    numeric_md += f"- KS Similarity (1 – D): **{ks_sim:.4f}**\n"
                if not np.isnan(w_raw):
                    if not np.isnan(w_norm):
                        numeric_md += (
                            f"- Wasserstein Distance: **{w_raw:.4f} units**  "
                            f"(normalised = {w_norm:.4f})\n"
                        )
                    else:
                        numeric_md += f"- Wasserstein Distance: **{w_raw:.4f} units**  (normalised = N/A)\n"

                if not np.isnan(mean_r) and not np.isnan(mean_s):
                    pct_txt = f" ({mean_pct*100:+.2f}% of real‐range)" if not np.isnan(mean_pct) else ""
                    numeric_md += (
                        f"- Mean → Real: **{mean_r:.4f}**, Syn: **{mean_s:.4f}**, "
                        f"Δ = **{mean_d:+.4f}**{pct_txt}\n"
                    )
                if not np.isnan(med_r) and not np.isnan(med_s):
                    pct_txt = f" ({med_pct*100:+.2f}% of real‐range)" if not np.isnan(med_pct) else ""
                    numeric_md += (
                        f"- Median → Real: **{med_r:.4f}**, Syn: **{med_s:.4f}**, "
                        f"Δ = **{med_d:+.4f}**{pct_txt}\n"
                    )
                if not np.isnan(std_r) and not np.isnan(std_s):
                    pct_txt = f" ({std_pct*100:+.1f}% of real‐std)" if not np.isnan(std_pct) else ""
                    numeric_md += (
                        f"- Std Dev → Real: **{std_r:.4f}**, Syn: **{std_s:.4f}**, "
                        f"Δ = **{std_d:+.4f}**{pct_txt}\n"
                    )

                if not np.isnan(range_cov):
                    numeric_md += f"- Range Coverage: **{range_cov:.4f}** (1.0 = perfect)\n"

                # Generate plots
                cdf_fig  = report.plot_column_cdf(col_name)
                hist_fig = report.plot_column_histogram(col_name)
                kde_fig  = report.plot_column_kde(col_name)

            #######  (C) CATEGORICAL #######
            cat_md   = ""
            cat_fig  = None
            show_cat = (ctype == "categorical")

            if show_cat:
                tvd_sim  = col_metrics.get("tvd_similarity",    np.nan)
                coverage = col_metrics.get("category_coverage", np.nan)

                cat_md += "### Categorical Frequencies (Real vs Synthetic)\n\n"
                if not np.isnan(tvd_sim):
                    cat_md += f"- TVD Similarity: **{tvd_sim:.4f}**\n"
                if not np.isnan(coverage):
                    cat_md += f"- Category Coverage: **{coverage:.4f}**\n"

                cat_fig = report.plot_category_bar(col_name)

            #######  (D) TEXT #######
            text_md    = ""
            tok_fig    = None
            len_fig    = None
            samples_r  = ""
            samples_s  = ""
            show_text  = (ctype == "text")

            if show_text:
                tok_js     = col_metrics.get("text_tok_js",        np.nan)
                bigram_j   = col_metrics.get("text_bigram_jaccard", np.nan)
                bigram_js  = col_metrics.get("text_bigram_js",      np.nan)
                tfidf_cos  = col_metrics.get("text_tfidf_cosine",   np.nan)
                vocab_j    = col_metrics.get("text_vocab_jaccard",  np.nan)
                oov_rate   = col_metrics.get("text_oov_rate",       np.nan)
                topic_cos  = col_metrics.get("text_topic_cosine",   np.nan)
                len_r      = col_metrics.get("len_real",            np.nan)
                len_s      = col_metrics.get("len_syn",             np.nan)
                len_d      = col_metrics.get("len_diff",            np.nan)
                emb_cos    = col_metrics.get("text_emb_cosine",     np.nan)
                emb_mmd    = col_metrics.get("text_emb_mmd",        np.nan)

                text_md += "### Text Metadata (Real vs Synthetic)\n\n"
                if not np.isnan(tok_js):
                    text_md += f"- Token‐JS Divergence (unigrams): **{tok_js:.4f}**\n"
                if not np.isnan(bigram_j):
                    text_md += f"- Bigram Jaccard: **{bigram_j:.4f}**\n"
                if not np.isnan(bigram_js):
                    text_md += f"- Bigram‐JS Divergence: **{bigram_js:.4f}**\n"
                if not np.isnan(tfidf_cos):
                    text_md += f"- TF–IDF Cosine Similarity: **{tfidf_cos:.4f}**\n"
                if not np.isnan(vocab_j):
                    text_md += f"- Vocabulary Jaccard (unigrams): **{vocab_j:.4f}**\n"
                if not np.isnan(oov_rate):
                    text_md += f"- OOV Rate (real→synth): **{oov_rate*100:.1f}%**\n"
                if not np.isnan(topic_cos):
                    text_md += f"- Topic‐Distribution Cosine: **{topic_cos:.4f}**\n"
                if not np.isnan(len_r) and not np.isnan(len_s):
                    text_md += (
                        f"- Average Length → Real: **{len_r:.1f}** tokens, "
                        f"Synthetic: **{len_s:.1f}**, Δ = **{len_d:+.1f} tokens**\n"
                    )
                if not np.isnan(emb_cos):
                    text_md += f"- Embedding Cosine: **{emb_cos:.4f}**\n"
                if not np.isnan(emb_mmd):
                    text_md += f"- Embedding MMD: **{emb_mmd:.4f}**\n"

                # Plots for text
                tok_fig = report.plot_text_top_tokens(col_name)
                len_fig = report.plot_text_length_hist(col_name)

                # Sample up to 3 real vs. synthetic texts
                real_texts = report.ref_df[col_name].dropna().astype(str)
                synth_texts = report.cmp_df[col_name].dropna().astype(str)
                if len(real_texts) > 0 and len(synth_texts) > 0:
                    samples_r = "\n\n".join(real_texts.sample(min(3, len(real_texts))).tolist())
                    samples_s = "\n\n".join(synth_texts.sample(min(3, len(synth_texts))).tolist())

            # Return all 22 outputs in the exact order declared in build_ui()
            return (
                # 1) Missingness Accordion visibility
                gr.update(visible=True),
                # 2) Missingness Markdown content
                missing_md,
                # 3) Missingness Markdown visibility
                gr.update(visible=True),

                # 4) Numeric Accordion visibility
                gr.update(visible=show_numeric),
                # 5) Numeric Markdown content
                numeric_md,
                # 6) Numeric Markdown visibility
                gr.update(visible=show_numeric),
                # 7) CDF plot (numeric)
                cdf_fig,
                gr.update(visible=show_numeric),
                # 8) Histogram plot (numeric)
                hist_fig,
                gr.update(visible=show_numeric),
                # 9) KDE plot (numeric)
                kde_fig,
                gr.update(visible=show_numeric),

                # 10) Categorical Accordion visibility
                gr.update(visible=show_cat),
                # 11) Categorical Markdown content
                cat_md,
                # 12) Categorical Markdown visibility
                gr.update(visible=show_cat),
                # 13) Categorical bar‐plot
                cat_fig,
                gr.update(visible=show_cat),

                # 14) Text Accordion visibility
                gr.update(visible=show_text),
                # 15) Text Markdown content
                text_md,
                # 16) Text Markdown visibility
                gr.update(visible=show_text),
                # 17) Top‐token bar (text)
                tok_fig,
                gr.update(visible=show_text),
                # 18) Document‐length histogram (text)
                len_fig,
                gr.update(visible=show_text),
                # 19) Sample Real Text
                samples_r,
                gr.update(visible=show_text and bool(samples_r)),
                # 20) Sample Synthetic Text
                samples_s,
                gr.update(visible=show_text and bool(samples_s)),
            )

        # Bind the dropdown to all 22 outputs (each paired with a visibility toggle)
        self.column_dropdown.change(
            show_column_stats,
            inputs=[self.report_state, self.column_dropdown],
            outputs=[
                # Missingness
                self.missing_accordion,
                self.missing_md,
                self.missing_md,

                # Numeric
                self.numeric_accordion,
                self.numeric_md,
                self.numeric_md,
                self.numeric_cdf_plot,
                self.numeric_cdf_plot,
                self.numeric_hist_plot,
                self.numeric_hist_plot,
                self.numeric_kde_plot,
                self.numeric_kde_plot,

                # Categorical
                self.categorical_accordion,
                self.categorical_md,
                self.categorical_md,
                self.categorical_bar_plot,
                self.categorical_bar_plot,

                # Text
                self.text_accordion,
                self.text_md,
                self.text_md,
                self.text_token_plot,
                self.text_token_plot,
                self.text_len_plot,
                self.text_len_plot,
                self.text_samples_real,
                self.text_samples_real,
                self.text_samples_synth,
                self.text_samples_synth,
            ]
        )
