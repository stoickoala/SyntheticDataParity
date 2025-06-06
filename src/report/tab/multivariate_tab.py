import gradio as gr
import numpy as np

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

class MultivariateTab(Tab):
    """
    “PCA Comparison” tab, now extended to include:
      1. PCA eigenvector‐cosine metrics
      2. Correlation Similarity (numeric × numeric)
      3. Contingency Similarity (categorical × categorical)
      4. Distinguishability AUC (classifier test)
    """
    def __init__(self, report_state, common_cols_state):
        super().__init__(report_state, common_cols_state)

    def build_ui(self):
        with gr.TabItem("Multivariate Analysis"):
            # Slider & button for PCA
            self.ncomp_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=2,
                label="Number of PCA Components to Compare"
            )
            self.compute_button = gr.Button("Compute PCA & Multivariate Metrics")

            # PCA outputs
            self.pca_metrics_output = gr.Markdown("", visible=False)
            self.pca_scatter_plot  = gr.Plot(visible=False)

            # NEW: Multivariate relationship outputs
            self.corr_sim_output        = gr.Markdown("", visible=False, label="Correlation Similarity")
            self.contingency_sim_output = gr.Markdown("", visible=False, label="Contingency Similarity")
            self.distinguish_auc_output = gr.Markdown("", visible=False, label="Distinguishability (ROC AUC)")

    def register_callbacks(self):
        # 1) Enable controls once report is ready
        def enable_controls(report):
            logger.info("MultivariateTab: enable_controls called.")
            if report is not None:
                logger.info("  → Enabling PCA & multivariate controls.")
                return gr.update(interactive=True), gr.update(interactive=True)
            else:
                logger.info("  → Disabling PCA & multivariate controls.")
                return gr.update(interactive=False), gr.update(interactive=False)

        self.report_state.change(
            enable_controls,
            inputs=[self.report_state],
            outputs=[self.ncomp_slider, self.compute_button]
        )

        # 2) When the button is clicked:
        def compute_multivariate(report, n_components):
            logger.info(f"MultivariateTab: compute_multivariate called (n_components={n_components}).")
            if report is None:
                logger.info("  → No report available: clearing all outputs.")
                return (
                    "",       # pca_metrics_output
                    None,     # pca_scatter_plot
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                )

            # --- (A) PCA Eigen‐Cosines (as before) ---
            pca_res = report.compute_pca_eigen_cosines(n_components=n_components)
            cosines = pca_res['cosine_similarities']
            ev_ref  = pca_res['explained_variance_ref']
            ev_cmp  = pca_res['explained_variance_cmp']
            weighted_cos = pca_res['weighted_cosine']

            md_pca = f"## PCA Eigenvector Cosines (first {len(cosines)} components)\n\n"
            for i, cos in enumerate(cosines, start=1):
                md_pca += (
                    f"- PC{i} cosine: **{cos:.4f}**  "
                    f"(exp_var_ref: {ev_ref[i-1]:.4f}, exp_var_cmp: {ev_cmp[i-1]:.4f})\n"
                )
            md_pca += f"\n**Weighted Cosine** (weights = ref exp_var_ratio): **{weighted_cos:.4f}**\n"
            logger.info("  → PCA Markdown prepared.")

            scatter_fig = report.plot_pca_scatter(n_components=max(n_components, 2))
            show_scatter = len(cosines) >= 2
            if show_scatter:
                logger.info("  → PCA scatter figure prepared.")
            else:
                logger.info("  → Not enough PCA components for scatter.")

            # --- (B) Correlation Similarity ---
            corr_sim = report.compute_correlation_similarity()
            if np.isnan(corr_sim):
                md_corr = "Correlation Similarity: **N/A** (need ≥ 2 numeric columns)"
                logger.info("  → Correlation similarity: N/A.")
            else:
                md_corr = f"**Correlation Similarity (numeric × numeric)**: **{corr_sim:.4f}**\n"
                logger.info(f"  → Correlation similarity = {corr_sim:.6f}")
            show_corr = not np.isnan(corr_sim)

            # --- (C) Contingency Similarity ---
            cont_sim = report.compute_contingency_similarity()
            if np.isnan(cont_sim):
                md_cont = "Contingency Similarity: **N/A** (need ≥ 2 categorical columns)"
                logger.info("  → Contingency similarity: N/A.")
            else:
                md_cont = f"**Contingency Similarity (cat × cat)**: **{cont_sim:.4f}**\n"
                logger.info(f"  → Contingency similarity = {cont_sim:.6f}")
            show_cont = not np.isnan(cont_sim)

            # --- (D) Distinguishability AUC ---
            auc = report.compute_distinguishability_auc()
            if np.isnan(auc):
                md_auc = "Distinguishability AUC: **N/A** (insufficient mix of numeric/categorical data)"
                logger.info("  → Distinguishability AUC: N/A.")
            else:
                md_auc = f"**Distinguishability (ROC AUC)**: **{auc:.4f}**  \n"
                md_auc += (
                    "- AUC near 0.5 ⇒ synthetic is hard to distinguish from real  \n"
                    "- AUC near 1.0 ⇒ classifier easily separates real vs. synthetic\n"
                )
                logger.info(f"  → Distinguishability AUC = {auc:.6f}")
            show_auc = not np.isnan(auc)

            return (
                # PCA outputs
                md_pca,                # 1: PCA Markdown
                scatter_fig,           # 2: PCA scatter
                gr.update(visible=True),
                gr.update(visible=show_scatter),
                # Correlation similarity
                md_corr,               # 5: correlation Markdown
                gr.update(visible=show_corr),
                # Contingency similarity
                md_cont,               # 7: contingency Markdown
                gr.update(visible=show_cont),
                # Distinguishability AUC
                md_auc,                # 9: AUC Markdown
                gr.update(visible=show_auc),
            )

        # Bind the button to 10 outputs (5 pairs of (value, visibility))
        self.compute_button.click(
            compute_multivariate,
            inputs=[self.report_state, self.ncomp_slider],
            outputs=[
                self.pca_metrics_output,      # 1
                self.pca_scatter_plot,        # 2
                self.pca_metrics_output,      # 3 (visibility)
                self.pca_scatter_plot,        # 4
                self.corr_sim_output,         # 5
                self.corr_sim_output,         # 6 (visibility)
                self.contingency_sim_output,  # 7
                self.contingency_sim_output,  # 8 (visibility)
                self.distinguish_auc_output,  # 9
                self.distinguish_auc_output   # 10 (visibility)
            ]
        )
