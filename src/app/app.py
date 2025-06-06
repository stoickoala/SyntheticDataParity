# src/app/app.py

import gradio as gr

# Now adjust these to point into your `src` packages:
from src.data_management.data_manager import DataManager
from src.report.report import Report

from src.report.tab.general_stats_tab import GeneralStatsTab
from src.report.tab.multivariate_tab import MultivariateTab
from src.report.tab.global_summary_tab import GlobalSummaryTab

if __name__ == "__main__":
    # (1) Two shared State objects
    report_state      = gr.State(None)
    common_cols_state = gr.State(None)

    # (2) Instantiate each Tab subclass, passing the States:
    general_tab  = GeneralStatsTab(report_state, common_cols_state)
    multivar_tab = MultivariateTab(report_state, common_cols_state)
    global_tab   = GlobalSummaryTab(report_state, common_cols_state)

    # (3) Build the UI under one Tabs container:
    with gr.Blocks() as demo:
        with gr.Tabs():
            general_tab.build_ui()
            multivar_tab.build_ui()
            global_tab.build_ui()

        # (4) Register callbacks after the Tabs block:
        general_tab.register_callbacks()
        multivar_tab.register_callbacks()
        global_tab.register_callbacks()

    # (5) Launch the app:
    demo.launch(share=True)