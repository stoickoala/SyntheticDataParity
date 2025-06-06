from abc import ABC, abstractmethod
import gradio as gr

class Tab(ABC):
    """
    Abstract base for a UI “tab.” Each subclass must implement build_ui() and register_callbacks().
    """
    def __init__(self, report_state, common_cols_state):
        self.report_state = report_state
        self.common_cols_state = common_cols_state

    @abstractmethod
    def build_ui(self):
        pass

    @abstractmethod
    def register_callbacks(self):
        pass