from yaml import safe_load
from typing import Union, Optional, List
import pandas as pd
from pathlib import Path
from json import load as json_load

class ConfigLoader:
    """
    Handles loading configuration files for data analysis.
    Supports YAML and JSON formats.

    Reads configuration data from a specified file path and provides a dictionary-like interface to access configuration values.
    """

    def __init__(self):
        self.config: Optional[dict] = None
        self.file_path: Optional[Union[str, Path]] = None
        self.file_type: Optional[str] = None

    def load(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.

        Parameters
        ----------
        file_path : str or Path
            Path to the configuration file (YAML or JSON).

        Raises
        ------
        ValueError
            If the file format is unsupported or if the file cannot be read.
        """
        self.file_path = Path(file_path)
        ext = self.file_path.suffix.lower()

        if ext in ['.yaml', '.yml']:
            self.file_type = 'yaml'
            with open(self.file_path, 'r') as f:
                self.config = safe_load(f)
        elif ext in ['.json']:
            self.file_type = 'json'
            with open(self.file_path, 'r') as f:
                self.config = json_load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats are YAML and JSON.")
        if not isinstance(self.config, dict):
            raise ValueError("Configuration file must contain a valid dictionary.")
        if not self.config:
            raise ValueError("Configuration file is empty.")

    def get_configs(self, file_path: Optional[Union[str, Path]] = None) -> dict:
        """
        Get the loaded configuration data.

        Parameters
        ----------
        file_path : str or Path, optional
            If provided, will load the configuration from this file instead of the previously loaded one.

        Returns
        -------
        dict
            The loaded configuration data.

        Raises
        ------
        ValueError
            If no configuration has been loaded.
        """
        if file_path:
            self.load(file_path)

        if self.config is None:
            raise ValueError("No configuration has been loaded.")

        return self.config