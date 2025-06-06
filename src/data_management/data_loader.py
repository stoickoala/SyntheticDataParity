import pandas as pd
import os
import logging
from typing import List, Tuple, Union, Optional
from pathlib import Path

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class DataLoader:
    """
    Loads and manages reference and comparison datasets for analysis.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df_ref: Optional[pd.DataFrame] = None
        self.df_cmp: Optional[pd.DataFrame] = None
        self.common_columns: List[str] = []
        self.logger.info("Initialized DataLoader instance.")

    def load(
        self,
        ref_path: Union[str, Path],
        cmp_path: Union[str, Path],
        ref_sheet_name: Optional[str] = None,
        cmp_sheet_name: Optional[str] = None
    ) -> None:
        """
        Load datasets from specified file paths.

        Parameters
        ----------
        ref_path : str or Path
            Path to the reference dataset file (CSV, Parquet or Excel).
        cmp_path : str or Path
            Path to the comparison dataset file (CSV, Parquet or Excel).
        ref_sheet_name : str, optional
            Name of the sheet to read (for Excel files).
        cmp_sheet_name : str, optional
            Name of the sheet to read (for Excel files).

        Raises
        ------
        ValueError
            If file extensions are unsupported or if there are no common columns.
        """
        self.logger.info(f"Starting load: ref_path={ref_path}, cmp_path={cmp_path}")

        # Read reference file
        self.logger.info(f"Reading reference dataset from: {ref_path}")
        self.df_ref = self._read_file(ref_path, sheet_name=ref_sheet_name)
        if self.df_ref is None:
            self.logger.error(f"Failed to load reference dataset from {ref_path}")
            raise ValueError(f"Failed to load reference dataset from {ref_path}")
        if not isinstance(self.df_ref, pd.DataFrame):
            self.logger.error(f"Reference dataset at {ref_path} is not a valid DataFrame.")
            raise ValueError(f"Reference dataset at {ref_path} is not a valid DataFrame.")
        self.logger.info(f"Successfully loaded reference dataset ({len(self.df_ref)} rows, {len(self.df_ref.columns)} columns).")

        # Read comparison file
        self.logger.info(f"Reading comparison dataset from: {cmp_path}")
        self.df_cmp = self._read_file(cmp_path, sheet_name=cmp_sheet_name)
        if self.df_cmp is None:
            self.logger.error(f"Failed to load comparison dataset from {cmp_path}")
            raise ValueError(f"Failed to load comparison dataset from {cmp_path}")
        if not isinstance(self.df_cmp, pd.DataFrame):
            self.logger.error(f"Comparison dataset at {cmp_path} is not a valid DataFrame.")
            raise ValueError(f"Comparison dataset at {cmp_path} is not a valid DataFrame.")
        self.logger.info(f"Successfully loaded comparison dataset ({len(self.df_cmp)} rows, {len(self.df_cmp.columns)} columns).")

        # Determine common columns
        self.common_columns = list(set(self.df_ref.columns).intersection(self.df_cmp.columns))
        if not self.common_columns:
            self.logger.error("No common columns found between reference and comparison datasets.")
            raise ValueError("No common columns between reference and comparison datasets.")

        self.logger.info(f"Common columns ({len(self.common_columns)}): {self.common_columns}")

    def _read_file(
        self,
        path: Union[str, Path],
        sheet_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Read a DataFrame from a file path, inferring format from the extension.

        Parameters
        ----------
        path : str or Path
            Path to the dataset file.
        sheet_name : str, optional
            Name of the sheet to read (for Excel files).

        Returns
        -------
        pd.DataFrame or None
            Loaded DataFrame, or None if reading failed.
        """
        ext = Path(path).suffix.lower()
        self.logger.info(f"Attempting to read file '{path}' with extension '{ext}'")
        try:
            if ext == '.csv':
                df = pd.read_csv(path)
                self.logger.info(f"Read CSV file: {path}")
                return df
            elif ext in ('.parquet', '.parq', '.pq'):
                df = pd.read_parquet(path)
                self.logger.info(f"Read Parquet file: {path}")
                return df
            elif ext in ('.xlsx', '.xls', '.xlsm'):
                if sheet_name is None:
                    df = pd.read_excel(path)
                    self.logger.info(f"Read Excel file (default sheet): {path}")
                else:
                    df = pd.read_excel(path, sheet_name=sheet_name)
                    self.logger.info(f"Read Excel file (sheet='{sheet_name}'): {path}")
                return df
            else:
                error_msg = f"Unsupported file format: {ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            self.logger.error(f"Error reading file '{path}': {e}")
            return None

    def get_data(
        self,
        ref_path: Optional[Union[str, Path]] = None,
        cmp_path: Optional[Union[str, Path]] = None,
        ref_sheet_name: Optional[str] = None,
        cmp_sheet_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Return the loaded reference and comparison DataFrames (and common columns).

        Parameters
        ----------
        ref_path : str or Path, optional
            Path to the reference dataset file (if not already loaded).
        cmp_path : str or Path, optional
            Path to the comparison dataset file (if not already loaded).

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame, List[str])
            (reference DataFrame, comparison DataFrame, list of common columns)
        """
        if ref_path and cmp_path:
            self.logger.info("Paths provided to get_data; invoking load()")
            self.load(
                ref_path=ref_path,
                cmp_path=cmp_path,
                ref_sheet_name=ref_sheet_name,
                cmp_sheet_name=cmp_sheet_name
            )
        if self.df_ref is None:
            self.logger.error("Reference dataset is not loaded.")
            raise ValueError("Reference dataset is not loaded.")
        if self.df_cmp is None:
            self.logger.error("Comparison dataset is not loaded.")
            raise ValueError("Comparison dataset is not loaded.")
        if not self.common_columns:
            self.logger.error("No common columns identified; cannot proceed.")
            raise ValueError("No common columns between reference and comparison datasets.")

        self.logger.info("Returning loaded data and common columns.")
        return self.df_ref, self.df_cmp, self.common_columns
