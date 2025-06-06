import pandas as pd
import logging
from src.data_management.config_loader import ConfigLoader
from src.data_management.data_loader import DataLoader
from src.data_management.schema_enforcer import DataFrameSchemaEnforcer

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class DataManager:
    """
    Manages data loading and configuration for data analysis.

    This class integrates the functionality of loading configuration files and datasets,
    providing a unified interface for data management tasks.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing DataManager.")
        self.config_loader = ConfigLoader()
        self.data_loader = DataLoader()

    def load_config(self, file_path: str) -> dict:
        """
        Load configuration from a specified file path.

        Parameters
        ----------
        file_path : str
            Path to the configuration file (YAML or JSON).

        Returns
        -------
        dict
            Loaded configuration data.
        """
        self.logger.info(f"Loading configuration from: {file_path}")
        try:
            self.config_loader.get_configs(file_path)
            config = self.config_loader.config
            self.logger.info(f"Configuration loaded successfully from: {file_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {file_path}: {e}")
            raise

    def load_data(
        self,
        ref_path: str,
        cmp_path: str,
        ref_sheet_name: str = None,
        cmp_sheet_name: str = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Load datasets from specified file paths.

        Parameters
        ----------
        ref_path : str
            Path to the reference dataset file (CSV, Parquet or Excel).
        cmp_path : str
            Path to the comparison dataset file (CSV, Parquet or Excel).
        ref_sheet_name : str, optional
            Name of the sheet to read for the reference dataset (for Excel files).
        cmp_sheet_name : str, optional
            Name of the sheet to read for the comparison dataset (for Excel files).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, list[str]]
            Reference dataset, comparison dataset, and list of common columns.

        Raises
        ------
        ValueError
            If file extensions are unsupported or if there are no common columns.
        """
        self.logger.info(
            f"Loading data: ref_path={ref_path}, cmp_path={cmp_path}, "
            f"ref_sheet_name={ref_sheet_name}, cmp_sheet_name={cmp_sheet_name}"
        )
        try:
            df_ref, df_cmp, common_columns = self.data_loader.get_data(
                ref_path, cmp_path, ref_sheet_name, cmp_sheet_name
            )
            self.logger.info(
                f"Data loaded successfully: "
                f"ref_rows={len(df_ref)}, ref_cols={len(df_ref.columns)}; "
                f"cmp_rows={len(df_cmp)}, cmp_cols={len(df_cmp.columns)}; "
                f"common_columns={common_columns}"
            )
            return df_ref, df_cmp, common_columns
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def enforce_schema(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Enforce a schema on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to enforce the schema on.
        schema : dict
            Schema definition to enforce.

        Returns
        -------
        pd.DataFrame
            DataFrame with enforced schema.

        Raises
        ------
        ValueError
            If the schema enforcement fails.
        """
        self.logger.info("Enforcing schema on DataFrame.")
        try:
            enforcer = DataFrameSchemaEnforcer(schema)
            df_enforced = enforcer.enforce(df)
            self.logger.info("Schema enforcement successful.")
            return df_enforced
        except Exception as e:
            self.logger.error(f"Schema enforcement failed: {e}")
            raise

    def pre_process_data(
        self,
        df: pd.DataFrame,
        schema: dict
    ) -> pd.DataFrame:
        """
        Pre-process a DataFrame by enforcing a schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to pre-process.
        schema : dict
            Schema definition to enforce.

        Returns
        -------
        pd.DataFrame
            Pre-processed DataFrame with enforced schema.

        Raises
        ------
        ValueError
            If the pre-processing fails.
        """
        self.logger.info("Pre-processing DataFrame.")
        try:
            df_preprocessed = self.enforce_schema(df, schema)
            self.logger.info("Pre-processing completed successfully.")
            return df_preprocessed
        except Exception as e:
            self.logger.error(f"Pre-processing failed: {e}")
            raise

    def get_ref_and_cmp_data(
        self,
        ref_path: str,
        cmp_path: str,
        ref_data_schema_path: str,
        cmp_data_schema_path: str,
        ref_sheet_name: str = None,
        cmp_sheet_name: str = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict, dict]:
        """
        Load reference and comparison datasets, enforce schemas, and return them with common columns.

        Parameters
        ----------
        ref_path : str
            Path to the reference dataset file.
        cmp_path : str
            Path to the comparison dataset file.
        ref_data_schema_path : str
            Path to the reference data schema file (JSON/YAML).
        cmp_data_schema_path : str
            Path to the comparison data schema file (JSON/YAML).
        ref_sheet_name : str, optional
            Name of the sheet for the reference dataset (if applicable).
        cmp_sheet_name : str, optional
            Name of the sheet for the comparison dataset (if applicable).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, list[str], dict, dict]
            Tuple containing:
            - Reference DataFrame (schema enforced)
            - Comparison DataFrame (schema enforced)
            - List of common columns
            - Reference schema dict
            - Comparison schema dict
        """
        self.logger.info("Starting full data retrieval and schema enforcement process.")
        try:
            # Load raw data
            df_ref, df_cmp, common_columns = self.load_data(
                ref_path, cmp_path, ref_sheet_name, cmp_sheet_name
            )

            # Load schemas
            self.logger.info(f"Loading reference schema from: {ref_data_schema_path}")
            ref_schema = self.load_config(ref_data_schema_path)
            self.logger.info(f"Loading comparison schema from: {cmp_data_schema_path}")
            cmp_schema = self.load_config(cmp_data_schema_path)

            # Enforce schemas
            self.logger.info("Enforcing schema on reference DataFrame.")
            df_ref_enforced = self.pre_process_data(df_ref, ref_schema)
            self.logger.info("Enforcing schema on comparison DataFrame.")
            df_cmp_enforced = self.pre_process_data(df_cmp, cmp_schema)

            self.logger.info(
                f"Finished processing reference and comparison data. "
                f"Returning data with {len(common_columns)} common columns."
            )
            return df_ref_enforced, df_cmp_enforced, common_columns, ref_schema, cmp_schema

        except Exception as e:
            self.logger.error(f"get_ref_and_cmp_data failed: {e}")
            raise