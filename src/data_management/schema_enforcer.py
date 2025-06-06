import pandas as pd
import numpy as np
import re
import datetime
import logging
from decimal import Decimal
from dateutil.parser import parse as parse_date

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class DataFrameSchemaEnforcer:
    """
    Enforces column types on a DataFrame based on a user-provided schema.

    schema: dict mapping column names to target types, where target types can be:
      - Python types: int, float, str, bool, Decimal, datetime.datetime
      - Strings: 'int', 'float', 'string', 'boolean', 'datetime', 'decimal', 'category'
    """
    _NUM_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)([kmbtKMBT])?\s*$")
    _SUFFIX_FACTORS = {
        'k': 1e3,
        'm': 1e6,
        'b': 1e9,
        't': 1e12,
    }

    def __init__(self, schema: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.schema = schema
        self.handlers = {
            int: self._handle_int,
            'int': self._handle_int,
            float: self._handle_float,
            'float': self._handle_float,
            str: self._handle_str,
            'string': self._handle_str,
            'text': self._handle_str,
            bool: self._handle_bool,
            'boolean': self._handle_bool,
            datetime.datetime: self._handle_datetime,
            'datetime': self._handle_datetime,
            Decimal: self._handle_decimal,
            'decimal': self._handle_decimal,
            'category': self._handle_category,
        }
        self.logger.info(f"Initialized DataFrameSchemaEnforcer with schema: {self.schema}")

    def enforce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame with columns cast according to the schema.
        """
        self.logger.info("Starting schema enforcement on DataFrame.")
        result = df.copy()
        for col, tgt in self.schema.items():
            if col not in result.columns:
                self.logger.warning(f"Column '{col}' not found in DataFrame; skipping.")
                continue

            handler = self.handlers.get(tgt)
            if handler is None:
                self.logger.warning(f"No handler for target type '{tgt}' on column '{col}'; skipping.")
                continue

            self.logger.info(f"Enforcing type '{tgt}' on column '{col}'.")
            try:
                result[col] = handler(result[col])
                self.logger.info(f"Column '{col}' successfully cast to '{tgt}'.")
            except Exception as e:
                self.logger.error(f"Failed to cast column '{col}' to '{tgt}': {e}")
                raise

        self.logger.info("Schema enforcement complete.")
        return result

    def _handle_numeric(self, series: pd.Series) -> pd.Series:
        """Parse numeric strings with commas and suffixes to floats."""
        self.logger.info("Parsing numeric values.")
        def parse(val):
            if pd.isna(val):
                return np.nan
            s = str(val).replace(',', '').strip()
            m = self._NUM_RE.match(s)
            if m:
                num = float(m.group(1))
                suf = m.group(2)
                if suf:
                    num *= self._SUFFIX_FACTORS[suf.lower()]
                return num
            try:
                return float(s)
            except Exception:
                return np.nan

        parsed_series = series.map(parse)
        self.logger.info("Numeric parsing complete.")
        return parsed_series

    def _handle_int(self, series: pd.Series) -> pd.Series:
        self.logger.info("Casting series to integer ('Int64').")
        floats = self._handle_numeric(series)
        try:
            int_series = floats.round(0).astype('Int64')
            self.logger.info("Integer casting complete.")
            return int_series
        except Exception as e:
            self.logger.error(f"Error casting to integer: {e}")
            raise

    def _handle_float(self, series: pd.Series) -> pd.Series:
        self.logger.info("Casting series to float.")
        try:
            float_series = self._handle_numeric(series).astype(float)
            self.logger.info("Float casting complete.")
            return float_series
        except Exception as e:
            self.logger.error(f"Error casting to float: {e}")
            raise

    def _handle_str(self, series: pd.Series) -> pd.Series:
        self.logger.info("Casting series to string.")
        try:
            str_series = series.astype(str)
            self.logger.info("String casting complete.")
            return str_series
        except Exception as e:
            self.logger.error(f"Error casting to string: {e}")
            raise

    def _handle_bool(self, series: pd.Series) -> pd.Series:
        self.logger.info("Casting series to boolean.")
        def parse(val):
            if pd.isna(val):
                return pd.NA
            s = str(val).strip().lower()
            if s in ('true','1','yes','y','t'):
                return True
            if s in ('false','0','no','n','f'):
                return False
            return pd.NA

        try:
            bool_series = series.map(parse).astype('boolean')
            self.logger.info("Boolean casting complete.")
            return bool_series
        except Exception as e:
            self.logger.error(f"Error casting to boolean: {e}")
            raise

    def _handle_datetime(self, series: pd.Series) -> pd.Series:
        self.logger.info("Casting series to datetime.")
        try:
            dt_series = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            self.logger.info("Datetime casting complete.")
            return dt_series
        except Exception as e:
            self.logger.error(f"Error casting to datetime: {e}")
            raise

    def _handle_decimal(self, series: pd.Series) -> pd.Series:
        self.logger.info("Casting series to Decimal.")
        def parse(val):
            if pd.isna(val):
                return None
            s = str(val).replace(',', '').strip()
            try:
                return Decimal(s)
            except Exception:
                return None

        try:
            dec_series = series.map(parse)
            self.logger.info("Decimal casting complete.")
            return dec_series
        except Exception as e:
            self.logger.error(f"Error casting to Decimal: {e}")
            raise

    def _handle_category(self, series: pd.Series) -> pd.Series:
        self.logger.info("Casting series to category.")
        try:
            cat_series = series.astype('category')
            self.logger.info("Category casting complete.")
            return cat_series
        except Exception as e:
            self.logger.error(f"Error casting to category: {e}")
            raise