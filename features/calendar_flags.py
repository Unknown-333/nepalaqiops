"""
Calendar Flags — Nepal festival and seasonal flags for feature engineering.
"""

import logging
import os
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

FESTIVALS_CSV = os.path.join(os.path.dirname(__file__), "nepal_festivals.csv")


class CalendarFlags:
    """Computes Nepal-specific calendar and seasonal flags."""

    def __init__(self):
        self.festivals_df = self._load_festivals()

    def _load_festivals(self) -> pd.DataFrame:
        """Load the Nepal festivals calendar CSV."""
        try:
            df = pd.read_csv(FESTIVALS_CSV, parse_dates=["date"])
            logger.info(f"Loaded {len(df)} festival/holiday entries")
            return df
        except FileNotFoundError:
            logger.warning(f"Festival calendar not found at {FESTIVALS_CSV}")
            return pd.DataFrame(columns=["date", "festival_name", "flag_column_name"])

    def get_flags_for_date(self, dt: datetime) -> dict[str, bool]:
        """Get all calendar flags for a specific date."""
        date_only = pd.Timestamp(dt.date())

        # Check festival flags
        day_festivals = self.festivals_df[
            self.festivals_df["date"].dt.date == date_only.date()
        ]

        flags = {
            "is_tihar": False,
            "is_dashain": False,
            "is_indra_jatra": False,
            "is_public_holiday": False,
            "is_monsoon": self._is_monsoon(dt),
            "is_pre_monsoon": self._is_pre_monsoon(dt),
            "is_brick_kiln_season": self._is_brick_kiln_season(dt),
        }

        for _, row in day_festivals.iterrows():
            flag_col = row["flag_column_name"]
            if flag_col in flags:
                flags[flag_col] = True
            # All festival days are also public holidays
            flags["is_public_holiday"] = True

        return flags

    def add_flags_to_dataframe(self, df: pd.DataFrame, timestamp_col: str = "timestamp_utc") -> pd.DataFrame:
        """Add all calendar flag columns to a DataFrame."""
        df = df.copy()
        ts = pd.to_datetime(df[timestamp_col])

        # Seasonal flags (vectorized)
        month = ts.dt.month
        day = ts.dt.day

        df["is_monsoon"] = (month >= 6) & (month <= 9)
        df["is_pre_monsoon"] = (month >= 3) & (month <= 5)
        df["is_brick_kiln_season"] = (month >= 10) | (month <= 5)

        # Festival flags (date lookup)
        df["is_tihar"] = False
        df["is_dashain"] = False
        df["is_indra_jatra"] = False
        df["is_public_holiday"] = False

        if not self.festivals_df.empty:
            tihar_dates = set(
                self.festivals_df[
                    self.festivals_df["flag_column_name"] == "is_tihar"
                ]["date"].dt.date
            )
            dashain_dates = set(
                self.festivals_df[
                    self.festivals_df["flag_column_name"] == "is_dashain"
                ]["date"].dt.date
            )
            indra_dates = set(
                self.festivals_df[
                    self.festivals_df["flag_column_name"] == "is_indra_jatra"
                ]["date"].dt.date
            )
            holiday_dates = set(self.festivals_df["date"].dt.date)

            dates = ts.dt.date
            df["is_tihar"] = dates.isin(tihar_dates)
            df["is_dashain"] = dates.isin(dashain_dates)
            df["is_indra_jatra"] = dates.isin(indra_dates)
            df["is_public_holiday"] = dates.isin(holiday_dates)

        return df

    @staticmethod
    def _is_monsoon(dt: datetime) -> bool:
        """June 1 – September 30."""
        return 6 <= dt.month <= 9

    @staticmethod
    def _is_pre_monsoon(dt: datetime) -> bool:
        """March 1 – May 31 (worst pollution period)."""
        return 3 <= dt.month <= 5

    @staticmethod
    def _is_brick_kiln_season(dt: datetime) -> bool:
        """October – May (kilns operational)."""
        return dt.month >= 10 or dt.month <= 5
