"""
Airflow Hook: OpenAQ v3 API connection hook.
"""

import sys

sys.path.insert(0, "/opt/airflow")

from airflow.hooks.base import BaseHook

from ingestion.openaq_client import OpenAQClient


class OpenAQHook(BaseHook):
    """Airflow hook for OpenAQ API v3."""

    conn_name_attr = "openaq_conn_id"
    default_conn_name = "openaq_default"
    conn_type = "http"
    hook_name = "OpenAQ v3"

    def __init__(self, openaq_conn_id: str = default_conn_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openaq_conn_id = openaq_conn_id
        self._client = None

    def get_client(self) -> OpenAQClient:
        """Get an OpenAQ client with configured API key."""
        if self._client is None:
            conn = self.get_connection(self.openaq_conn_id)
            api_key = conn.password or ""
            self._client = OpenAQClient(api_key=api_key)
        return self._client
