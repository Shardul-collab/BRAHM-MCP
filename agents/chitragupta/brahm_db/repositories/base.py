"""
brahm_db/repositories/base.py
==============================
Base repository — thin SQLite wrapper used by all brahm_db repositories.
"""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from brahm_db.schema import get_connection

log = logging.getLogger("brahm_db.repository")


class BaseRepository:
    def __init__(self):
        self._conn: sqlite3.Connection = get_connection()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def fetch_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchall()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)
