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

    def insert_returning_id(self, sql: str, params: tuple = ()) -> int:
        """
        INSERT and return the new row's id, from the cursor that performed it.

        Every repository used to recover the id by re-querying after the
        commit — `SELECT id FROM <table> [WHERE project_id=?] ORDER BY id
        DESC LIMIT 1` — in ten places across five repos, with `lastrowid`
        used nowhere. That reads whatever row is newest at SELECT time, not
        the row this call inserted, so a write that lands in between hands
        the caller someone else's id. Callers then link children to it
        (papers to projects, sections to documents), so a wrong id is a
        silently mis-parented record rather than an error.

        Honest caveat: this was NOT reproducible — 40 trials x 16 concurrent
        threads through PaperRepo.register_paper produced zero mis-attributed
        ids, because SQLite serialises writers and the commit->SELECT window
        is very small. It is a latent correctness bug, not an observed one.
        `lastrowid` is read off the same cursor inside the transaction, so it
        cannot have the window at all — strictly correct and no slower.
        """
        with self.transaction() as cursor:
            cursor.execute(sql, params)
            return cursor.lastrowid

    def fetch_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchall()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)
