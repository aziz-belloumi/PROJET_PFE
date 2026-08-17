#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Connection & Query Interface
=====================================
Provides connection pooling, transaction-safe query execution, and streaming
for extracting raw articles from MySQL database.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from src.config import Config


class DatabaseConnection:
    """
    Manages MySQL database connection pool via SQLAlchemy.
    Provides thread-safe connections, query fetching, and streaming.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.engine = None
        self._connect()

    def _connect(self):
        try:
            self.engine = create_engine(
                Config.SQLALCHEMY_DATABASE_URI,
                poolclass=QueuePool,
                pool_pre_ping=True,
                pool_recycle=Config.DB_POOL_RECYCLE,
                pool_size=Config.DB_POOL_SIZE,
                max_overflow=Config.DB_MAX_OVERFLOW,
                echo=False,
            )
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("MySQL connection established successfully.")
        except Exception as e:
            self.logger.error(f"MySQL connection error: {e}")
            raise

    def get_engine(self):
        return self.engine

    def is_connected(self) -> bool:
        """Verify that the database connection pool is active and reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_pool_status(self) -> dict:
        """Return current connection pool metrics."""
        if not self.engine or not hasattr(self.engine, "pool"):
            return {}
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }

    # ============================================================
    # Query Execution & Fetching
    # ============================================================

    def execute_write(self, sql_query: str, params: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute an INSERT / UPDATE / DELETE / DDL statement inside an atomic transaction.
        Automatically commits on success and rolls back on exception.
        """
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(sql_query), params or {})
                return result.rowcount
        except Exception as e:
            self.logger.error(f"SQL Write Error: {e}")
            raise

    def execute_query(self, sql_query: str, params: Optional[Dict[str, Any]] = None):
        """Backward-compatible alias for execute_write."""
        return self.execute_write(sql_query, params)

    def fetch_one(self, sql_query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Tuple]:
        """Execute a SELECT query and return the first row, or None."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query), params or {})
                return result.fetchone()
        except Exception as e:
            self.logger.error(f"SQL Read Error (fetch_one): {e}")
            raise

    def fetch_all(self, sql_query: str, params: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """Execute a SELECT query and return all matching rows."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query), params or {})
                return result.fetchall()
        except Exception as e:
            self.logger.error(f"SQL Read Error (fetch_all): {e}")
            raise

    def close(self):
        """Gracefully close and dispose the connection pool."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection pool disposed.")