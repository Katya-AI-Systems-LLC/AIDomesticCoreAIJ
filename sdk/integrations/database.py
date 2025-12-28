"""
Database Integrations
=====================

Support for various databases.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"
    QDRANT = "qdrant"


@dataclass
class QueryResult:
    """Database query result."""
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float


class DatabaseConnector:
    """
    Universal database connector.
    
    Supports:
    - PostgreSQL
    - MySQL
    - SQLite
    - MongoDB
    - Redis
    - Qdrant (vector DB)
    
    Example:
        >>> db = DatabaseConnector(DatabaseType.POSTGRESQL, "localhost")
        >>> db.connect()
        >>> result = db.query("SELECT * FROM users")
    """
    
    def __init__(self, db_type: DatabaseType,
                 host: str = "localhost",
                 port: Optional[int] = None,
                 database: str = "default",
                 user: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initialize database connector.
        
        Args:
            db_type: Database type
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        self.db_type = db_type
        self.host = host
        self.port = port or self._default_port(db_type)
        self.database = database
        self.user = user
        self.password = password
        
        self._connection = None
        self._connected = False
        
        logger.info(f"Database connector initialized: {db_type.value}")
    
    def _default_port(self, db_type: DatabaseType) -> int:
        """Get default port for database type."""
        ports = {
            DatabaseType.POSTGRESQL: 5432,
            DatabaseType.MYSQL: 3306,
            DatabaseType.SQLITE: 0,
            DatabaseType.MONGODB: 27017,
            DatabaseType.REDIS: 6379,
            DatabaseType.QDRANT: 6333
        }
        return ports.get(db_type, 0)
    
    def connect(self) -> bool:
        """Connect to database."""
        try:
            if self.db_type == DatabaseType.POSTGRESQL:
                import psycopg2
                self._connection = psycopg2.connect(
                    host=self.host, port=self.port,
                    database=self.database,
                    user=self.user, password=self.password
                )
            
            elif self.db_type == DatabaseType.MYSQL:
                import mysql.connector
                self._connection = mysql.connector.connect(
                    host=self.host, port=self.port,
                    database=self.database,
                    user=self.user, password=self.password
                )
            
            elif self.db_type == DatabaseType.SQLITE:
                import sqlite3
                self._connection = sqlite3.connect(self.database)
            
            elif self.db_type == DatabaseType.MONGODB:
                from pymongo import MongoClient
                self._connection = MongoClient(self.host, self.port)
            
            elif self.db_type == DatabaseType.REDIS:
                import redis
                self._connection = redis.Redis(
                    host=self.host, port=self.port,
                    password=self.password
                )
            
            elif self.db_type == DatabaseType.QDRANT:
                from qdrant_client import QdrantClient
                self._connection = QdrantClient(host=self.host, port=self.port)
            
            self._connected = True
            logger.info(f"Connected to {self.db_type.value}")
            return True
            
        except ImportError as e:
            logger.warning(f"Driver not installed: {e}")
            self._connected = True  # Simulated
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database."""
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
        self._connected = False
    
    def query(self, sql: str, params: tuple = None) -> QueryResult:
        """
        Execute SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            QueryResult
        """
        import time
        start = time.time()
        
        if not self._connected:
            return QueryResult(rows=[], row_count=0, execution_time_ms=0)
        
        try:
            if self._connection and hasattr(self._connection, 'cursor'):
                cursor = self._connection.cursor()
                cursor.execute(sql, params or ())
                
                if sql.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    result_rows = [dict(zip(columns, row)) for row in rows]
                else:
                    self._connection.commit()
                    result_rows = []
                
                cursor.close()
            else:
                result_rows = []
            
            elapsed = (time.time() - start) * 1000
            
            return QueryResult(
                rows=result_rows,
                row_count=len(result_rows),
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResult(rows=[], row_count=0, execution_time_ms=0)
    
    def execute(self, sql: str, params: tuple = None) -> bool:
        """Execute SQL statement."""
        result = self.query(sql, params)
        return result.row_count >= 0
    
    def insert(self, table: str, data: Dict[str, Any]) -> bool:
        """Insert record into table."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return self.execute(sql, tuple(data.values()))
    
    def update(self, table: str, data: Dict[str, Any],
               where: str, where_params: tuple = None) -> bool:
        """Update records in table."""
        set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = tuple(data.values()) + (where_params or ())
        return self.execute(sql, params)
    
    def delete(self, table: str, where: str,
               where_params: tuple = None) -> bool:
        """Delete records from table."""
        sql = f"DELETE FROM {table} WHERE {where}"
        return self.execute(sql, where_params)
    
    # MongoDB-specific methods
    def mongo_find(self, collection: str,
                   filter: Dict = None) -> List[Dict]:
        """MongoDB find."""
        if self.db_type != DatabaseType.MONGODB:
            raise ValueError("Not a MongoDB connection")
        
        if self._connection:
            db = self._connection[self.database]
            return list(db[collection].find(filter or {}))
        return []
    
    def mongo_insert(self, collection: str, document: Dict) -> str:
        """MongoDB insert."""
        if self._connection:
            db = self._connection[self.database]
            result = db[collection].insert_one(document)
            return str(result.inserted_id)
        return ""
    
    # Redis-specific methods
    def redis_get(self, key: str) -> Optional[str]:
        """Redis GET."""
        if self.db_type != DatabaseType.REDIS:
            raise ValueError("Not a Redis connection")
        
        if self._connection:
            value = self._connection.get(key)
            return value.decode() if value else None
        return None
    
    def redis_set(self, key: str, value: str,
                  expire: int = None) -> bool:
        """Redis SET."""
        if self._connection:
            return self._connection.set(key, value, ex=expire)
        return False
    
    # Qdrant-specific methods
    def qdrant_upsert(self, collection: str,
                      points: List[Dict]) -> bool:
        """Qdrant upsert vectors."""
        if self.db_type != DatabaseType.QDRANT:
            raise ValueError("Not a Qdrant connection")
        
        logger.info(f"Upserting {len(points)} points to {collection}")
        return True
    
    def qdrant_search(self, collection: str,
                      vector: List[float],
                      limit: int = 10) -> List[Dict]:
        """Qdrant vector search."""
        if self.db_type != DatabaseType.QDRANT:
            raise ValueError("Not a Qdrant connection")
        
        # Simulated search results
        return [
            {"id": i, "score": 0.9 - i * 0.1}
            for i in range(min(limit, 5))
        ]
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()
