import sqlite3
import hashlib
from typing import List,Optional,Dict,Any
import logging

logger = logging.getLogger("db")

class DbManager:
    def __init__(self, db_path: str = "app.db"):
        self.db_path = db_path
        self.setup_db()

    def setup_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
            """)
            # Documents table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userid INTEGER NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                filehash TEXT NOT NULL,
                filesize INTEGER NOT NULL,
                filetype TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                vectorstore_id TEXT,
                FOREIGN KEY(userid) REFERENCES users(id),
                UNIQUE(userid, filehash)
            )
            """)
            # Search history
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userid INTEGER NOT NULL,
                query TEXT NOT NULL,
                results_count INTEGER DEFAULT 0,
                search_time REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(userid) REFERENCES users(id)
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(userid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(filehash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_user ON search_history(userid)")
            conn.commit()

    def get_conn(self):
        return sqlite3.connect(self.db_path)

    def create_user(self, username: str, email: str, hashedpwd: str) -> int:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
                    (username, email, hashedpwd),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError as e:
                raise ValueError(f"User exists: {e}")

    def get_user_by_name(self, username: str) -> Optional[Dict[str, Any]]:
        with self.get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, userid: int) -> Optional[Dict[str, Any]]:
        with self.get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (userid,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_user_activity(self, userid: int):

        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET is_active = TRUE WHERE id = ?", (userid,))
            conn.commit()

    def create_doc(
        self,
        userid: int,
        filename: str,
        filepath: str,
        filehash: str,
        filesize: int,
        filetype: str,
        vector_id: Optional[str] = None,
    ) -> int:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO documents (userid, filename, filepath, filehash, filesize, filetype, vectorstore_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (userid, filename, filepath, filehash, filesize, filetype, vector_id),
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError("Document hash already exists for this user")

    def get_doc_by_hash(self, userid: int, filehash: str) -> Optional[Dict[str, Any]]:
        with self.get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE userid = ? AND filehash = ?", (userid, filehash))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_docs(self, userid: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE userid = ? ORDER BY upload_date DESC LIMIT ? OFFSET ?",
                (userid, limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_doc(self, docid: int, userid: int) -> bool:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ? AND userid = ?", (docid, userid))
            conn.commit()
            return cursor.rowcount > 0

    def save_search(self, userid: int, query: str, results_count: int, search_time: float) -> int:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO search_history (userid, query, results_count, search_time) VALUES (?, ?, ?, ?)",
                (userid, query, results_count, search_time),
            )
            conn.commit()
            return cursor.lastrowid

    def get_search_history(self, userid: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM search_history WHERE userid = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (userid, limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self, userid: int) -> Dict[str, Any]:
        with self.get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) AS count, SUM(filesize) AS totalsize FROM documents WHERE userid = ?", (userid,))
            docstats = cursor.fetchone()
            total_docs = docstats[0] if docstats and docstats[0] else 0
            total_bytes = docstats[1] if docstats and docstats[1] else 0
            total_mb = (total_bytes or 0) / (1024 * 1024)

            cursor.execute("SELECT COUNT(*) FROM search_history WHERE userid = ?", (userid,))
            total_searches = cursor.fetchone()[0] or 0

            cursor.execute("SELECT MAX(created_at) FROM search_history WHERE userid = ?", (userid,))
            last_activity = cursor.fetchone()[0]

            return {
                "userid": userid,
                "total_documents": total_docs,
                "total_searches": total_searches,
                "storage_used_mb": round(total_mb, 2),
                "last_activity": last_activity,
            }

    @staticmethod
    def calc_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def close(self):
        pass



