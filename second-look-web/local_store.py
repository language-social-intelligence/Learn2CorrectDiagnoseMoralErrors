"""Private, durable application storage on this machine only."""
import json
import os
from pathlib import Path
import sqlite3
import time


class LocalStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL)")
        self.db.commit()
        # Requests interrupted by a process restart must never look successful.
        interrupted = self.db.execute("SELECT id,payload FROM jobs WHERE status IN ('queued','running')").fetchall()
        for key, payload in interrupted:
            job = json.loads(payload)
            job.update(status="error", error="The service restarted during this request. Please submit it again.", finished=time.time())
            self.put(key, job)

    def put(self, key, job):
        with self.db:
            self.db.execute(
                "INSERT INTO jobs (id,status,created,payload) VALUES (?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status,payload=excluded.payload",
                (key, job["status"], job["created"], json.dumps(job, ensure_ascii=False)),
            )

    def get(self, key):
        row = self.db.execute("SELECT payload FROM jobs WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.db.close()
