"""AIRSYNC SQLite Persistence Store.
Engine stays DB-agnostic by publishing updates via hook to this store.
"""
import sqlite3
import json
import threading
import os

class Store:
    def __init__(self, path=None):
        if path is None:
            path = os.getenv("DB_PATH", "airsync.db")
        self.path = path
        self.c = sqlite3.connect(path, check_same_thread=False)
        self.l = threading.Lock()
        self.c.executescript("""
        create table if not exists audit_logs(id integer primary key, ts text, event text, score real, status text, payload text);
        create table if not exists assignments(id integer primary key, audit_id integer, task text, aircraft text, crew text, score real);
        create table if not exists events(id integer primary key, ts text, detail text);
        create table if not exists recommendations(id integer primary key, audit_id integer, task text, reason text, confidence real);
        """)

    def save(self, rec, assignments):
        with self.l:
            aid = self.c.execute(
                "insert into audit_logs(ts,event,score,status,payload) values(?,?,?,?,?)",
                (rec["ts"], rec["event"], rec["score"], rec["status"], json.dumps(rec))
            ).lastrowid
            self.c.executemany(
                "insert into assignments(audit_id,task,aircraft,crew,score) values(?,?,?,?,?)",
                [(aid, k, v["aircraft"], v["crew"], v["score"]) for k, v in assignments.items()]
            )
            self.c.executemany(
                "insert into recommendations(audit_id,task,reason,confidence) values(?,?,?,?)",
                [(aid, k, json.dumps(v["reason"]), v["confidence"]) for k, v in assignments.items()]
            )
            if rec.get("steps"):
                self.c.execute("insert into events(ts,detail) values(?,?)", (rec["ts"], rec["event"]))
            self.c.commit()

    def history(self, n=50):
        with self.l:
            rows = self.c.execute("select payload from audit_logs order by id desc limit ?", (n,)).fetchall()
            return [json.loads(r[0]) for r in rows]
