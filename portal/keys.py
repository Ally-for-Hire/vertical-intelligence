"""API-key store (sqlite) and CLI.  Budgets are in tokens (input + output).

DB lives at <repo>/portal/data/portal.db; repo = PORTAL_REPO or the parent of portal/.
Usage:  python portal/keys.py new --name alice --budget 500000 --daily 100000
"""
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys(id INTEGER PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT UNIQUE NOT NULL,
  budget_tokens INTEGER NOT NULL, daily_tokens INTEGER, created TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS usage(id INTEGER PRIMARY KEY, key_id INTEGER NOT NULL, ts TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('ask','propose')), input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL, note TEXT);
CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY, key_id INTEGER NOT NULL, ts TEXT NOT NULL,
  path TEXT NOT NULL, title TEXT NOT NULL);
"""


def repo_root() -> Path:
    env = os.environ.get("PORTAL_REPO")
    return Path(env).resolve() if env else Path(__file__).resolve().parent.parent


def db_path() -> Path:
    return repo_root() / "portal" / "data" / "portal.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)  # created on demand
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _commit(sql: str, args: tuple = ()) -> None:
    con = connect()
    try:
        with con:
            con.execute(sql, args)
    finally:
        con.close()


def _one(sql: str, args: tuple = ()):
    con = connect()
    try:
        return con.execute(sql, args).fetchone()
    finally:
        con.close()


def new_key(name: str, budget: int, daily: int | None = None) -> str:
    """Create a key; the plaintext is returned once and never stored."""
    key = "vi_" + secrets.token_urlsafe(24)[:32]
    _commit("INSERT INTO keys(name,key_hash,budget_tokens,daily_tokens,created) VALUES(?,?,?,?,?)",
            (name, _hash(key), budget, daily, _now()))
    return key


def verify(key: str | None) -> sqlite3.Row | None:
    """Key row, or None if unknown or revoked."""
    if not key:
        return None
    row = _one("SELECT * FROM keys WHERE key_hash=?", (_hash(key),))
    return None if row is None or row["revoked"] else row


def remaining(key_id: int) -> dict:
    """Total and today's (UTC) budget state. daily_cap None = no daily cap."""
    k = _one("SELECT * FROM keys WHERE id=?", (key_id,))
    used = _one("SELECT COALESCE(SUM(input_tokens+output_tokens),0) FROM usage WHERE key_id=?", (key_id,))[0]
    today = _one("SELECT COALESCE(SUM(input_tokens+output_tokens),0) FROM usage "
                 "WHERE key_id=? AND substr(ts,1,10)=?", (key_id, _today()))[0]
    total_rem = max(0, k["budget_tokens"] - used)
    daily_rem = None if k["daily_tokens"] is None else max(0, k["daily_tokens"] - today)
    available = total_rem if daily_rem is None else min(total_rem, daily_rem)
    return {"budget_total": k["budget_tokens"], "used": used, "remaining": total_rem,
            "daily_cap": k["daily_tokens"], "used_today": today, "daily_remaining": daily_rem,
            "available": available}


def global_used_today() -> int:
    return _one("SELECT COALESCE(SUM(input_tokens+output_tokens),0) FROM usage "
                "WHERE substr(ts,1,10)=?", (_today(),))[0]


def record_usage(key_id: int, kind: str, input_tokens: int, output_tokens: int, note: str = "") -> None:
    _commit("INSERT INTO usage(key_id,ts,kind,input_tokens,output_tokens,note) VALUES(?,?,?,?,?,?)",
            (key_id, _now(), kind, int(input_tokens), int(output_tokens), note[:200]))


def record_proposal(key_id: int, path: str, title: str) -> None:
    _commit("INSERT INTO proposals(key_id,ts,path,title) VALUES(?,?,?,?)", (key_id, _now(), path, title))


def _find(name_or_id) -> sqlite3.Row | None:
    if str(name_or_id).isdigit():
        r = _one("SELECT * FROM keys WHERE id=?", (int(name_or_id),))
        if r:
            return r
    return _one("SELECT * FROM keys WHERE name=? ORDER BY id DESC", (str(name_or_id),))


def revoke(name_or_id) -> bool:
    r = _find(name_or_id)
    if not r:
        return False
    _commit("UPDATE keys SET revoked=1 WHERE id=?", (r["id"],))
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="keys", description="Manage portal API keys")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--name", required=True)
    n.add_argument("--budget", type=int, required=True, help="total tokens (input+output)")
    n.add_argument("--daily", type=int, default=None, help="tokens per UTC day (default: no daily cap)")
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("who", help="id or name")
    u = sub.add_parser("usage")
    u.add_argument("who", nargs="?")
    a = ap.parse_args(argv)

    if a.cmd == "new":
        key = new_key(a.name, a.budget, a.daily)
        print(f"Key for {a.name} (shown once, store it now):\n{key}")
    elif a.cmd == "list":
        con = connect()
        rows = con.execute("SELECT * FROM keys ORDER BY id").fetchall()
        con.close()
        for k in rows:
            s = remaining(k["id"])
            print(f"{k['id']:>3} {k['name']:<20} budget={k['budget_tokens']} used={s['used']} "
                  f"daily={k['daily_tokens']} {'REVOKED' if k['revoked'] else 'active'} created={k['created']}")
    elif a.cmd == "revoke":
        ok = revoke(a.who)
        print("revoked" if ok else "no such key")
        return 0 if ok else 1
    elif a.cmd == "usage":
        k = _find(a.who) if a.who else None
        if a.who and not k:
            print("no such key")
            return 1
        con = connect()
        q = "SELECT u.*, k.name FROM usage u JOIN keys k ON k.id=u.key_id"
        rows = con.execute(q + (" WHERE k.id=?" if k else "") + " ORDER BY u.id DESC LIMIT 50",
                           (k["id"],) if k else ()).fetchall()
        con.close()
        for x in rows:
            print(f"{x['ts']} {x['name']:<16} {x['kind']:<8} in={x['input_tokens']} out={x['output_tokens']} {x['note'] or ''}")
        if not rows:
            print("no usage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
