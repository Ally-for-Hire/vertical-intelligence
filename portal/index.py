"""Markdown indexing for the portal: file listing, chunking, a tiny BM25, HEAD polling.

Everything is read from the repo on disk (repo root = parent of portal/ unless PORTAL_REPO is set).
"""
from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from pathlib import Path

MAX_FILE_BYTES = 2 * 1024 * 1024
CHUNK_CHARS = 3200  # ~800 tokens
EXCLUDED_PARTS = {"portal", ".git", "wandb"}
# Glob patterns (relative to repo root) that make up the indexed corpus.
PATTERNS = [
    "README.md",
    "program/**/*.md",
    "AGENTS.md",
    "LOG.md",
    "HANDOFF.md",
    "paper/*.md",
    "experiments/**/*.md",
]


def _excluded(rel: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in rel.parts)


def list_markdown(repo: Path) -> list[str]:
    """Posix-style relative paths of indexable markdown files."""
    found: set[str] = set()
    for pat in PATTERNS:
        for p in repo.glob(pat):
            if not p.is_file():
                continue
            rel = p.relative_to(repo)
            if _excluded(rel):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.add(rel.as_posix())
    return sorted(found)


def chunk_markdown(path: str, text: str) -> list[tuple[str, str, str]]:
    """Split into (path, heading, text) chunks on heading/paragraph boundaries."""
    sections: list[tuple[str, list[str]]] = []
    heading, lines = "(top)", []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,6}\s", line):
            sections.append((heading, lines))
            heading, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    sections.append((heading, lines))

    chunks: list[tuple[str, str, str]] = []
    for head, body in sections:
        paras = [p for p in "\n".join(body).split("\n\n") if p.strip()]
        buf = ""
        for para in paras:
            while len(para) > CHUNK_CHARS:  # a single huge paragraph: hard split
                if buf:
                    chunks.append((path, head, buf))
                    buf = ""
                chunks.append((path, head, para[:CHUNK_CHARS]))
                para = para[CHUNK_CHARS:]
            if buf and len(buf) + len(para) + 2 > CHUNK_CHARS:
                chunks.append((path, head, buf))
                buf = ""
            buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            chunks.append((path, head, buf))
    return chunks


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", s.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs: list[dict[str, int]] = []
        self.lens: list[int] = []
        df: dict[str, int] = {}
        for d in docs:
            tf: dict[str, int] = {}
            for t in tokenize(d):
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)
            self.lens.append(sum(tf.values()))
            for t in tf:
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.avg = (sum(self.lens) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = set(tokenize(query))
        out = []
        for tf, ln in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out


def current_head(repo: Path) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class Index:
    def __init__(self, repo: Path) -> None:
        self.repo = Path(repo).resolve()
        self.lock = threading.Lock()
        self.head: str | None = None
        self.built_at = 0.0
        self.files: list[str] = []
        self.chunks: list[tuple[str, str, str]] = []
        self.bm25 = BM25([])
        self._built = False

    def build(self) -> None:
        files = list_markdown(self.repo)
        chunks: list[tuple[str, str, str]] = []
        for rel in files:
            try:
                text = (self.repo / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            chunks.extend(chunk_markdown(rel, text))
        # path and heading are searchable too
        bm = BM25([f"{p} {h} {t}" for p, h, t in chunks])
        head = current_head(self.repo)
        with self.lock:  # swap in one step so readers never see a half-built index
            self.files, self.chunks, self.bm25 = files, chunks, bm
            self.head, self.built_at, self._built = head, time.time(), True

    def refresh_if_changed(self) -> bool:
        """Rebuild if HEAD moved (or never built). Returns True if rebuilt."""
        if self._built and current_head(self.repo) == self.head:
            return False
        self.build()
        return True

    def search(self, query: str, k: int = 8) -> list[tuple[float, str, str, str]]:
        with self.lock:
            sc, chunks = self.bm25.scores(query), self.chunks
        top = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [(sc[i], *chunks[i]) for i in top if sc[i] > 0]

    def read_file(self, relpath: str) -> str:
        return read_file(self.repo, relpath)


def read_file(repo: Path, relpath: str) -> str:
    """Read a file inside the repo; refuse escapes and anything under portal/ (or .git, wandb)."""
    repo = Path(repo).resolve()
    if not relpath or "\x00" in relpath:
        raise ValueError("bad path")
    target = (repo / relpath).resolve()
    try:
        rel = target.relative_to(repo)
    except ValueError:
        raise ValueError("path outside repository")
    if _excluded(rel):
        raise ValueError("path not readable")
    if not target.is_file():
        raise FileNotFoundError(relpath)
    return target.read_text(encoding="utf-8", errors="replace")
