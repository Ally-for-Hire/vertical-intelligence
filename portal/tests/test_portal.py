"""Offline tests: temp git repo, mocked model. Run: python -m pytest portal/tests -q"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # flat modules in portal/

from fastapi.testclient import TestClient  # noqa: E402

PHRASE = "zebrafish gradient quokka"
FM = ("---\ntitle: Test idea\nstatus: proposed\nproposed_by: someone\ndate: 2026-10-30\n"
      "section: S1\n---\n\n## Question\nWhy?\n")


def git(repo, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "program").mkdir(parents=True)
    (repo / "contrib" / "queue").mkdir(parents=True)
    (repo / "README.md").write_text("# Readme\n\nHello world.\n")
    (repo / "program" / "a.md").write_text(f"# Alpha\n\nIntro.\n\n## Planted\n\n{PHRASE} is here.\n")
    (repo / "contrib" / "queue" / "README.md").write_text("queue readme\n")
    (repo / "portal").mkdir()
    (repo / "portal" / "keys.py").write_text("secret = 1\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    monkeypatch.setenv("PORTAL_REPO", str(repo))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")
    monkeypatch.setenv("PORTAL_RATE_PER_MIN", "100")
    import app as portal
    import keys

    def fake_call_model(system, user, stream=False, max_tokens=1200):
        usage = {"input_tokens": 700, "output_tokens": 300}
        if stream:
            return iter(["Canned ", "answer [program/a.md].", {"usage": usage}])
        return ("---\ntitle: X\n---\n", usage)

    monkeypatch.setattr(portal, "call_model", fake_call_model)
    with TestClient(portal.app) as client:
        yield client, repo, keys


def H(k):
    return {"X-Portal-Key": k}


def test_unknown_key_401(env):
    c, _, _ = env
    assert c.get("/api/me").status_code == 401
    assert c.get("/api/me", headers=H("vi_nope")).status_code == 401


def test_status_no_auth(env):
    c, _, _ = env
    j = c.get("/api/status").json()
    assert j["files"] == 2 and j["head"]


def test_new_key_me_and_revoke(env):
    c, _, keys = env
    k = keys.new_key("alice", 10_000, 5_000)
    assert k.startswith("vi_") and len(k) == 35
    j = c.get("/api/me", headers=H(k)).json()
    assert j["name"] == "alice" and j["remaining"] == 10_000 and j["daily_remaining"] == 5_000
    assert keys.revoke("alice")
    assert c.get("/api/me", headers=H(k)).status_code == 401


def test_budget_exhaustion_402(env):
    c, _, keys = env
    k = keys.new_key("bob", 2500)
    r = c.post("/api/ask", headers=H(k), json={"question": PHRASE})
    assert r.status_code == 200 and "Canned " in r.text and "answer [program" in r.text
    assert c.get("/api/me", headers=H(k)).json()["used"] == 1000
    r = c.post("/api/ask", headers=H(k), json={"question": PHRASE})  # 1500 left < 2000
    assert r.status_code == 402 and r.json()["detail"]["remaining"] == 1500


def test_search_finds_phrase(env):
    c, _, keys = env
    k = keys.new_key("carol", 100_000)
    res = c.get("/api/search", params={"q": "quokka zebrafish"}, headers=H(k)).json()["results"]
    assert res and res[0]["path"] == "program/a.md" and res[0]["heading"] == "Planted"


def test_submit_and_queue(env):
    c, repo, keys = env
    k = keys.new_key("dave", 100_000)
    r = c.post("/api/propose/submit", headers=H(k), json={"title": "Test idea!", "markdown": FM})
    assert r.status_code == 200, r.text
    j = r.json()
    f = repo / j["path"]
    assert j["path"].startswith("contrib/queue/") and j["path"].endswith("-test-idea.md")
    assert "proposed_by: dave" in f.read_text() and "status: proposed" in f.read_text()
    assert git(repo, "rev-parse", "HEAD") == j["commit"]
    assert git(repo, "log", "-1", "--format=%an|%ae|%s") == "Portal dave|portal@localhost|contrib: proposal 'Test idea!' by dave"
    assert git(repo, "status", "--porcelain", "--", "contrib") == ""
    # second submit does not overwrite
    j2 = c.post("/api/propose/submit", headers=H(k), json={"title": "Test idea!", "markdown": FM}).json()
    assert j2["path"].endswith("-test-idea-2.md")
    q = c.get("/api/queue", headers=H(k)).json()["queue"]
    assert len(q) == 2 and q[0]["status"] == "proposed" and q[0]["proposed_by"] == "dave"
    # missing frontmatter rejected
    assert c.post("/api/propose/submit", headers=H(k), json={"title": "t", "markdown": "no fm"}).status_code == 422


def test_draft_records_usage(env):
    c, _, keys = env
    k = keys.new_key("erin", 100_000)
    j = c.post("/api/propose/draft", headers=H(k), json={"idea": "try x", "section": "S1", "success": "y"}).json()
    assert "draft" in j and c.get("/api/me", headers=H(k)).json()["used"] == 1000


def test_path_traversal_refused(env):
    c, _, keys = env
    k = keys.new_key("frank", 100_000)
    for p in ["../../etc/passwd", "..%2f..%2fetc/passwd", "portal/keys.py", "/etc/passwd", ".git/config"]:
        assert c.get("/api/file", params={"path": p}, headers=H(k)).status_code in (403, 404)
    assert c.get("/api/file", params={"path": "program/a.md"}, headers=H(k)).status_code == 200


def test_read_file_refuses_portal(env):
    import index
    _, repo, _ = env
    with pytest.raises(ValueError):
        index.read_file(repo, "portal/keys.py")
    with pytest.raises(ValueError):
        index.read_file(repo, "../outside.md")


def test_index_refresh_on_head_change(env):
    _, repo, _ = env
    import index
    ix = index.Index(repo)
    assert ix.refresh_if_changed() is True and ix.refresh_if_changed() is False
    (repo / "README.md").write_text("# New\n\nfresh\n")
    git(repo, "commit", "-qam", "edit")
    assert ix.refresh_if_changed() is True


def test_startup_requires_api_key(tmp_path, monkeypatch):
    import app as portal
    monkeypatch.setenv("PORTAL_REPO", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        portal.load_settings()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("PORTAL_MODEL", "other")
    with pytest.raises(RuntimeError, match="not allowed"):
        portal.load_settings()
