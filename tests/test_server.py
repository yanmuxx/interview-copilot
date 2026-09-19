"""服务接口测试：只测非破坏性路径，不启动面试流水线（CI 无模型/音频）。"""
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_index_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "面试 Copilot" in r.text


def test_overlay_page():
    r = client.get("/overlay")
    assert r.status_code == 200
    assert "悬浮" in r.text


def test_get_config():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"has_key", "base_url", "model"}


def test_csrf_middleware_rejects_cross_origin():
    r = client.post("/api/config", data={"api_key": "hacked"}, headers={"Origin": "http://evil.com"})
    assert r.status_code == 403


def test_same_origin_passes_middleware():
    r = client.get("/api/config", headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_parse_resume_without_file_rejected():
    r = client.post("/api/resume/parse", data={"jd": ""})
    assert r.status_code == 400


def test_interview_state():
    r = client.get("/api/interview/state")
    assert r.status_code == 200
    assert r.json() == {"running": False}


def test_history_endpoint():
    r = client.get("/api/interview/history")
    assert r.status_code == 200
    assert isinstance(r.json()["events"], list)
