import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_main():
    path = os.path.join(ROOT, "main.py")
    spec = importlib.util.spec_from_file_location("arc_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_offline_uses_arcade_not_http(tmp_path, monkeypatch):
    # Em OFFLINE a lista de jogos vem do Arcade local (get_environments), nunca do
    # HTTP. Diretorio vazio -> 0 jogos, sem crash e sem tocar requests.
    # _load_main() dispara load_dotenv(override=True); setar o env DEPOIS do import
    # pra o teste nao depender de um .env de dev no CWD (fetch le o env no call-time).
    m = _load_main()
    monkeypatch.setenv("OPERATION_MODE", "offline")
    monkeypatch.setenv("ENVIRONMENTS_DIR", str(tmp_path))

    def _boom(*a, **k):
        raise AssertionError("HTTP usado no modo offline")

    monkeypatch.setattr(m.requests, "Session", _boom)
    games = m.fetch_full_games("http://unused", {})
    assert games == []


def test_online_still_uses_http(monkeypatch):
    # Em ONLINE o helper usa o HTTP normalmente (aqui simulamos resposta 200).
    m = _load_main()
    monkeypatch.setenv("OPERATION_MODE", "online")

    class _Resp:
        status_code = 200

        def json(self):
            return [{"game_id": "vc33-abc"}, {"game_id": "ls20-def"}]

    class _Sess:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(m.requests, "Session", lambda *a, **k: _Sess())
    games = m.fetch_full_games("http://gateway:8001", {})
    assert games == ["vc33-abc", "ls20-def"]


def test_offline_games_filter(monkeypatch):
    m = _load_main()
    monkeypatch.setenv("OPERATION_MODE", "offline")
    monkeypatch.setenv("OFFLINE_GAMES", "vc33,ls20")

    class _Env:
        def __init__(self, gid): self.game_id = gid
    class _Arc:
        def __init__(self, *a, **k): pass
        def get_environments(self):
            return [_Env("sk48-d80"), _Env("vc33-543"), _Env("ls20-960"), _Env("tn36-ef4")]
    import arc_agi
    monkeypatch.setattr(arc_agi, "Arcade", _Arc, raising=False)
    monkeypatch.setattr(arc_agi, "OperationMode", type("OM", (), {"OFFLINE": 1}), raising=False)

    out = m.fetch_full_games("http://x", {})
    assert out == ["vc33-543", "ls20-960"]     # só os prefixos pedidos
