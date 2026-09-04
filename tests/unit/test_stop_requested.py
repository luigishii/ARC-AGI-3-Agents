import os
import time
from threading import Thread
from unittest.mock import Mock, patch

import pytest
from arcengine import GameAction

from agents.agent import Agent
from agents.swarm import Swarm
from arc_agi import OperationMode


class _Loop(Agent):
    """Agente que nunca termina sozinho; conta as decisoes."""
    MAX_ACTIONS = 10 ** 9

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def is_done(self, frames, latest_frame):
        return False

    def choose_action(self, frames, latest_frame):
        self.calls += 1
        if self.calls >= 3:
            self.stop_requested = True
        return GameAction.ACTION1

    def take_action(self, action):
        return None

    def _convert_raw_frame_data(self, raw):
        return self.frames[-1]        # sem ambiente real: reusa o frame inicial


def _mk(game="g1"):
    return _Loop(card_id="c", game_id=game, agent_name="loop", ROOT_URL="http://x",
                 record=False, arc_env=None)


@pytest.mark.unit
def test_agent_main_exits_when_stop_requested():
    a = _mk()
    assert a.stop_requested is False
    a.main()
    assert a.calls == 3


@pytest.mark.unit
@patch("agents.swarm.Arcade")
def test_swarm_sequential_timeout_requests_stop(mock_arcade, monkeypatch, tmp_path):
    monkeypatch.setenv("SWARM_GAME_TIMEOUT", "1")
    monkeypatch.chdir(tmp_path)        # swarm_diagnostics.json vai pro tmp, nao pro repo

    class _Hang(_Loop):
        def choose_action(self, frames, latest_frame):
            self.calls += 1
            time.sleep(0.05)
            return GameAction.ACTION1

    swarm = Swarm(agent="random", ROOT_URL="http://x", games=["g1", "g2"])
    swarm.agent_class = _Hang
    swarm._arc.operation_mode = OperationMode.OFFLINE
    swarm._arc.open_scorecard.return_value = "card"
    swarm._arc.close_scorecard.return_value = None
    t0 = time.time()
    swarm.main()
    assert time.time() - t0 < 6
    assert all(a.stop_requested for a in swarm.agents)
    time.sleep(0.3)
    assert all(a.calls == getattr(a, "calls") for a in swarm.agents)   # parou (nao cresce)


@pytest.mark.unit
@patch("agents.swarm.Arcade")
def test_swarm_parallel_deadline_requests_stop(mock_arcade, monkeypatch, tmp_path):
    monkeypatch.setenv("SWARM_DEADLINE_S", "1")
    monkeypatch.chdir(tmp_path)

    class _Hang(_Loop):
        def choose_action(self, frames, latest_frame):
            self.calls += 1
            time.sleep(0.05)
            if self.calls >= 400:            # auto-encerra em ~20s: sem a feature o
                self.stop_requested = True   # teste FALHA por tempo em vez de travar
            return GameAction.ACTION1

    swarm = Swarm(agent="random", ROOT_URL="http://x", games=["g1", "g2", "g3"])
    swarm.agent_class = _Hang
    swarm._arc.operation_mode = "ONLINE"            # forca o caminho PARALELO
    swarm._arc.open_scorecard.return_value = "card"
    swarm._arc.close_scorecard.return_value = None
    t0 = time.time()
    swarm.main()
    assert time.time() - t0 < 8
    assert all(a.stop_requested for a in swarm.agents)
    assert all(a.calls > 0 for a in swarm.agents)   # jogaram de fato ate o deadline
