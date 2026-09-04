from __future__ import annotations

import json
import logging
import os
import time
from threading import Thread
from typing import TYPE_CHECKING, Optional, Type

from arc_agi import Arcade, OperationMode
from arc_agi.scorecard import EnvironmentScorecard

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger()

# Prioridade dos jogos: tier menor = roda primeiro. Jogos que ja pontuaram
# e click-only (rapidos, ~20fps) rodam antes dos que precisam de LLM (~0.04fps).
# Assim, mesmo com timeout, os pontos faceis ja estao garantidos.
_GAME_TIERS: dict[str, int] = {
    # Tier 0: ja pontuaram (1L confirmado), click-only
    "vc33": 0, "tn36": 0, "lp85": 0,
    # Tier 1: click-only restantes (rapidos, sem LLM)
    "r11l": 1, "s5i5": 1, "ft09": 1, "su15": 1, "lf52": 1,
    # Tier 2: navegacao avatar→alvo (teclado simples, heuristica navigate funciona)
    "dc22": 2, "g50t": 2, "tu93": 2, "sc25": 2, "ls20": 2, "m0r0": 2,
    # Tier 3: sokoban / manipulacao (teclado + clique, mecanica complexa)
    "ka59": 3, "wa30": 3, "ar25": 3, "cn04": 3,
    # Tier 4: sequencia/pintura/fluxo (precisa de LLM ou mecanica complexa)
    "sb26": 4, "tr87": 4, "sk48": 4, "cd82": 4, "re86": 4, "sp80": 4,
    # Tier 5: desconhecido
    "bp35": 5,
}


def prioritize_games(games: list[str]) -> list[str]:
    """Ordena jogos por prioridade: pontuados > click-only > navegacao > complexos."""
    def _tier(g: str) -> int:
        for prefix, t in _GAME_TIERS.items():
            if g.startswith(prefix):
                return t
        return 9
    return sorted(games, key=_tier)


class Swarm:
    """Orchestration for many agents playing many ARC-AGI-3 games."""

    GAMES: list[str]
    ROOT_URL: str
    COUNT: int
    agent_name: str
    agent_class: Type[Agent]
    threads: list[Thread]
    agents: list[Agent]
    record_games: list[str]
    cleanup_threads: list[Thread]
    headers: dict[str, str]
    card_id: Optional[str]
    _arc: Arcade

    def __init__(
        self,
        agent: str,
        ROOT_URL: str,
        games: list[str],
        tags: list[str] = [],
    ) -> None:
        from . import AVAILABLE_AGENTS

        self.GAMES = games
        self.ROOT_URL = ROOT_URL
        self.agent_name = agent
        self.agent_class = AVAILABLE_AGENTS[agent]
        self.threads = []
        self.agents = []
        self.cleanup_threads = []
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self.tags = tags.copy() if tags is not None else []
        self._arc = Arcade()

        # Set up base tags for tracing
        if self.agent_name.endswith(".recording.jsonl"):
            # Extract GUID from playback filename
            # Format: game.agent.count.guid.recording.jsonl
            parts = self.agent_name.split(".")
            guid = parts[-3] if len(parts) >= 4 else "unknown"
            self.tags.extend(["playback", guid])
        else:
            self.tags.extend(["agent", self.agent_name])

    def main(self) -> EnvironmentScorecard | None:
        """The main orchestration loop, continues until all agents are done."""

        # submit start of scorecard
        print("***** MAKING SCORECARD")
        self.card_id = self.open_scorecard()

        # Offline: prioriza jogos e roda sequencialmente (1 por vez).
        # Evita contencao de GPU (25 threads disputando 1 modelo) e garante
        # que jogos faceis (click-only, ~20fps) rodam antes dos lentos (LLM, ~0.04fps).
        sequential = self._arc.operation_mode == OperationMode.OFFLINE
        ordered = prioritize_games(self.GAMES) if sequential else self.GAMES

        print(f"***** MAKING AGENTS with card id: {self.card_id}"
              f" (mode={'SEQUENTIAL' if sequential else 'PARALLEL'},"
              f" {len(ordered)} games)")
        if sequential:
            logger.info(f"Game order (prioritized): {ordered}")

        # Diagnostico por-jogo: coletado no modo sequencial
        self._game_results: list[dict] = []

        for g in ordered:
            a = self.agent_class(
                card_id=self.card_id,
                game_id=g,
                agent_name=self.agent_name,
                ROOT_URL=self.ROOT_URL,
                record=True,
                arc_env=self._arc.make(g, scorecard_id=self.card_id),
                tags=self.tags,
            )
            self.agents.append(a)

        # Per-game timeout: evita que 1 jogo lento (LLM OOM/deadlock) trave tudo.
        game_timeout = int(os.environ.get("SWARM_GAME_TIMEOUT", "120"))

        if sequential:
            for a in self.agents:
                t0 = time.time()
                logger.info(f">>> STARTING {a.game_id}")
                t = Thread(target=a.main, daemon=True)
                t.start()
                t.join(timeout=game_timeout)
                if t.is_alive():
                    logger.warning(f"!!! TIMEOUT {a.game_id} after {game_timeout}s")
                elapsed = time.time() - t0
                levels = getattr(a, "levels_completed", 0)
                acts = getattr(a, "action_counter", 0)
                fps = round(acts / max(elapsed, 0.1), 2)
                self._game_results.append({
                    "game_id": a.game_id, "levels": levels,
                    "actions": acts, "time_s": round(elapsed, 1), "fps": fps,
                })
                logger.info(f"<<< DONE {a.game_id}: {levels}L, "
                            f"{acts} actions, {elapsed:.1f}s, {fps} fps")
        else:
            for a in self.agents:
                self.threads.append(Thread(target=a.main, daemon=True))
            for t in self.threads:
                t.start()
            for t in self.threads:
                t.join()

        # all agents are now done
        card_id = self.card_id
        scorecard = self.close_scorecard(card_id)
        if scorecard:
            logger.info("--- FINAL SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))

        # Provide web link to scorecard
        if card_id:
            if self._arc.operation_mode == OperationMode.ONLINE:
                scorecard_url = f"{self.ROOT_URL}/scorecards/{card_id}"
                logger.info(f"View your scorecard online: {scorecard_url}")
            else:
                logger.info(
                    "Online scorecard is not available, to use the online API set the ONLINE_ONLY envvar to True"
                )

        # Diagnostico estruturado: exporta resumo por-jogo
        if self._game_results:
            self._export_diagnostics()

        self.cleanup(scorecard)

        return scorecard

    def _export_diagnostics(self) -> None:
        """Exporta diagnostico por-jogo em JSON (facil de analisar entre runs)."""
        try:
            out = {
                "total_games": len(self._game_results),
                "total_levels": sum(r["levels"] for r in self._game_results),
                "total_time_s": round(sum(r["time_s"] for r in self._game_results), 1),
                "games": self._game_results,
            }
            path = "/kaggle/working/swarm_diagnostics.json"
            if not os.path.isdir(os.path.dirname(path)):
                path = "swarm_diagnostics.json"
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            logger.info(f"Diagnostics -> {path}")
            # Tabela resumo no log
            logger.info("=== GAME RESULTS ===")
            for r in self._game_results:
                logger.info(f"  {r['game_id']:6s} | {r['levels']}L | "
                            f"{r['actions']:3d} acts | {r['time_s']:6.1f}s | "
                            f"{r['fps']:6.2f} fps")
            logger.info(f"  TOTAL: {out['total_levels']}L in {out['total_time_s']}s")
        except Exception as e:
            logger.warning(f"Failed to export diagnostics: {e}")

    def open_scorecard(self) -> str:
        return self._arc.open_scorecard(tags=self.tags)  # type: ignore[no-any-return]

    def close_scorecard(self, card_id: str) -> Optional[EnvironmentScorecard]:
        self.card_id = None

        return self._arc.close_scorecard(card_id)

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        """Cleanup all agents."""
        for a in self.agents:
            a.cleanup(scorecard)
        if hasattr(self, "_session"):
            self._session.close()
