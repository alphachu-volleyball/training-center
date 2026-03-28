"""Detailed game statistics collection and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.env.pikachu_volleyball import PikachuVolleyballEnv
from pika_zoo.wrappers.normalize_observation import NormalizeObservation
from pika_zoo.wrappers.simplify_action import SimplifyAction

from training_center.eval.elo import Player, make_player


@dataclass
class RoundStats:
    server: str  # "player_1" or "player_2"
    winner: str  # "player_1" or "player_2"
    rally_length: int


@dataclass
class GameStats:
    p1_score: int = 0
    p2_score: int = 0
    rounds: list[RoundStats] = field(default_factory=list)
    truncated_rallies: int = 0
    seed: int | None = None

    @property
    def winner(self) -> str:
        if self.p1_score == self.p2_score:
            return "draw"
        return "player_1" if self.p1_score > self.p2_score else "player_2"

    @property
    def p1_serve_win(self) -> int:
        return sum(1 for r in self.rounds if r.server == "player_1" and r.winner == "player_1")

    @property
    def p1_serve_total(self) -> int:
        return sum(1 for r in self.rounds if r.server == "player_1")

    @property
    def p2_serve_win(self) -> int:
        return sum(1 for r in self.rounds if r.server == "player_2" and r.winner == "player_2")

    @property
    def p2_serve_total(self) -> int:
        return sum(1 for r in self.rounds if r.server == "player_2")

    @property
    def avg_rally_length(self) -> float:
        if not self.rounds:
            return 0
        return sum(r.rally_length for r in self.rounds) / len(self.rounds)


MAX_RALLY_STEPS = 3000
MAX_GAME_STEPS = 30000


def play_game_detailed(
    p1: Player,
    p2: Player,
    winning_score: int = 15,
    seed: int | None = None,
    max_game_steps: int = MAX_GAME_STEPS,
) -> GameStats:
    """Play one game with detailed round-by-round statistics.

    Truncates infinite rallies at MAX_RALLY_STEPS and games at max_game_steps.
    """
    ai_policies: dict[str, BuiltinAI] = {}
    if p1.player_type == "builtin":
        ai_policies["player_1"] = BuiltinAI()
    if p2.player_type == "builtin":
        ai_policies["player_2"] = BuiltinAI()

    env = PikachuVolleyballEnv(winning_score=winning_score, serve="winner", ai_policies=ai_policies or None)
    env = SimplifyAction(env)
    env = NormalizeObservation(env)

    obs, _info = env.reset(seed=seed)
    stats = GameStats()
    total_steps = 0
    rally_steps = 0
    current_server = "player_1"
    truncated_rallies = 0

    while env.agents:
        actions = {
            "player_1": p1.get_action(obs.get("player_1"), env, "player_1"),
            "player_2": p2.get_action(obs.get("player_2"), env, "player_2"),
        }
        obs, rewards, _terminated, _truncated, _infos = env.step(actions)
        rally_steps += 1
        total_steps += 1

        if total_steps >= max_game_steps:
            if rally_steps > 0:
                truncated_rallies += 1
                stats.rounds.append(RoundStats(server=current_server, winner="draw", rally_length=rally_steps))
            break

        if rally_steps >= MAX_RALLY_STEPS:
            truncated_rallies += 1
            stats.rounds.append(RoundStats(server=current_server, winner="draw", rally_length=rally_steps))
            rally_steps = 0
            continue

        if rewards.get("player_1", 0) != 0:
            round_winner = "player_1" if rewards["player_1"] > 0 else "player_2"
            stats.rounds.append(RoundStats(server=current_server, winner=round_winner, rally_length=rally_steps))
            if round_winner == "player_1":
                stats.p1_score += 1
            else:
                stats.p2_score += 1
            current_server = round_winner
            rally_steps = 0

    env.close()
    stats.truncated_rallies = truncated_rallies
    stats.seed = seed
    return stats


def analyze_games(
    p1_spec: str,
    p2_spec: str,
    games: int = 100,
    winning_score: int = 15,
    seed: int = 42,
) -> list[GameStats]:
    """Run multiple games and print aggregate statistics."""
    p1 = make_player(p1_spec)
    p2 = make_player(p2_spec)
    rng = np.random.default_rng(seed)

    all_stats: list[GameStats] = []
    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        stats = play_game_detailed(p1, p2, winning_score=winning_score, seed=game_seed)
        all_stats.append(stats)

    p1_wins = sum(1 for s in all_stats if s.winner == "player_1")
    all_rounds = [r for s in all_stats for r in s.rounds]
    p1_serve_rounds = [r for r in all_rounds if r.server == "player_1"]
    p2_serve_rounds = [r for r in all_rounds if r.server == "player_2"]
    rally_lengths = [r.rally_length for r in all_rounds]

    print(f"=== {p1.name} (p1) vs {p2.name} (p2) — {games} games, {winning_score} pts ===\n")
    print(f"Record: p1 {p1_wins}W {games - p1_wins}L ({p1_wins / games * 100:.0f}%)\n")

    if p1_serve_rounds:
        p1s_p1w = sum(1 for r in p1_serve_rounds if r.winner == "player_1")
        pct = p1s_p1w / len(p1_serve_rounds) * 100
        print(f"p1 serve → p1 wins: {p1s_p1w}/{len(p1_serve_rounds)} ({pct:.1f}%)")
    if p2_serve_rounds:
        p2s_p2w = sum(1 for r in p2_serve_rounds if r.winner == "player_2")
        pct = p2s_p2w / len(p2_serve_rounds) * 100
        print(f"p2 serve → p2 wins: {p2s_p2w}/{len(p2_serve_rounds)} ({pct:.1f}%)")

    if rally_lengths:
        print(
            f"\nRally length: mean {np.mean(rally_lengths):.1f}, "
            f"median {np.median(rally_lengths):.1f}, "
            f"range {np.min(rally_lengths)}-{np.max(rally_lengths)}"
        )

    return all_stats
