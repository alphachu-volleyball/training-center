"""Round-robin evaluation script (ELO + detailed stats).

Usage:
  uv run evaluate --players random,builtin,experiments/baseline/model --games 50
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import wandb

from training_center.eval.elo import INITIAL_ELO, update_elo
from training_center.eval.game import make_player, play_game
from training_center.metadata import get_experiment_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin evaluation with detailed stats")
    parser.add_argument("--players", required=True, help="Comma-separated: random, builtin, or model path")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--score", type=int, default=15, help="Winning score")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity (user or team)")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    meta = get_experiment_metadata()

    player_specs = [s.strip() for s in args.players.split(",")]
    players = [make_player(spec) for spec in player_specs]
    rng = np.random.default_rng(args.seed)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "script": "evaluate",
            "players": player_specs,
            "games_per_pair": args.games,
            "winning_score": args.score,
            "seed": args.seed,
            **meta,
        },
    )

    print(f"Players: {[p.name for p in players]}")
    print(f"Games per pair: {args.games}")
    print(f"Winning score: {args.score}")

    elos = {p.name: INITIAL_ELO for p in players}

    for p1, p2 in combinations(players, 2):
        all_rounds = []
        p1_wins = 0
        all_stats = []

        for _ in range(args.games):
            game_seed = int(rng.integers(0, 2**31))
            episode = play_game(p1, p2, winning_score=args.score, seed=game_seed)
            all_stats.append(episode)
            result = 1 if episode.winner == "player_1" else 0
            p1_wins += result
            elos[p1.name], elos[p2.name] = update_elo(elos[p1.name], elos[p2.name], result)
            all_rounds.extend(episode.rounds)

        p2_wins = args.games - p1_wins
        avg_p1_score = np.mean([e.scores[0] for e in all_stats])
        avg_p2_score = np.mean([e.scores[1] for e in all_stats])

        p1_serve = [r for r in all_rounds if r.server == "player_1"]
        p2_serve = [r for r in all_rounds if r.server == "player_2"]
        durations = [r.duration for r in all_rounds]

        matchup_key = f"{p1.name}_vs_{p2.name}"
        p1s_p1w = sum(1 for r in p1_serve if r.scorer == "player_1") if p1_serve else 0
        p2s_p2w = sum(1 for r in p2_serve if r.scorer == "player_2") if p2_serve else 0

        run.log(
            {
                f"{matchup_key}/p1_wins": p1_wins,
                f"{matchup_key}/p2_wins": p2_wins,
                f"{matchup_key}/p1_win_rate": p1_wins / args.games,
                f"{matchup_key}/avg_p1_score": float(avg_p1_score),
                f"{matchup_key}/avg_p2_score": float(avg_p2_score),
                f"{matchup_key}/p1_serve_win_rate": p1s_p1w / max(len(p1_serve), 1),
                f"{matchup_key}/p2_serve_win_rate": p2s_p2w / max(len(p2_serve), 1),
                f"{matchup_key}/avg_round_frames": float(np.mean(durations)) if durations else 0,
                f"{matchup_key}/median_round_frames": float(np.median(durations)) if durations else 0,
            }
        )

        print(f"\n{'=' * 60}")
        print(f"  {p1.name} (p1) vs {p2.name} (p2)")
        print(f"{'=' * 60}")
        print(f"  Record: p1 {p1_wins}W {p2_wins}L ({p1_wins / args.games * 100:.0f}%)")
        print(f"  Avg score: p1 {avg_p1_score:.1f} - p2 {avg_p2_score:.1f}")

        if p1_serve:
            p1s_p2w = len(p1_serve) - p1s_p1w
            print(
                f"\n  [p1 serve] p1 wins {p1s_p1w:4d} ({p1s_p1w / len(p1_serve) * 100:5.1f}%)"
                f"  |  p2 wins {p1s_p2w:4d} ({p1s_p2w / len(p1_serve) * 100:5.1f}%)"
                f"  |  {len(p1_serve)} rounds"
            )
        if p2_serve:
            p2s_p1w = len(p2_serve) - p2s_p2w
            print(
                f"  [p2 serve] p1 wins {p2s_p1w:4d} ({p2s_p1w / len(p2_serve) * 100:5.1f}%)"
                f"  |  p2 wins {p2s_p2w:4d} ({p2s_p2w / len(p2_serve) * 100:5.1f}%)"
                f"  |  {len(p2_serve)} rounds"
            )

        print(
            f"\n  Round frames: mean {np.mean(durations):.0f}"
            f"  median {np.median(durations):.0f}"
            f"  range {np.min(durations)}-{np.max(durations)}"
        )

    # Log final ELO ratings
    elo_data = {f"elo/{name}": elo for name, elo in elos.items()}
    run.log(elo_data)

    print(f"\n{'=' * 60}")
    print("  ELO Ratings")
    print(f"{'=' * 60}")
    for name, elo in sorted(elos.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {elo:7.1f}")

    run.finish()


if __name__ == "__main__":
    main()
