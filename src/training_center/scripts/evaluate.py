"""Round-robin evaluation script (ELO + detailed stats).

Cross-product of p1 pool × p2 pool. Each model is evaluated on its training side.

Usage:
  uv run evaluate --p1 random,builtin,experiments/001/model --p2 random,builtin,experiments/003/model --games 50
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import wandb

from training_center.elo import INITIAL_ELO, update_elo
from training_center.game import make_player, play_game
from training_center.metadata import get_experiment_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin evaluation with detailed stats")
    parser.add_argument("--p1", required=True, help="Comma-separated p1 players: random, builtin, or model path")
    parser.add_argument("--p2", required=True, help="Comma-separated p2 players: random, builtin, or model path")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--score", type=int, default=15, help="Winning score")
    parser.add_argument(
        "--simplify-observation", action="store_true", help="Models were trained with SimplifyObservation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity (user or team)")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    meta = get_experiment_metadata()

    p1_specs = [s.strip() for s in args.p1.split(",")]
    p2_specs = [s.strip() for s in args.p2.split(",")]
    p1_players = [
        make_player(spec, agent="player_1", simplify_observation=args.simplify_observation) for spec in p1_specs
    ]
    p2_players = [
        make_player(spec, agent="player_2", simplify_observation=args.simplify_observation) for spec in p2_specs
    ]
    rng = np.random.default_rng(args.seed)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "script": "evaluate",
            "p1_players": p1_specs,
            "p2_players": p2_specs,
            "games_per_pair": args.games,
            "winning_score": args.score,
            "seed": args.seed,
            **meta,
        },
    )

    print(f"P1 players: {[p.name for p in p1_players]}")
    print(f"P2 players: {[p.name for p in p2_players]}")
    print(f"Games per pair: {args.games}")
    print(f"Winning score: {args.score}")

    elos = {}
    for p in p1_players + p2_players:
        if p.name not in elos:
            elos[p.name] = INITIAL_ELO

    for p1, p2 in product(p1_players, p2_players):
        if p1.name == p2.name:
            continue

        all_rounds = []
        p1_wins = 0
        all_episodes = []

        for _ in range(args.games):
            game_seed = int(rng.integers(0, 2**31))
            episode = play_game(p1, p2, winning_score=args.score, seed=game_seed)
            all_episodes.append(episode)
            result = 1 if episode.winner == "player_1" else 0
            p1_wins += result
            elos[p1.name], elos[p2.name] = update_elo(elos[p1.name], elos[p2.name], result)
            all_rounds.extend(episode.rounds)

        p2_wins = args.games - p1_wins
        avg_p1_score = np.mean([e.scores[0] for e in all_episodes])
        avg_p2_score = np.mean([e.scores[1] for e in all_episodes])

        p1_serve = [r for r in all_rounds if r.server == "player_1"]
        p2_serve = [r for r in all_rounds if r.server == "player_2"]
        durations = [r.duration for r in all_rounds]

        matchup_key = f"{p1.name}_vs_{p2.name}"
        p1_serve_win = sum(1 for r in p1_serve if r.scorer == "player_1") if p1_serve else 0
        p2_receive_win = sum(1 for r in p1_serve if r.scorer == "player_2") if p1_serve else 0
        p2_serve_win = sum(1 for r in p2_serve if r.scorer == "player_2") if p2_serve else 0
        p1_receive_win = sum(1 for r in p2_serve if r.scorer == "player_1") if p2_serve else 0

        run.log(
            {
                f"{matchup_key}/p1_win_rate": p1_wins / args.games,
                f"{matchup_key}/p1_avg_score": float(avg_p1_score),
                f"{matchup_key}/p2_avg_score": float(avg_p2_score),
                f"{matchup_key}/p1_serve_win_rate": p1_serve_win / max(len(p1_serve), 1),
                f"{matchup_key}/p1_receive_win_rate": p1_receive_win / max(len(p2_serve), 1),
                f"{matchup_key}/p2_serve_win_rate": p2_serve_win / max(len(p2_serve), 1),
                f"{matchup_key}/p2_receive_win_rate": p2_receive_win / max(len(p1_serve), 1),
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
            print(
                f"\n  [p1 serve] p1 wins {p1_serve_win:4d} ({p1_serve_win / len(p1_serve) * 100:5.1f}%)"
                f"  |  p2 wins {p2_receive_win:4d} ({p2_receive_win / len(p1_serve) * 100:5.1f}%)"
                f"  |  {len(p1_serve)} rounds"
            )
        if p2_serve:
            print(
                f"  [p2 serve] p1 wins {p1_receive_win:4d} ({p1_receive_win / len(p2_serve) * 100:5.1f}%)"
                f"  |  p2 wins {p2_serve_win:4d} ({p2_serve_win / len(p2_serve) * 100:5.1f}%)"
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
