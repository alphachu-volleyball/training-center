"""Round-robin evaluation script (ELO + detailed stats).

Cross-product of p1 pool × p2 pool. Each model is evaluated on its training side.
Games are executed in parallel using ProcessPoolExecutor.

Usage:
  uv run evaluate --p1 random,builtin,experiments/001/model --p2 random,builtin,experiments/003/model --games 50
  uv run evaluate --p1 random,builtin,duckll:0,duckll:10 --p2 random,builtin,duckll:0,duckll:10 --workers 8
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import wandb

from training_center.elo import INITIAL_ELO, update_elo
from training_center.game import make_player, play_game
from training_center.metadata import get_experiment_metadata


def _play_single_game(
    p1_spec: str,
    p2_spec: str,
    simplify_observation: bool,
    winning_score: int,
    seed: int,
) -> tuple[str, str, dict]:
    """Worker function for parallel game execution.

    Reconstructs Player objects inside each worker to avoid pickling AIPolicy.
    Returns (p1_name, p2_name, serialized GameRecord fields).
    """
    p1 = make_player(p1_spec, agent="player_1", simplify_observation=simplify_observation)
    p2 = make_player(p2_spec, agent="player_2", simplify_observation=simplify_observation)
    episode = play_game(p1, p2, winning_score=winning_score, seed=seed)
    return (
        p1.name,
        p2.name,
        {
            "winner": episode.winner,
            "scores": episode.scores,
            "rounds": [(r.server, r.scorer, r.duration) for r in episode.rounds],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin evaluation with detailed stats")
    parser.add_argument("--p1", required=True, help="Comma-separated p1 players: random, builtin, or model path")
    parser.add_argument("--p2", required=True, help="Comma-separated p2 players: random, builtin, or model path")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--winning-score", type=int, default=5, help="Points to win per game")
    parser.add_argument(
        "--simplify-observation", action="store_true", help="Models were trained with SimplifyObservation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: cpu_count)")
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity (user or team)")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    meta = get_experiment_metadata()
    n_workers = args.workers or os.cpu_count()

    p1_specs = [s.strip() for s in args.p1.split(",")]
    p2_specs = [s.strip() for s in args.p2.split(",")]

    # Resolve names (create players once just for name resolution)
    so = args.simplify_observation
    p1_names = {s: make_player(s, agent="player_1", simplify_observation=so).name for s in p1_specs}
    p2_names = {s: make_player(s, agent="player_2", simplify_observation=so).name for s in p2_specs}

    # Build matchup list (skip self-play)
    matchups: list[tuple[str, str, str, str]] = []  # (p1_spec, p2_spec, p1_name, p2_name)
    for p1_spec, p2_spec in product(p1_specs, p2_specs):
        p1_name = p1_names[p1_spec]
        p2_name = p2_names[p2_spec]
        if p1_name == p2_name:
            continue
        matchups.append((p1_spec, p2_spec, p1_name, p2_name))

    # Pre-generate all tasks with deterministic seeds
    rng = np.random.default_rng(args.seed)
    tasks: list[tuple[str, str, int]] = []
    for p1_spec, p2_spec, _, _ in matchups:
        for _ in range(args.games):
            game_seed = int(rng.integers(0, 2**31))
            tasks.append((p1_spec, p2_spec, game_seed))

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "script": "evaluate",
            "p1_players": p1_specs,
            "p2_players": p2_specs,
            "games_per_pair": args.games,
            "winning_score": args.winning_score,
            "seed": args.seed,
            "workers": n_workers,
            **meta,
        },
    )

    print(f"P1 players: {list(p1_names.values())}")
    print(f"P2 players: {list(p2_names.values())}")
    print(f"Matchups: {len(matchups)}, Games per pair: {args.games}, Total: {len(tasks)}")
    print(f"Workers: {n_workers}")

    # Execute all games in parallel
    print(f"\nPlaying {len(tasks)} games...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        all_results = list(
            executor.map(
                _play_single_game,
                [t[0] for t in tasks],
                [t[1] for t in tasks],
                [args.simplify_observation] * len(tasks),
                [args.winning_score] * len(tasks),
                [t[2] for t in tasks],
            )
        )
    print("All games complete.", flush=True)

    # Process results per matchup (order preserved by executor.map)
    elos: dict[str, float] = {}
    for _, _, p1_name, p2_name in matchups:
        if p1_name not in elos:
            elos[p1_name] = INITIAL_ELO
        if p2_name not in elos:
            elos[p2_name] = INITIAL_ELO

    table_rows: list[list] = []

    idx = 0
    for i, (p1_spec, p2_spec, p1_name, p2_name) in enumerate(matchups):
        matchup_results = all_results[idx : idx + args.games]
        idx += args.games

        p1_wins = 0
        all_rounds_data: list[tuple[str, str, int]] = []
        all_scores: list[tuple[int, int]] = []

        for _, _, data in matchup_results:
            result = 1 if data["winner"] == "player_1" else 0
            p1_wins += result
            elos[p1_name], elos[p2_name] = update_elo(elos[p1_name], elos[p2_name], result)
            all_rounds_data.extend(data["rounds"])
            all_scores.append(tuple(data["scores"]))

        p2_wins = args.games - p1_wins
        avg_p1_score = float(np.mean([s[0] for s in all_scores]))
        avg_p2_score = float(np.mean([s[1] for s in all_scores]))

        p1_serve = [(server, scorer, dur) for server, scorer, dur in all_rounds_data if server == "player_1"]
        p2_serve = [(server, scorer, dur) for server, scorer, dur in all_rounds_data if server == "player_2"]
        durations = [dur for _, _, dur in all_rounds_data]

        p1_serve_win = sum(1 for _, scorer, _ in p1_serve if scorer == "player_1") if p1_serve else 0
        p2_serve_win = sum(1 for _, scorer, _ in p2_serve if scorer == "player_2") if p2_serve else 0
        p1_receive_win = sum(1 for _, scorer, _ in p2_serve if scorer == "player_1") if p2_serve else 0
        p2_receive_win = sum(1 for _, scorer, _ in p1_serve if scorer == "player_2") if p1_serve else 0

        table_rows.append(
            [
                p1_name,
                p2_name,
                p1_wins / args.games,
                avg_p1_score,
                avg_p2_score,
                p1_serve_win / max(len(p1_serve), 1),
                p1_receive_win / max(len(p2_serve), 1),
                p2_serve_win / max(len(p2_serve), 1),
                p2_receive_win / max(len(p1_serve), 1),
                float(np.mean(durations)) if durations else 0,
                float(np.median(durations)) if durations else 0,
            ]
        )

        print(
            f"  [{i + 1}/{len(matchups)}] {p1_name} vs {p2_name}: "
            f"{p1_wins}W {p2_wins}L ({p1_wins / args.games * 100:.0f}%) "
            f"score {avg_p1_score:.1f}-{avg_p2_score:.1f} "
            f"round {np.mean(durations):.0f}f",
            flush=True,
        )

    # Log matchup results as wandb.Table
    matchup_table = wandb.Table(
        columns=[
            "p1",
            "p2",
            "p1_win_rate",
            "p1_avg_score",
            "p2_avg_score",
            "p1_serve_win_rate",
            "p1_receive_win_rate",
            "p2_serve_win_rate",
            "p2_receive_win_rate",
            "avg_round_frames",
            "median_round_frames",
        ],
        data=table_rows,
    )
    run.log({"matchups": matchup_table})

    # Log ELO ratings as wandb.Table and run.summary
    elo_table = wandb.Table(
        columns=["agent", "elo"],
        data=[[name, elo] for name, elo in sorted(elos.items(), key=lambda x: x[1], reverse=True)],
    )
    run.log({"elo_ratings": elo_table})
    for name, elo in elos.items():
        run.summary[f"elo/{name}"] = elo

    print(f"\n{'=' * 60}")
    print("  ELO Ratings")
    print(f"{'=' * 60}")
    for name, elo in sorted(elos.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {elo:7.1f}")

    run.finish()


if __name__ == "__main__":
    main()
