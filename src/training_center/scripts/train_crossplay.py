"""Cross-play training script (PFP + configurable anchor).

Alternately trains p1_model (left) and p2_model (right) as separate models.
Opponent mix: anchor_prob (rule AI) + remaining (PFP pool).

Usage:
  uv run train-crossplay --total-iterations 100 --steps-per-iter 20000 --save-dir experiments/001
  uv run train-crossplay --p1-init exp/001/p1 --p2-init exp/001/p2 --save-dir experiments/002
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import wandb
from pika_zoo.ai import BuiltinAI, DuckllAI, StoneAI
from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.records.types import GamesRecord
from stable_baselines3 import PPO

from training_center.elo import compute_elo
from training_center.env_factory import ensure_stack_size, make_vec_env, set_opponent_policy
from training_center.game import make_player, play_game
from training_center.metadata import get_experiment_metadata
from training_center.metrics import compute_eval_metrics
from training_center.model_config import ModelConfig, save_model
from training_center.pool import OpponentPool, make_opponent_policy
from training_center.scripts.utils import (
    build_eval_log_data,
    parse_noise,
    record_video,
    setup_graceful_shutdown,
    shutdown_executor,
    worker_init,
)


def _log_sb3_metrics(run: wandb.sdk.wandb_run.Run, model: PPO, prefix: str) -> None:
    """Read SB3 logger metrics and log to wandb with a prefix."""
    if model.logger is not None and hasattr(model.logger, "name_to_value"):
        metrics = {f"{prefix}/{k}": v for k, v in model.logger.name_to_value.items()}
        if metrics:
            run.log(metrics, step=model.num_timesteps)


def _log_model_artifact(run: wandb.sdk.wandb_run.Run, name: str, path: str) -> None:
    """Log a saved model directory as a wandb artifact."""
    artifact = wandb.Artifact(name, type="model")
    p = Path(path)
    if p.is_dir():
        artifact.add_dir(str(p))
    else:
        artifact.add_file(path + ".zip")
    run.log_artifact(artifact)


def _run_matchup_worker(
    name: str,
    p1_spec: str,
    p2_spec: str,
    games: int,
    winning_score: int,
    perspective: str,
    seed: int,
    simplify_observation: bool,
) -> tuple[str, dict]:
    """Worker: run a matchup evaluation in a child process.

    Reconstructs Player objects from specs (model paths or AI names)
    to avoid pickling PPO/AIPolicy objects across process boundaries.
    """
    p1 = make_player(p1_spec, agent="player_1", simplify_observation=simplify_observation)
    p2 = make_player(p2_spec, agent="player_2", simplify_observation=simplify_observation)
    rng = np.random.default_rng(seed)
    rounds_all = []
    all_stats = []
    wins = 0

    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        episode = play_game(p1, p2, winning_score=winning_score, seed=game_seed, record_frames=True)
        all_stats.append(episode)
        if perspective == "p1":
            wins += 1 if episode.winner == "player_1" else 0
        else:
            wins += 1 if episode.winner == "player_2" else 0
        rounds_all.extend(episode.rounds)

    summary = _summarize(wins, games, rounds_all, all_stats, perspective)
    return name, summary


def _eval_checkpoint_worker(
    current_model_path: str,
    checkpoint_path: str,
    side: str,
    games: int,
    winning_score: int,
    simplify_observation: bool,
    seed: int,
) -> tuple[str, list[bool]]:
    """Worker: evaluate current model vs one pool checkpoint.

    Returns (checkpoint_name, list of win booleans).
    """
    current = make_player(
        current_model_path, agent="player_1" if side == "p1" else "player_2", simplify_observation=simplify_observation
    )
    opp = make_player(
        checkpoint_path, agent="player_2" if side == "p1" else "player_1", simplify_observation=simplify_observation
    )
    rng = np.random.default_rng(seed)
    wins: list[bool] = []
    model_side = "player_1" if side == "p1" else "player_2"

    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        if side == "p1":
            stats = play_game(current, opp, winning_score=winning_score, seed=game_seed)
        else:
            stats = play_game(opp, current, winning_score=winning_score, seed=game_seed)
        wins.append(stats.winner == model_side)

    name = Path(checkpoint_path).name
    return name, wins


def evaluate_crossplay_detailed(
    p1_model_path: str,
    p2_model_path: str,
    games: int = 20,
    winning_score: int = 15,
    seed: int = 42,
    simplify_observation: bool = False,
    eval_opponents: list[str] | None = None,
) -> dict[str, dict]:
    """Evaluate p1/p2 models against each other and eval opponents."""
    rng = np.random.default_rng(seed)
    opponents = eval_opponents or ["random", "builtin"]

    matchup_defs: list[tuple[str, str, str, str]] = [
        ("p1_vs_p2", p1_model_path, p2_model_path, "p1"),
    ]
    for opp in opponents:
        matchup_defs.append((f"p1_vs_{opp}", p1_model_path, opp, "p1"))
        matchup_defs.append((f"p2_vs_{opp}", opp, p2_model_path, "p2"))

    matchups: dict[str, dict] = {}
    for mname, p1s, p2s, perspective in matchup_defs:
        matchup_seed = int(rng.integers(0, 2**31))
        mname, summary = _run_matchup_worker(
            mname,
            p1s,
            p2s,
            games,
            winning_score,
            perspective,
            matchup_seed,
            simplify_observation,
        )
        matchups[mname] = summary
    return matchups


def _summarize(
    wins: int,
    games: int,
    rounds: list,
    all_stats: list,
    perspective: str,
) -> dict:
    """Aggregate statistics over multiple games."""
    model_side = "player_1" if perspective == "p1" else "player_2"

    if perspective == "p1":
        avg_score = float(np.mean([e.scores[0] for e in all_stats])) if all_stats else 0
        avg_opp_score = float(np.mean([e.scores[1] for e in all_stats])) if all_stats else 0
    else:
        avg_score = float(np.mean([e.scores[1] for e in all_stats])) if all_stats else 0
        avg_opp_score = float(np.mean([e.scores[0] for e in all_stats])) if all_stats else 0

    detail = compute_eval_metrics(GamesRecord(games=all_stats), model_side)

    return {
        "wins": wins,
        "losses": games - wins,
        "win_rate": wins / games,
        "avg_score": avg_score,
        "avg_opp_score": avg_opp_score,
        **detail,
    }


def _update_pool_stats(
    model_path: str,
    pool: OpponentPool,
    side: str,
    games: int = 10,
    winning_score: int = 15,
    max_eval: int = 20,
    simplify_observation: bool = False,
    executor: ProcessPoolExecutor | None = None,
) -> dict | None:
    """Play current model vs pool checkpoints to update PFP win-rates (parallel)."""
    if not pool.checkpoints:
        return None

    checkpoints = list(pool.checkpoints)
    if len(checkpoints) > max_eval:
        recent = checkpoints[-5:]
        rest = checkpoints[:-5]
        sampled = random.sample(rest, max_eval - 5)
        checkpoints = sampled + recent

    print(f"  [PFP] {side} pool update: {len(checkpoints)}/{len(pool.checkpoints)} checkpoints", flush=True)

    rng = np.random.default_rng()
    checkpoint_seeds = {path: int(rng.integers(0, 2**31)) for path in checkpoints}

    if executor is not None:
        futures = {}
        for path in checkpoints:
            f = executor.submit(
                _eval_checkpoint_worker,
                model_path,
                path,
                side,
                games,
                winning_score,
                simplify_observation,
                checkpoint_seeds[path],
            )
            futures[f] = path

        results: dict[str, list[bool]] = {}
        for f in as_completed(futures):
            name, wins = f.result()
            results[name] = wins
    else:
        results = {}
        for path in checkpoints:
            name, wins = _eval_checkpoint_worker(
                model_path,
                path,
                side,
                games,
                winning_score,
                simplify_observation,
                checkpoint_seeds[path],
            )
            results[name] = wins

    # Apply results to pool (main process only)
    win_rates = []
    for path in checkpoints:
        name = Path(path).name
        wins_list = results[name]
        for won in wins_list:
            pool.update_stats(name, won)

        n_wins = sum(wins_list)
        wr = pool.get_win_rate(name)
        win_rates.append(wr)
        weight = 1.0 - wr + 0.1
        print(f"    {name}: {n_wins}W {games - n_wins}L (wr={wr:.2f}, weight={weight:.2f})", flush=True)

    return {
        "avg_winrate": float(np.mean(win_rates)),
        "min_winrate": float(np.min(win_rates)),
        "pool_size": len(pool.checkpoints),
    }


def main() -> None:
    ensure_stack_size()
    parser = argparse.ArgumentParser(description="Cross-play training (PFP + builtin anchor)")
    parser.add_argument("--total-iterations", type=int, default=100)
    parser.add_argument("--steps-per-iter", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-level", "--noise_level", type=int, default=None, choices=[0, 1, 2, 3, 4, 5], help="Noise preset level"
    )
    parser.add_argument("--noise-x", type=int, default=None, help="Ball x position noise ±N pixels")
    parser.add_argument("--noise-x-vel", type=int, default=None, help="Ball x velocity noise ±N")
    parser.add_argument("--noise-y-vel", type=int, default=None, help="Ball y velocity noise ±N")
    parser.add_argument("--simplify-observation", action="store_true", help="Mirror player_2 x-axis observations")
    parser.add_argument("--anchor", default="builtin", help="Anchor opponent: builtin, stone, duckll, duckll:N")
    parser.add_argument("--anchor-prob", type=float, default=0.6, help="Probability of anchor opponent per iteration")
    parser.add_argument("--curriculum", default=None, help="Path to curriculum JSON file")
    parser.add_argument("--adaptive", default=None, help="Path to adaptive curriculum JSON file")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-score", type=int, default=5)
    parser.add_argument(
        "--eval-opponents",
        default="random,builtin",
        help="Comma-separated eval opponents (e.g. random,builtin,duckll:5)",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--p1-init", default=None)
    parser.add_argument("--p2-init", default=None)
    parser.add_argument("--pfp-eval-max", type=int, default=20)
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity (user or team)")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    meta = get_experiment_metadata()

    noise = parse_noise(args.noise_level, args.noise_x, args.noise_x_vel, args.noise_y_vel)

    def _make_anchor(spec: str) -> AIPolicy:
        if spec == "builtin":
            return BuiltinAI()
        elif spec == "stone":
            return StoneAI()
        elif spec == "duckll" or spec.startswith("duckll:"):
            preset = int(spec.split(":")[1]) if ":" in spec else None
            return DuckllAI(preset=preset) if preset is not None else DuckllAI()
        return BuiltinAI()

    anchor_policy = _make_anchor(args.anchor)
    anchor_name = args.anchor

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "script": "train_crossplay",
            "total_iterations": args.total_iterations,
            "steps_per_iter": args.steps_per_iter,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "anchor": args.anchor,
            "anchor_prob": args.anchor_prob,
            "ent_coef": args.ent_coef,
            "eval_freq": args.eval_freq,
            "eval_opponents": args.eval_opponents,
            "noise_level": args.noise_level,
            "noise_x": noise.x_range if noise else None,
            "noise_x_vel": noise.x_velocity_range if noise else None,
            "noise_y_vel": noise.y_velocity_range if noise else None,
            "simplify_observation": args.simplify_observation,
            "eval_games": args.eval_games,
            "save_dir": args.save_dir,
            **meta,
        },
    )

    # Load curriculum
    curriculum_schedule = None
    if args.curriculum:
        with open(args.curriculum) as f:
            curriculum_schedule = json.load(f)["schedule"]
        curriculum_schedule.sort(key=lambda x: x["iter"])

    adaptive_config = None
    if args.adaptive:
        with open(args.adaptive) as f:
            adaptive_config = json.load(f)

    p1_anchor_winrate = 0.0
    p2_anchor_winrate = 0.0

    def _interpolate_entry(a: dict, b: dict, t: float) -> tuple[float, float]:
        builtin = a["builtin"] + t * (b["builtin"] - a["builtin"])
        a_pool = a.get("latest", 0) + a.get("pool", 1.0 - a["builtin"])
        b_pool = b.get("latest", 0) + b.get("pool", 1.0 - b["builtin"])
        pool = a_pool + t * (b_pool - a_pool)
        return builtin, pool

    def _adaptive_probs(winrate: float) -> tuple[float, float]:
        thresholds = adaptive_config["thresholds"]
        if winrate <= thresholds[0]["winrate"]:
            t0 = thresholds[0]
            return t0["builtin"], t0.get("latest", 0) + t0.get("pool", 1.0 - t0["builtin"])
        if winrate >= thresholds[-1]["winrate"]:
            tn = thresholds[-1]
            return tn["builtin"], tn.get("latest", 0) + tn.get("pool", 1.0 - tn["builtin"])
        for i in range(len(thresholds) - 1):
            a, b = thresholds[i], thresholds[i + 1]
            if a["winrate"] <= winrate <= b["winrate"]:
                t = (winrate - a["winrate"]) / (b["winrate"] - a["winrate"])
                return _interpolate_entry(a, b, t)
        tn = thresholds[-1]
        return tn["builtin"], tn.get("latest", 0) + tn.get("pool", 1.0 - tn["builtin"])

    def get_probs(iteration: int, side: str = "p1") -> tuple[float, float]:
        nonlocal p1_anchor_winrate, p2_anchor_winrate
        if adaptive_config:
            wr = p1_anchor_winrate if side == "p1" else p2_anchor_winrate
            return _adaptive_probs(wr)
        if curriculum_schedule is None:
            return args.anchor_prob, 1.0 - args.anchor_prob
        if iteration <= curriculum_schedule[0]["iter"]:
            s = curriculum_schedule[0]
            return s["builtin"], s.get("latest", 0) + s.get("pool", 1.0 - s["builtin"])
        if iteration >= curriculum_schedule[-1]["iter"]:
            s = curriculum_schedule[-1]
            return s["builtin"], s.get("latest", 0) + s.get("pool", 1.0 - s["builtin"])
        for i in range(len(curriculum_schedule) - 1):
            a, b = curriculum_schedule[i], curriculum_schedule[i + 1]
            if a["iter"] <= iteration <= b["iter"]:
                t = (iteration - a["iter"]) / (b["iter"] - a["iter"])
                return _interpolate_entry(a, b, t)
        return args.anchor_prob, 1.0 - args.anchor_prob

    # Create envs (DummyVecEnv for opponent policy swapping)
    p1_envs = make_vec_env(
        n_envs=args.num_envs,
        agent="player_1",
        use_subproc=False,
        seed=args.seed,
        simplify_observation=args.simplify_observation,
        noise=noise,
    )
    p2_envs = make_vec_env(
        n_envs=args.num_envs,
        agent="player_2",
        use_subproc=False,
        seed=args.seed + 100,
        simplify_observation=args.simplify_observation,
        noise=noise,
    )

    # Model configs
    p1_cfg = ModelConfig(side="player_1", observation_simplified=args.simplify_observation)
    p2_cfg = ModelConfig(side="player_2", observation_simplified=args.simplify_observation)

    # Initialize models
    ppo_kwargs = dict(device="cpu", verbose=0, ent_coef=args.ent_coef)
    if args.p1_init:
        p1_model = PPO.load(args.p1_init, env=p1_envs, seed=args.seed, **ppo_kwargs)
        print(f"Loaded p1 from {args.p1_init}")
    else:
        p1_model = PPO("MlpPolicy", p1_envs, seed=args.seed, **ppo_kwargs)

    if args.p2_init:
        p2_model = PPO.load(args.p2_init, env=p2_envs, seed=args.seed + 1, **ppo_kwargs)
        print(f"Loaded p2 from {args.p2_init}")
    else:
        p2_model = PPO("MlpPolicy", p2_envs, seed=args.seed + 1, **ppo_kwargs)

    # Opponent pools
    pool_p1 = OpponentPool(str(save_dir / "p1"), "p1", anchor=anchor_policy)
    pool_p2 = OpponentPool(str(save_dir / "p2"), "p2", anchor=anchor_policy)

    eval_workers = os.cpu_count()
    mp_context = multiprocessing.get_context("forkserver")
    eval_executor = ProcessPoolExecutor(max_workers=eval_workers, mp_context=mp_context, initializer=worker_init)

    print(f"Self-play training: {args.total_iterations} iterations x {args.steps_per_iter} steps")
    print(f"Envs: {args.num_envs} (DummyVecEnv), Eval workers: {eval_workers}")
    if adaptive_config:
        first, last = adaptive_config["thresholds"][0], adaptive_config["thresholds"][-1]
        print(
            f"Adaptive curriculum: anchor({anchor_name}) {first['builtin'] * 100:.0f}%"
            f"->{last['builtin'] * 100:.0f}% based on win rate"
        )
    elif curriculum_schedule:
        first, last = curriculum_schedule[0], curriculum_schedule[-1]
        print(f"Curriculum: anchor({anchor_name}) {first['builtin'] * 100:.0f}%->{last['builtin'] * 100:.0f}%")
    else:
        print(f"Opponent mix: {anchor_name}={args.anchor_prob}, pool(PFP)={1.0 - args.anchor_prob:.1f}")

    best_p1_anchor = -1.0
    best_p2_anchor = -1.0

    setup_graceful_shutdown()

    try:
        for iteration in range(args.total_iterations):
            step = p1_model.num_timesteps

            # --- Evaluate ---
            if iteration % args.eval_freq == 0:
                p1_latest_dir = save_model(p1_model, save_dir / "p1" / "crossplay_latest", p1_cfg)
                p2_latest_dir = save_model(p2_model, save_dir / "p2" / "crossplay_latest", p2_cfg)
                _log_model_artifact(run, "p1-latest", str(p1_latest_dir))
                _log_model_artifact(run, "p2-latest", str(p2_latest_dir))
                eval_opps = [s.strip() for s in args.eval_opponents.split(",")]
                matchups = evaluate_crossplay_detailed(
                    str(p1_latest_dir),
                    str(p2_latest_dir),
                    games=args.eval_games,
                    winning_score=args.eval_score,
                    simplify_observation=args.simplify_observation,
                    eval_opponents=eval_opps,
                )

                print(f"\n[Iter {iteration}/{args.total_iterations}, p1_step={step}]", flush=True)

                log_data: dict = {"iteration": iteration}
                for match, s in matchups.items():
                    print(
                        f"  {match}: {s['wins']}W {s['losses']}L ({s['win_rate'] * 100:.0f}%)"
                        f"  score: {s['avg_score']:.1f}-{s['avg_opp_score']:.1f}"
                        f"  serve: {s['serve_win_rate'] * 100:.0f}% receive: {s['receive_win_rate'] * 100:.0f}%"
                        f"  round: {s['avg_round_frames']:.0f}f",
                        flush=True,
                    )

                # Build per-side eval results for log_data
                p1_results = {}
                p2_results = {}
                for match, s in matchups.items():
                    if match.startswith("p1_vs_"):
                        p1_results[match[len("p1_vs_") :]] = s
                    if match.startswith("p2_vs_"):
                        p2_results[match[len("p2_vs_") :]] = s
                    if match == "p1_vs_p2":
                        log_data["p2/eval/vs_p1/win_rate"] = 1.0 - s["win_rate"]
                        log_data["p2/eval/vs_p1/avg_score"] = s["avg_opp_score"]
                        log_data["p2/eval/vs_p1/avg_round_frames"] = s["avg_round_frames"]
                log_data.update(build_eval_log_data(p1_results, "p1/eval"))
                log_data.update(build_eval_log_data(p2_results, "p2/eval"))

                # Compute ELO for p1 and p2
                for side_label in ["p1", "p2"]:
                    model_name = f"__{side_label}__"
                    win_counts: dict[tuple[str, str], tuple[int, int]] = {}
                    for match, s in matchups.items():
                        if match.startswith(f"{side_label}_vs_") and match != "p1_vs_p2":
                            opp_name = match[len(f"{side_label}_vs_") :]
                            win_counts[(model_name, opp_name)] = (s["wins"], s["losses"])
                    elos = compute_elo(win_counts)
                    log_data[f"{side_label}/eval/elo"] = elos.get(model_name, 1500.0)

                # PFP pool stats update
                p1_pfp = _update_pool_stats(
                    str(p1_latest_dir),
                    pool_p2,
                    side="p1",
                    games=args.eval_games,
                    winning_score=args.eval_score,
                    max_eval=args.pfp_eval_max,
                    simplify_observation=args.simplify_observation,
                    executor=eval_executor,
                )
                p2_pfp = _update_pool_stats(
                    str(p2_latest_dir),
                    pool_p1,
                    side="p2",
                    games=args.eval_games,
                    winning_score=args.eval_score,
                    max_eval=args.pfp_eval_max,
                    simplify_observation=args.simplify_observation,
                    executor=eval_executor,
                )
                if p1_pfp:
                    log_data["p1/pfp/avg_pool_win_rate"] = p1_pfp["avg_winrate"]
                    log_data["p1/pfp/min_win_rate"] = p1_pfp["min_winrate"]
                    log_data["p1/pfp/pool_size"] = p1_pfp["pool_size"]
                if p2_pfp:
                    log_data["p2/pfp/avg_pool_win_rate"] = p2_pfp["avg_winrate"]
                    log_data["p2/pfp/min_win_rate"] = p2_pfp["min_winrate"]
                    log_data["p2/pfp/pool_size"] = p2_pfp["pool_size"]

                # Adaptive curriculum update
                p1_wr = matchups.get(f"p1_vs_{anchor_name}", {}).get("win_rate", 0)
                p2_wr = matchups.get(f"p2_vs_{anchor_name}", {}).get("win_rate", 0)
                if adaptive_config:
                    p1_anchor_winrate = p1_wr
                    p2_anchor_winrate = p2_wr
                    p1_bp, p1_pp = get_probs(iteration, side="p1")
                    p2_bp, p2_pp = get_probs(iteration, side="p2")
                    print(
                        f"  [ADAPTIVE] p1: wr={p1_wr * 100:.0f}% -> anchor={p1_bp * 100:.0f}% pool={p1_pp * 100:.0f}%"
                        f"  |  p2: wr={p2_wr * 100:.0f}% -> anchor={p2_bp * 100:.0f}% pool={p2_pp * 100:.0f}%",
                        flush=True,
                    )

                # Save best models
                if p1_wr > best_p1_anchor:
                    best_p1_anchor = p1_wr
                    p1_best_dir = save_model(p1_model, save_dir / "p1" / "crossplay_best", p1_cfg)
                    _log_model_artifact(run, "p1-best", str(p1_best_dir))
                    print(f"  [BEST] p1 vs {anchor_name}: {p1_wr * 100:.0f}% (iter {iteration})", flush=True)
                if p2_wr > best_p2_anchor:
                    best_p2_anchor = p2_wr
                    p2_best_dir = save_model(p2_model, save_dir / "p2" / "crossplay_best", p2_cfg)
                    _log_model_artifact(run, "p2-best", str(p2_best_dir))
                    print(f"  [BEST] p2 vs {anchor_name}: {p2_wr * 100:.0f}% (iter {iteration})", flush=True)

                run.log(log_data, step=step)

            # --- Train ---
            p1_anchor_prob, p1_pool_prob = get_probs(iteration, side="p1")
            p2_anchor_prob, p2_pool_prob = get_probs(iteration, side="p2")

            run.log(
                {
                    "p1/curriculum/anchor_prob": p1_anchor_prob,
                    "p1/curriculum/pool_prob": p1_pool_prob,
                    "p2/curriculum/anchor_prob": p2_anchor_prob,
                    "p2/curriculum/pool_prob": p2_pool_prob,
                },
                step=step,
            )

            # Save to pool
            if iteration % args.save_interval == 0:
                p1_path = pool_p1.add_checkpoint(p1_model, iteration)
                p2_path = pool_p2.add_checkpoint(p2_model, iteration)
                _log_model_artifact(run, f"p1-pool-iter{iteration:06d}", p1_path)
                _log_model_artifact(run, f"p2-pool-iter{iteration:06d}", p2_path)

            # Train p1 against p2 opponent
            opp, opp_name, is_anchor = pool_p2.sample_opponent(latest_model=p2_model, anchor_prob=p1_anchor_prob)
            if is_anchor:
                for env in p1_envs.envs:
                    set_opponent_policy(env, anchor_policy)
            else:
                policy = make_opponent_policy(opp)
                for env in p1_envs.envs:
                    set_opponent_policy(env, policy)
            print(
                f"  [iter {iteration}] p1 vs {opp_name} | anchor={p1_anchor_prob:.0%} pool={p1_pool_prob:.0%}",
                flush=True,
            )
            p1_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)
            _log_sb3_metrics(run, p1_model, "p1")

            # Train p2 against p1 opponent
            opp, opp_name, is_anchor = pool_p1.sample_opponent(latest_model=p1_model, anchor_prob=p2_anchor_prob)
            if is_anchor:
                for env in p2_envs.envs:
                    set_opponent_policy(env, anchor_policy)
            else:
                policy = make_opponent_policy(opp)
                for env in p2_envs.envs:
                    set_opponent_policy(env, policy)
            print(
                f"  [iter {iteration}] p2 vs {opp_name} | anchor={p2_anchor_prob:.0%} pool={p2_pool_prob:.0%}",
                flush=True,
            )
            p2_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)
            _log_sb3_metrics(run, p2_model, "p2")

        # Save final models
        p1_final_dir = save_model(p1_model, save_dir / "p1" / "crossplay_final", p1_cfg)
        p2_final_dir = save_model(p2_model, save_dir / "p2" / "crossplay_final", p2_cfg)
        _log_model_artifact(run, "p1-final", str(p1_final_dir))
        _log_model_artifact(run, "p2-final", str(p2_final_dir))

        # Record sample videos
        from pika_zoo.scripts.play import play

        p1_final_zip = str(p1_final_dir / "model.zip")
        p2_final_zip = str(p2_final_dir / "model.zip")
        for side, model_zip in [("player_1", p1_final_zip), ("player_2", p2_final_zip)]:
            label = "p1" if side == "player_1" else "p2"
            for opp in eval_opps:
                video_path = str(save_dir / f"{label}_vs_{opp}.mp4")
                record_video(model_zip, side, opp, video_path)
                run.log({f"video/{label}_vs_{opp}": wandb.Video(video_path, fps=25, format="mp4")})

        # p1 vs p2
        p1v2_path = str(save_dir / "p1_vs_p2.mp4")
        play(p1=p1_final_zip, p2=p2_final_zip, winning_score=5, render=False, record=p1v2_path, seed=0)
        run.log({"video/p1_vs_p2": wandb.Video(p1v2_path, fps=25, format="mp4")})

        print(f"\nTraining complete. Models saved to {save_dir}/p1/ and {save_dir}/p2/")
    finally:
        shutdown_executor(eval_executor)
        p1_envs.close()
        p2_envs.close()
        run.finish()


if __name__ == "__main__":
    main()
