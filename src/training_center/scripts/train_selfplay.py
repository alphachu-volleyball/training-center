"""Self-play training script (PFSP + builtin anchor).

Alternately trains p1_model (left) and p2_model (right).
Opponent mix: builtin_prob (rule AI) + remaining (PFSP pool).

Usage:
  uv run train-selfplay --total-iterations 100 --steps-per-iter 20000 --save-dir experiments/001
  uv run train-selfplay --p1-init exp/001/p1 --p2-init exp/001/p2 --save-dir experiments/002
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from pika_zoo.ai import BuiltinAI
from stable_baselines3 import PPO

import wandb
from training_center.env_factory import make_vec_env, set_opponent_policy
from training_center.eval.match import Player, play_game
from training_center.eval.opponent_pool import OpponentPool, make_opponent_policy
from training_center.metadata import get_experiment_metadata


def _log_sb3_metrics(run: wandb.sdk.wandb_run.Run, model: PPO, prefix: str) -> None:
    """Read SB3 logger metrics and log to wandb with a prefix."""
    if model.logger is not None and hasattr(model.logger, "name_to_value"):
        metrics = {f"{prefix}/{k}": v for k, v in model.logger.name_to_value.items()}
        if metrics:
            run.log(metrics, step=model.num_timesteps)


def _log_model_artifact(run: wandb.sdk.wandb_run.Run, name: str, path: str) -> None:
    """Log a saved model as a wandb artifact."""
    artifact = wandb.Artifact(name, type="model")
    artifact.add_file(path + ".zip")
    run.log_artifact(artifact)


def _record_video(model_path: str, side: str, opponent: str, output_path: str) -> None:
    """Record a sample game video using pika-zoo's play script."""
    from pika_zoo.scripts.play import play

    p1 = model_path if side == "player_1" else opponent
    p2 = opponent if side == "player_1" else model_path
    play(p1=p1, p2=p2, winning_score=5, render=False, record=output_path, seed=0)


def evaluate_selfplay_detailed(
    p1_model: PPO,
    p2_model: PPO,
    games: int = 20,
    winning_score: int = 15,
    seed: int = 42,
) -> dict[str, dict]:
    """Evaluate with detailed stats across 5 matchups."""
    rng = np.random.default_rng(seed)
    p1 = Player("p1", "model", model=p1_model)
    p2 = Player("p2", "model", model=p2_model)
    random_p = Player("random", "random")
    builtin_p = Player("builtin", "builtin")

    matchups: dict[str, dict] = {}
    for name, p1_player, p2_player, perspective in [
        ("p1_vs_p2", p1, p2, "p1"),
        ("p1_vs_random", p1, random_p, "p1"),
        ("p1_vs_builtin", p1, builtin_p, "p1"),
        ("p2_vs_random", random_p, p2, "p2"),
        ("p2_vs_builtin", builtin_p, p2, "p2"),
    ]:
        matchup_seed = int(rng.integers(0, 2**31))
        name, summary = _run_matchup(name, p1_player, p2_player, games, winning_score, perspective, matchup_seed)
        matchups[name] = summary

    return matchups


def _run_matchup(
    name: str,
    p1_player: Player,
    p2_player: Player,
    games: int,
    winning_score: int,
    perspective: str,
    seed: int,
) -> tuple[str, dict]:
    """Run a single matchup evaluation."""
    rng = np.random.default_rng(seed)
    rounds_all = []
    all_stats = []
    wins = 0

    for _i in range(games):
        game_seed = int(rng.integers(0, 2**31))
        episode = play_game(p1_player, p2_player, winning_score=winning_score, seed=game_seed)
        all_stats.append(episode)
        if perspective == "p1":
            wins += 1 if episode.winner == "player_1" else 0
        else:
            wins += 1 if episode.winner == "player_2" else 0
        rounds_all.extend(episode.rounds)

    summary = _summarize(wins, games, rounds_all, all_stats, perspective)
    return name, summary


def _summarize(
    wins: int,
    games: int,
    rounds: list,
    all_stats: list,
    perspective: str,
) -> dict:
    """Aggregate match statistics."""
    p1_serve = [r for r in rounds if r.server == "player_1"]
    p2_serve = [r for r in rounds if r.server == "player_2"]
    durations = [r.duration for r in rounds]

    if perspective == "p1":
        avg_score = float(np.mean([e.scores[0] for e in all_stats])) if all_stats else 0
        avg_opp_score = float(np.mean([e.scores[1] for e in all_stats])) if all_stats else 0
    else:
        avg_score = float(np.mean([e.scores[1] for e in all_stats])) if all_stats else 0
        avg_opp_score = float(np.mean([e.scores[0] for e in all_stats])) if all_stats else 0

    return {
        "wins": wins,
        "losses": games - wins,
        "win_rate": wins / games,
        "avg_score": avg_score,
        "avg_opp_score": avg_opp_score,
        "p1_serve_win": sum(1 for r in p1_serve if r.scorer == "player_1") / max(len(p1_serve), 1),
        "p2_serve_win": sum(1 for r in p2_serve if r.scorer == "player_2") / max(len(p2_serve), 1),
        "avg_rally": float(np.mean(durations)) if durations else 0,
    }


def _update_pool_stats(
    model: PPO,
    pool: OpponentPool,
    side: str,
    games: int = 10,
    winning_score: int = 15,
    max_eval: int = 20,
) -> dict | None:
    """Play current model vs pool checkpoints to update PFSP win-rates."""
    if not pool.checkpoints:
        return None

    current_player = Player(side, "model", model=model)

    checkpoints = list(pool.checkpoints)
    if len(checkpoints) > max_eval:
        recent = checkpoints[-5:]
        rest = checkpoints[:-5]
        sampled = random.sample(rest, max_eval - 5)
        checkpoints = sampled + recent

    print(f"  [PFSP] {side} pool update: {len(checkpoints)}/{len(pool.checkpoints)} checkpoints", flush=True)

    win_rates = []
    rng = np.random.default_rng()
    for path in checkpoints:
        name = Path(path).name
        opp_model = PPO.load(path, device="cpu")
        opp_player = Player(name, "model", model=opp_model)

        wins = 0
        for _ in range(games):
            game_seed = int(rng.integers(0, 2**31))
            if side == "p1":
                stats = play_game(current_player, opp_player, winning_score=winning_score, seed=game_seed)
                won = stats.winner == "player_1"
            else:
                stats = play_game(opp_player, current_player, winning_score=winning_score, seed=game_seed)
                won = stats.winner == "player_2"
            pool.update_stats(name, won)
            if won:
                wins += 1

        wr = pool.get_win_rate(name)
        win_rates.append(wr)
        weight = 1.0 - wr + 0.1
        print(f"    {name}: {wins}W {games - wins}L (wr={wr:.2f}, weight={weight:.2f})", flush=True)

    return {
        "avg_winrate": float(np.mean(win_rates)),
        "min_winrate": float(np.min(win_rates)),
        "pool_size": len(pool.checkpoints),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play training (PFSP + builtin anchor)")
    parser.add_argument("--total-iterations", type=int, default=100)
    parser.add_argument("--steps-per-iter", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noisy", action="store_true", help="Add random perturbation to ball initial state")
    parser.add_argument("--builtin-prob", type=float, default=0.6)
    parser.add_argument("--curriculum", default=None, help="Path to curriculum JSON file")
    parser.add_argument("--adaptive", default=None, help="Path to adaptive curriculum JSON file")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-score", type=int, default=5)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--p1-init", default=None)
    parser.add_argument("--p2-init", default=None)
    parser.add_argument("--pfsp-eval-max", type=int, default=20)
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity (user or team)")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    meta = get_experiment_metadata()

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "script": "train_selfplay",
            "total_iterations": args.total_iterations,
            "steps_per_iter": args.steps_per_iter,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "builtin_prob": args.builtin_prob,
            "ent_coef": args.ent_coef,
            "eval_freq": args.eval_freq,
            "noisy": args.noisy,
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

    p1_builtin_winrate = 0.0
    p2_builtin_winrate = 0.0

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
        nonlocal p1_builtin_winrate, p2_builtin_winrate
        if adaptive_config:
            wr = p1_builtin_winrate if side == "p1" else p2_builtin_winrate
            return _adaptive_probs(wr)
        if curriculum_schedule is None:
            return args.builtin_prob, 1.0 - args.builtin_prob
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
        return args.builtin_prob, 1.0 - args.builtin_prob

    # Create envs (DummyVecEnv for opponent policy swapping)
    p1_envs = make_vec_env(n_envs=args.num_envs, agent="player_1", use_subproc=False, seed=args.seed, noisy=args.noisy)
    p2_envs = make_vec_env(
        n_envs=args.num_envs, agent="player_2", use_subproc=False, seed=args.seed + 100, noisy=args.noisy
    )

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
    pool_p1 = OpponentPool(str(save_dir / "p1"), "p1")
    pool_p2 = OpponentPool(str(save_dir / "p2"), "p2")

    print(f"Self-play training: {args.total_iterations} iterations x {args.steps_per_iter} steps")
    print(f"Envs: {args.num_envs} (DummyVecEnv)")
    if adaptive_config:
        first, last = adaptive_config["thresholds"][0], adaptive_config["thresholds"][-1]
        print(
            f"Adaptive curriculum: builtin {first['builtin'] * 100:.0f}%"
            f"->{last['builtin'] * 100:.0f}% based on win rate"
        )
    elif curriculum_schedule:
        first, last = curriculum_schedule[0], curriculum_schedule[-1]
        print(f"Curriculum: builtin {first['builtin'] * 100:.0f}%->{last['builtin'] * 100:.0f}%")
    else:
        print(f"Opponent mix: builtin={args.builtin_prob}, pool(PFSP)={1.0 - args.builtin_prob:.1f}")

    best_p1_builtin = -1.0
    best_p2_builtin = -1.0

    for iteration in range(args.total_iterations):
        step = p1_model.num_timesteps

        # --- Evaluate ---
        if iteration % args.eval_freq == 0:
            p1_latest = str(save_dir / "p1" / "selfplay_latest")
            p2_latest = str(save_dir / "p2" / "selfplay_latest")
            p1_model.save(p1_latest)
            p2_model.save(p2_latest)
            _log_model_artifact(run, "p1-latest", p1_latest)
            _log_model_artifact(run, "p2-latest", p2_latest)
            matchups = evaluate_selfplay_detailed(
                p1_model, p2_model, games=args.eval_games, winning_score=args.eval_score
            )

            print(f"\n[Iter {iteration}/{args.total_iterations}, p1_step={step}]", flush=True)

            log_data: dict = {"iteration": iteration}
            for match, s in matchups.items():
                print(
                    f"  {match}: {s['wins']}W {s['losses']}L ({s['win_rate'] * 100:.0f}%)"
                    f"  score: {s['avg_score']:.1f}-{s['avg_opp_score']:.1f}"
                    f"  serve: p1={s['p1_serve_win'] * 100:.0f}% p2={s['p2_serve_win'] * 100:.0f}%"
                    f"  rally: {s['avg_rally']:.0f}",
                    flush=True,
                )

                if match.startswith("p1_vs_"):
                    opponent = match[len("p1_vs_") :]
                    log_data[f"p1/eval/vs_{opponent}_winrate"] = s["win_rate"]
                    log_data[f"p1/eval/vs_{opponent}_avg_score"] = s["avg_score"]
                    log_data[f"p1/eval/vs_{opponent}_avg_rally"] = s["avg_rally"]

                if match.startswith("p2_vs_"):
                    opponent = match[len("p2_vs_") :]
                    log_data[f"p2/eval/vs_{opponent}_winrate"] = s["win_rate"]
                    log_data[f"p2/eval/vs_{opponent}_avg_score"] = s["avg_score"]
                    log_data[f"p2/eval/vs_{opponent}_avg_rally"] = s["avg_rally"]

                if match == "p1_vs_p2":
                    log_data["p2/eval/vs_p1_winrate"] = 1.0 - s["win_rate"]
                    log_data["p2/eval/vs_p1_avg_score"] = s["avg_opp_score"]
                    log_data["p2/eval/vs_p1_avg_rally"] = s["avg_rally"]

            # PFSP pool stats update
            p1_pfsp = _update_pool_stats(
                p1_model,
                pool_p2,
                side="p1",
                games=args.eval_games,
                winning_score=args.eval_score,
                max_eval=args.pfsp_eval_max,
            )
            p2_pfsp = _update_pool_stats(
                p2_model,
                pool_p1,
                side="p2",
                games=args.eval_games,
                winning_score=args.eval_score,
                max_eval=args.pfsp_eval_max,
            )
            if p1_pfsp:
                log_data["p1/pfsp/avg_pool_winrate"] = p1_pfsp["avg_winrate"]
                log_data["p1/pfsp/min_winrate"] = p1_pfsp["min_winrate"]
                log_data["p1/pfsp/pool_size"] = p1_pfsp["pool_size"]
            if p2_pfsp:
                log_data["p2/pfsp/avg_pool_winrate"] = p2_pfsp["avg_winrate"]
                log_data["p2/pfsp/min_winrate"] = p2_pfsp["min_winrate"]
                log_data["p2/pfsp/pool_size"] = p2_pfsp["pool_size"]

            # Adaptive curriculum update
            p1_wr = matchups.get("p1_vs_builtin", {}).get("win_rate", 0)
            p2_wr = matchups.get("p2_vs_builtin", {}).get("win_rate", 0)
            if adaptive_config:
                p1_builtin_winrate = p1_wr
                p2_builtin_winrate = p2_wr
                p1_bp, p1_pp = get_probs(iteration, side="p1")
                p2_bp, p2_pp = get_probs(iteration, side="p2")
                print(
                    f"  [ADAPTIVE] p1: wr={p1_wr * 100:.0f}% -> builtin={p1_bp * 100:.0f}% pool={p1_pp * 100:.0f}%"
                    f"  |  p2: wr={p2_wr * 100:.0f}% -> builtin={p2_bp * 100:.0f}% pool={p2_pp * 100:.0f}%",
                    flush=True,
                )

            # Save best models
            if p1_wr > best_p1_builtin:
                best_p1_builtin = p1_wr
                p1_best = str(save_dir / "p1" / "selfplay_best")
                p1_model.save(p1_best)
                _log_model_artifact(run, "p1-best", p1_best)
                print(f"  [BEST] p1 vs builtin: {p1_wr * 100:.0f}% (iter {iteration})", flush=True)
            if p2_wr > best_p2_builtin:
                best_p2_builtin = p2_wr
                p2_best = str(save_dir / "p2" / "selfplay_best")
                p2_model.save(p2_best)
                _log_model_artifact(run, "p2-best", p2_best)
                print(f"  [BEST] p2 vs builtin: {p2_wr * 100:.0f}% (iter {iteration})", flush=True)

            run.log(log_data, step=step)

        # --- Train ---
        p1_builtin_prob, p1_pool_prob = get_probs(iteration, side="p1")
        p2_builtin_prob, p2_pool_prob = get_probs(iteration, side="p2")

        run.log(
            {
                "p1/curriculum/builtin_prob": p1_builtin_prob,
                "p1/curriculum/pool_prob": p1_pool_prob,
                "p2/curriculum/builtin_prob": p2_builtin_prob,
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
        opp, opp_name, is_builtin = pool_p2.sample_opponent(latest_model=p2_model, builtin_prob=p1_builtin_prob)
        if is_builtin:
            for env in p1_envs.envs:
                set_opponent_policy(env, BuiltinAI())
        else:
            policy = make_opponent_policy(opp)
            for env in p1_envs.envs:
                set_opponent_policy(env, policy)
        print(
            f"  [iter {iteration}] p1 vs {opp_name} | builtin={p1_builtin_prob:.0%} pool={p1_pool_prob:.0%}",
            flush=True,
        )
        p1_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)
        _log_sb3_metrics(run, p1_model, "p1")

        # Train p2 against p1 opponent
        opp, opp_name, is_builtin = pool_p1.sample_opponent(latest_model=p1_model, builtin_prob=p2_builtin_prob)
        if is_builtin:
            for env in p2_envs.envs:
                set_opponent_policy(env, BuiltinAI())
        else:
            policy = make_opponent_policy(opp)
            for env in p2_envs.envs:
                set_opponent_policy(env, policy)
        print(
            f"  [iter {iteration}] p2 vs {opp_name} | builtin={p2_builtin_prob:.0%} pool={p2_pool_prob:.0%}",
            flush=True,
        )
        p2_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)
        _log_sb3_metrics(run, p2_model, "p2")

    # Save final models
    p1_final = str(save_dir / "p1" / "selfplay_final")
    p2_final = str(save_dir / "p2" / "selfplay_final")
    p1_model.save(p1_final)
    p2_model.save(p2_final)
    _log_model_artifact(run, "p1-final", p1_final)
    _log_model_artifact(run, "p2-final", p2_final)

    # Record sample videos
    from pika_zoo.scripts.play import play

    for side, model_path in [("player_1", p1_final), ("player_2", p2_final)]:
        label = "p1" if side == "player_1" else "p2"
        for opp in ["builtin", "random"]:
            video_path = str(save_dir / f"{label}_vs_{opp}.mp4")
            _record_video(model_path + ".zip", side, opp, video_path)
            run.log({f"video/{label}_vs_{opp}": wandb.Video(video_path, fps=25, format="mp4")})

    # p1 vs p2
    p1v2_path = str(save_dir / "p1_vs_p2.mp4")
    play(p1=p1_final + ".zip", p2=p2_final + ".zip", winning_score=5, render=False, record=p1v2_path, seed=0)
    run.log({"video/p1_vs_p2": wandb.Video(p1v2_path, fps=25, format="mp4")})

    print(f"\nTraining complete. Models saved to {save_dir}/p1/ and {save_dir}/p2/")

    p1_envs.close()
    p2_envs.close()
    run.finish()


if __name__ == "__main__":
    main()
