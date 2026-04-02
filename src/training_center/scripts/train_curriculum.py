"""Curriculum training: single-side PPO against progressively harder rule-based AIs.

Starts with easy opponents, unlocks harder ones as the model improves.
Uses PFSP sampling within the unlocked pool.

Usage:
  uv run train-curriculum --save-dir experiments/010 --total-iterations 200
  uv run train-curriculum --save-dir experiments/010 --unlock-threshold 0.8 --initial-unlocked 3
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import wandb
from pika_zoo.ai import BuiltinAI, DuckllAI, RandomAI, StoneAI
from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.env.pikachu_volleyball import NoiseConfig
from pika_zoo.records.types import GamesRecord
from stable_baselines3 import PPO

from training_center.curriculum_pool import CurriculumPool
from training_center.env_factory import ensure_stack_size, make_vec_env, set_opponent_policy
from training_center.game import make_player, play_game
from training_center.metadata import get_experiment_metadata
from training_center.metrics import compute_eval_metrics
from training_center.model_config import ModelConfig, save_model

# ELO ladder from experiment 009 (ascending difficulty)
CURRICULUM_LADDER = [
    "stone",
    "random",
    "duckll:1",
    "builtin",
    "duckll:2",
    "duckll:0",
    "duckll:3",
    "duckll:7",
    "duckll:5",
    "duckll:4",
    "duckll:6",
    "duckll:8",
    "duckll:9",
    "duckll:10",
]

NOISE_LEVELS = {
    0: (0, 0, 0),
    1: (5, 3, 1),
    2: (10, 5, 2),
    3: (20, 10, 3),
    4: (35, 15, 4),
    5: (50, 20, 5),
}


def _make_opponent(spec: str) -> AIPolicy:
    """Instantiate an AI policy from a spec string."""
    if spec == "builtin":
        return BuiltinAI()
    elif spec == "stone":
        return StoneAI()
    elif spec == "duckll" or spec.startswith("duckll:"):
        preset = int(spec.split(":")[1]) if ":" in spec else None
        return DuckllAI(preset=preset) if preset is not None else DuckllAI()
    return RandomAI()


def _eval_matchup_worker(
    model_path: str,
    model_side: str,
    opp_name: str,
    games: int,
    winning_score: int,
    simplify_observation: bool,
    seed: int,
) -> tuple[str, dict]:
    """Worker: evaluate model vs one opponent. Returns (opp_name, result_dict)."""
    model_player = make_player(model_path, agent=model_side, simplify_observation=simplify_observation)
    opp_player = make_player(
        opp_name,
        agent="player_2" if model_side == "player_1" else "player_1",
        simplify_observation=simplify_observation,
    )
    rng = np.random.default_rng(seed)
    all_episodes = []

    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        if model_side == "player_1":
            episode = play_game(model_player, opp_player, winning_score=winning_score, seed=game_seed)
        else:
            episode = play_game(opp_player, model_player, winning_score=winning_score, seed=game_seed)
        all_episodes.append(episode)

    model_idx = 0 if model_side == "player_1" else 1
    wins = sum(1 for e in all_episodes if e.winner == model_side)
    detail = compute_eval_metrics(GamesRecord(games=all_episodes), model_side)

    return opp_name, {
        "wins": wins,
        "losses": games - wins,
        "win_rate": wins / games,
        "avg_score": float(np.mean([e.scores[model_idx] for e in all_episodes])),
        "game_winners": [e.winner for e in all_episodes],
        **detail,
    }


def _worker_init() -> None:
    """Ignore SIGINT in worker processes so only the main process handles it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _record_video(model_path: str, side: str, opponent: str, output_path: str) -> None:
    """Record a sample game video."""
    from pika_zoo.scripts.play import play

    p1 = model_path if side == "player_1" else opponent
    p2 = opponent if side == "player_1" else model_path
    play(p1=p1, p2=p2, winning_score=5, render=False, record=output_path, seed=0)


def main() -> None:
    ensure_stack_size()

    ladder_str = " → ".join(CURRICULUM_LADDER)
    parser = argparse.ArgumentParser(
        description="Curriculum training: progressive difficulty",
        epilog=(f"Difficulty ladder (ELO order):\n  {ladder_str}\n  Source: https://api.wandb.ai/links/ootzk/8za7h3er"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--total-iterations", type=int, default=50)
    parser.add_argument("--steps-per-iter", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--side", default="player_1", choices=["player_1", "player_2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-level", "--noise_level", type=int, default=None, choices=[0, 1, 2, 3, 4, 5], help="Noise preset level"
    )
    parser.add_argument("--noise-x", type=int, default=None, help="Ball x position noise ±N pixels")
    parser.add_argument("--noise-x-vel", type=int, default=None, help="Ball x velocity noise ±N")
    parser.add_argument("--noise-y-vel", type=int, default=None, help="Ball y velocity noise ±N")
    parser.add_argument("--simplify-observation", action="store_true", help="Mirror player_2 x-axis observations")
    parser.add_argument("--unlock-threshold", type=float, default=0.75, help="Min win rate to unlock next opponent")
    parser.add_argument("--initial-unlocked", type=int, default=3, help="Number of opponents unlocked at start")
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluate every N iterations")
    parser.add_argument("--eval-games", type=int, default=10, help="Games per opponent per evaluation")
    parser.add_argument("--eval-score", type=int, default=5, help="Winning score for eval games")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--init-model", default=None, help="Pretrained model path to resume from")
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    meta = get_experiment_metadata()

    # Noise config
    noise = None
    if args.noise_level is not None and args.noise_level > 0:
        x, xv, yv = NOISE_LEVELS[args.noise_level]
        noise = NoiseConfig(x_range=x, x_velocity_range=xv, y_velocity_range=yv)
    elif args.noise_x is not None or args.noise_x_vel is not None or args.noise_y_vel is not None:
        noise = NoiseConfig(
            x_range=args.noise_x or 0,
            x_velocity_range=args.noise_x_vel or 0,
            y_velocity_range=args.noise_y_vel or 0,
        )

    # Curriculum pool
    pool = CurriculumPool(CURRICULUM_LADDER, unlock_threshold=args.unlock_threshold)
    for i in range(min(args.initial_unlocked, len(CURRICULUM_LADDER))):
        pool.force_unlock(i)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "script": "train_curriculum",
            "total_iterations": args.total_iterations,
            "steps_per_iter": args.steps_per_iter,
            "num_envs": args.num_envs,
            "side": args.side,
            "seed": args.seed,
            "noise_level": args.noise_level,
            "unlock_threshold": args.unlock_threshold,
            "initial_unlocked": args.initial_unlocked,
            "eval_freq": args.eval_freq,
            "ent_coef": args.ent_coef,
            "ladder": CURRICULUM_LADDER,
            **meta,
        },
    )

    # Create DummyVecEnv (opponent swapping requires in-process access)
    envs = make_vec_env(
        n_envs=args.num_envs,
        agent=args.side,
        use_subproc=False,
        seed=args.seed,
        simplify_observation=args.simplify_observation,
        noise=noise,
    )

    # Initialize model
    model_cfg = ModelConfig(side=args.side, observation_simplified=args.simplify_observation)
    ppo_kwargs = dict(device="cpu", verbose=0, ent_coef=args.ent_coef)
    if args.init_model:
        model = PPO.load(args.init_model, env=envs, seed=args.seed, **ppo_kwargs)
        print(f"Loaded from {args.init_model}")
    else:
        model = PPO("MlpPolicy", envs, seed=args.seed, **ppo_kwargs)

    mp_context = multiprocessing.get_context("forkserver")
    eval_executor = ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=mp_context, initializer=_worker_init)

    print(f"Curriculum training: {args.total_iterations} iterations x {args.steps_per_iter} steps")
    print(f"Envs: {args.num_envs} (DummyVecEnv)")
    print(f"Unlock threshold: {args.unlock_threshold:.0%}")
    print(f"Initial pool: {pool.unlocked}")
    print(f"Ladder: {CURRICULUM_LADDER}")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    signal.signal(signal.SIGINT, lambda *_: os.kill(os.getpid(), signal.SIGTERM))

    try:
        for iteration in range(args.total_iterations):
            step = model.num_timesteps

            # --- EVALUATE ---
            if iteration % args.eval_freq == 0:
                ckpt_dir = save_model(model, save_dir / f"checkpoint_{iteration:06d}", model_cfg)
                model_path = str(ckpt_dir / "model.zip")

                artifact = wandb.Artifact(f"curriculum-checkpoint-{iteration:06d}", type="model")
                artifact.add_dir(str(ckpt_dir))
                run.log_artifact(artifact)

                # Evaluate vs all unlocked opponents (parallel)
                rng = np.random.default_rng()
                futures = {}
                for opp in pool.unlocked:
                    seed = int(rng.integers(0, 2**31))
                    f = eval_executor.submit(
                        _eval_matchup_worker,
                        model_path,
                        args.side,
                        opp,
                        args.eval_games,
                        args.eval_score,
                        args.simplify_observation,
                        seed,
                    )
                    futures[f] = opp

                results: dict[str, dict] = {}
                for f in as_completed(futures):
                    opp_name, result = f.result()
                    results[opp_name] = result

                # Update pool stats
                for opp_name, r in results.items():
                    for winner in r["game_winners"]:
                        pool.update_stats(opp_name, winner == args.side)

                # Try unlock
                newly_unlocked = pool.try_unlock()

                # Log
                log_data: dict = {"iteration": iteration}
                status = pool.status()
                log_data["curriculum/pool_size"] = status["pool_size"]
                log_data["curriculum/min_win_rate"] = status["min_win_rate"]
                log_data["curriculum/avg_win_rate"] = status["avg_win_rate"]

                for opp_name, r in results.items():
                    log_data[f"curriculum/vs_{opp_name}/win_rate"] = r["win_rate"]
                    log_data[f"curriculum/vs_{opp_name}/avg_score"] = r["avg_score"]
                    for k in [
                        "serve_win_rate",
                        "receive_win_rate",
                        "avg_round_frames",
                        "std_round_frames",
                        "action_entropy",
                        "power_hit_rate",
                        "ball_own_side_ratio",
                    ]:
                        if k in r:
                            log_data[f"curriculum/vs_{opp_name}/{k}"] = r[k]

                run.log(log_data, step=step)

                # Print
                print(f"\n[Iter {iteration}/{args.total_iterations}, step={step}]", flush=True)
                print(f"  Pool ({status['pool_size']}): {pool.unlocked}", flush=True)
                for opp_name, r in results.items():
                    wr = pool.get_win_rate(opp_name)
                    print(f"    vs {opp_name}: {r['wins']}W {r['losses']}L (pool wr={wr:.2f})", flush=True)
                if newly_unlocked:
                    print(f"  >>> UNLOCKED: {newly_unlocked}!", flush=True)

            # --- TRAIN ---
            opp_spec = pool.sample_opponent()
            opp_policy = _make_opponent(opp_spec)
            for env in envs.envs:
                set_opponent_policy(env, opp_policy)
                env.reset()

            model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)

            # Log SB3 metrics
            if model.logger is not None and hasattr(model.logger, "name_to_value"):
                metrics = {k: v for k, v in model.logger.name_to_value.items()}
                if metrics:
                    run.log(metrics, step=model.num_timesteps)

            # Save periodic checkpoints
            if iteration % args.save_interval == 0:
                save_model(model, save_dir / f"iter_{iteration:06d}", model_cfg)

        # Final save
        final_dir = save_model(model, save_dir / "final", model_cfg)
        final_zip = str(final_dir / "model.zip")
        artifact = wandb.Artifact("curriculum-final", type="model")
        artifact.add_dir(str(final_dir))
        run.log_artifact(artifact)

        # Record sample videos vs unlocked opponents
        for opp in pool.unlocked:
            video_path = str(save_dir / f"vs_{opp}.mp4")
            _record_video(final_zip, args.side, opp, video_path)
            run.log({f"video/vs_{opp}": wandb.Video(video_path, fps=25, format="mp4")})

        print(f"\nTraining complete. Final pool: {pool.unlocked}")
        print(f"Model saved to {final_dir}")
    finally:
        eval_executor.shutdown(wait=False, cancel_futures=True)
        envs.close()
        run.finish()


if __name__ == "__main__":
    main()
