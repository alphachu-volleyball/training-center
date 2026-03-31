"""Baseline training script: PPO against a fixed opponent (random or builtin).

Usage:
  uv run train-baseline --timesteps 1000000
  uv run train-baseline --opponent builtin --timesteps 1000000 --eval-freq 50000
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import wandb
from pika_zoo.ai import BuiltinAI, DuckllAI, RandomAI, StoneAI
from pika_zoo.env.pikachu_volleyball import NoiseConfig
from pika_zoo.records.types import GamesRecord
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from training_center.elo import INITIAL_ELO, update_elo
from training_center.env_factory import make_vec_env
from training_center.game import make_player, play_game
from training_center.metadata import get_experiment_metadata
from training_center.metrics import compute_eval_metrics


def _record_video(model_path: str, side: str, opponent: str, output_path: str) -> None:
    """Record a sample game video using pika-zoo's play script."""
    from pika_zoo.scripts.play import play

    p1 = model_path if side == "player_1" else opponent
    p2 = opponent if side == "player_1" else model_path
    play(p1=p1, p2=p2, winning_score=5, render=False, record=output_path, seed=0)


def _eval_matchup_worker(
    model_path: str,
    model_side: str,
    opp_name: str,
    games: int,
    winning_score: int,
    simplify_observation: bool,
    seed: int,
) -> tuple[str, dict]:
    """Worker: evaluate model vs one opponent in a child process.

    Returns (opp_name, result_dict) with per-game winners, scores, rounds, and metrics.
    """
    model_player = make_player(model_path, agent=model_side, simplify_observation=simplify_observation)
    opp_player = make_player(opp_name, agent="player_2" if model_side == "player_1" else "player_1",
                             simplify_observation=simplify_observation)
    rng = np.random.default_rng(seed)
    all_episodes = []

    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        if model_side == "player_1":
            episode = play_game(model_player, opp_player, winning_score=winning_score,
                                seed=game_seed, record_frames=True)
        else:
            episode = play_game(opp_player, model_player, winning_score=winning_score,
                                seed=game_seed, record_frames=True)
        all_episodes.append(episode)

    # Compute metrics inside worker to avoid serializing frame data
    opp_side = "player_2" if model_side == "player_1" else "player_1"
    model_idx = 0 if model_side == "player_1" else 1
    all_rounds = [r for e in all_episodes for r in e.rounds]
    model_serve = [r for r in all_rounds if r.server == model_side]
    opp_serve = [r for r in all_rounds if r.server == opp_side]

    wins = sum(1 for e in all_episodes if e.winner == model_side)
    detail = compute_eval_metrics(GamesRecord(games=all_episodes), model_side)

    result = {
        "wins": wins,
        "losses": games - wins,
        "win_rate": wins / games,
        "avg_score": float(np.mean([e.scores[model_idx] for e in all_episodes])),
        "serve_win_rate": (sum(1 for r in model_serve if r.scorer == model_side) / len(model_serve))
        if model_serve else None,
        "receive_win_rate": (sum(1 for r in opp_serve if r.scorer == model_side) / len(opp_serve))
        if opp_serve else None,
        "game_winners": [e.winner for e in all_episodes],
        **detail,
    }
    return opp_name, result


class WandbMetricsCallback(BaseCallback):
    """Forward SB3 training metrics to wandb."""

    def _on_step(self) -> bool:
        if self.logger is not None and hasattr(self.logger, "name_to_value"):
            metrics = {k: v for k, v in self.logger.name_to_value.items()}
            if metrics:
                wandb.run.log(metrics, step=self.num_timesteps)
        return True


class EvalCallback(BaseCallback):
    """Periodically evaluate with detailed stats, save checkpoints, and log to wandb."""

    def __init__(
        self,
        eval_freq: int,
        save_path: Path,
        side: str = "player_1",
        eval_games: int = 20,
        eval_opponents: list[str] | None = None,
        simplify_observation: bool = False,
        executor: ProcessPoolExecutor | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.side = side
        self.eval_games = eval_games
        self.eval_opponents = eval_opponents or ["random", "builtin"]
        self.simplify_observation = simplify_observation
        self.executor = executor

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Save checkpoint
            ckpt_path = self.save_path.parent / f"checkpoint_{self.num_timesteps}"
            self.model.save(str(ckpt_path))
            model_path = str(ckpt_path) + ".zip"

            artifact = wandb.Artifact(f"baseline-checkpoint-{self.num_timesteps}", type="model")
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)

            # Evaluate against each opponent in parallel
            model_side = self.side
            rng = np.random.default_rng()

            if self.executor is not None:
                futures = {}
                for opp_name in self.eval_opponents:
                    seed = int(rng.integers(0, 2**31))
                    f = self.executor.submit(
                        _eval_matchup_worker, model_path, model_side, opp_name,
                        self.eval_games, 5, self.simplify_observation, seed,
                    )
                    futures[f] = opp_name

                results: dict[str, dict] = {}
                for f in as_completed(futures):
                    opp_name, result = f.result()
                    results[opp_name] = result
            else:
                results = {}
                for opp_name in self.eval_opponents:
                    seed = int(rng.integers(0, 2**31))
                    _, result = _eval_matchup_worker(
                        model_path, model_side, opp_name,
                        self.eval_games, 5, self.simplify_observation, seed,
                    )
                    results[opp_name] = result

            # Compute ELO and log
            elo = INITIAL_ELO
            log_data: dict = {}

            for opp_name in self.eval_opponents:
                r = results[opp_name]
                # Update ELO from per-game winners
                opp_elo = INITIAL_ELO
                for winner in r["game_winners"]:
                    game_result = 1 if winner == model_side else 0
                    elo, opp_elo = update_elo(elo, opp_elo, game_result)

                log_data[f"eval/vs_{opp_name}/win_rate"] = r["win_rate"]
                log_data[f"eval/vs_{opp_name}/avg_score"] = r["avg_score"]
                for k in ["avg_round_frames", "std_round_frames", "action_entropy",
                          "power_hit_rate", "ball_own_side_ratio",
                          "serve_avg_round_frames", "receive_avg_round_frames"]:
                    if k in r:
                        log_data[f"eval/vs_{opp_name}/{k}"] = r[k]
                if r["serve_win_rate"] is not None:
                    log_data[f"eval/vs_{opp_name}/serve_win_rate"] = r["serve_win_rate"]
                if r["receive_win_rate"] is not None:
                    log_data[f"eval/vs_{opp_name}/receive_win_rate"] = r["receive_win_rate"]

                if self.verbose:
                    print(f"  vs {opp_name}: {r['wins']}W {r['losses']}L")

            log_data["eval/elo"] = elo
            wandb.run.log(log_data, step=self.num_timesteps)

            if self.verbose:
                print(f"\n[Eval @ {self.num_timesteps} steps] ELO: {elo:.0f}")

        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline PPO training against a fixed opponent")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--save-path", required=True, help="Path to save the trained model")
    parser.add_argument("--side", default="player_1", choices=["player_1", "player_2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-level", "--noise_level", type=int, default=None, choices=[0, 1, 2, 3, 4, 5], help="Noise preset level"
    )
    parser.add_argument("--noise-x", type=int, default=None, help="Ball x position noise ±N pixels")
    parser.add_argument("--noise-x-vel", type=int, default=None, help="Ball x velocity noise ±N")
    parser.add_argument("--noise-y-vel", type=int, default=None, help="Ball y velocity noise ±N")
    parser.add_argument("--simplify-observation", action="store_true", help="Mirror player_2 x-axis observations")
    parser.add_argument("--opponent", default="random", help="Opponent: random, builtin, stone, duckll, duckll:N")
    parser.add_argument("--eval-freq", type=int, default=0, help="ELO eval frequency in steps (0=disabled)")
    parser.add_argument("--init-model", default=None, help="Pretrained model path to resume from")
    parser.add_argument("--resume-steps", action="store_true", help="Continue step count from init-model")
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity (user or team)")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    meta = get_experiment_metadata()

    # wandb.init with argparse defaults — sweep agent overrides these via wandb.config
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "script": "train_baseline",
            "timesteps": args.timesteps,
            "num_envs": args.num_envs,
            "side": args.side,
            "opponent": args.opponent,
            "seed": args.seed,
            "noise_level": args.noise_level,
            "noise_x": args.noise_x,
            "noise_x_vel": args.noise_x_vel,
            "noise_y_vel": args.noise_y_vel,
            "simplify_observation": args.simplify_observation,
            "init_model": args.init_model,
            "eval_freq": args.eval_freq,
            **meta,
        },
    )

    # Read from wandb.config so sweep overrides take effect
    c = wandb.config

    NOISE_LEVELS = {
        0: (0, 0, 0),
        1: (5, 3, 1),
        2: (10, 5, 2),
        3: (20, 10, 3),
        4: (35, 15, 4),
        5: (50, 20, 5),
    }

    noise = None
    if c.noise_level is not None and c.noise_level > 0:
        x, xv, yv = NOISE_LEVELS[c.noise_level]
        noise = NoiseConfig(x_range=x, x_velocity_range=xv, y_velocity_range=yv)
    elif c.noise_x is not None or c.noise_x_vel is not None or c.noise_y_vel is not None:
        noise = NoiseConfig(
            x_range=c.noise_x or 0,
            x_velocity_range=c.noise_x_vel or 0,
            y_velocity_range=c.noise_y_vel or 0,
        )

    def _make_opponent(spec: str) -> BuiltinAI | RandomAI | StoneAI | DuckllAI:
        if spec == "builtin":
            return BuiltinAI()
        elif spec == "stone":
            return StoneAI()
        elif spec == "duckll" or spec.startswith("duckll:"):
            preset = int(spec.split(":")[1]) if ":" in spec else None
            return DuckllAI(preset=preset) if preset is not None else DuckllAI()
        return RandomAI()

    opponent_policy = _make_opponent(c.opponent)

    env = make_vec_env(
        n_envs=c.num_envs,
        agent=c.side,
        opponent_policy=opponent_policy,
        use_subproc=True,
        seed=c.seed,
        simplify_observation=c.simplify_observation,
        noise=noise,
    )

    if c.init_model:
        model = PPO.load(c.init_model, env=env, seed=c.seed, device="cpu", verbose=1)
        print(f"Resumed from {c.init_model}")
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=c.seed, device="cpu")

    eval_executor = ProcessPoolExecutor(max_workers=os.cpu_count()) if c.eval_freq > 0 else None

    callbacks = [WandbMetricsCallback()]
    if c.eval_freq > 0:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            EvalCallback(
                eval_freq=c.eval_freq // c.num_envs,
                save_path=save_path,
                side=c.side,
                simplify_observation=c.simplify_observation,
                executor=eval_executor,
            )
        )

    model.learn(total_timesteps=c.timesteps, callback=callbacks, reset_num_timesteps=not args.resume_steps)

    if eval_executor is not None:
        eval_executor.shutdown()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")

    model_zip = str(save_path) + ".zip"
    artifact = wandb.Artifact("baseline-final", type="model")
    artifact.add_file(model_zip)
    run.log_artifact(artifact)

    # Record sample videos
    for opp in ["builtin", "random"]:
        video_path = str(save_path.parent / f"vs_{opp}.mp4")
        _record_video(model_zip, c.side, opp, video_path)
        run.log({f"video/vs_{opp}": wandb.Video(video_path, fps=25, format="mp4")})

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
