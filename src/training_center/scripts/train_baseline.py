"""Baseline training script: PPO against a fixed opponent (random or builtin).

Usage:
  uv run train-baseline --timesteps 1000000
  uv run train-baseline --opponent builtin --timesteps 1000000 --eval-freq 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from pika_zoo.ai import BuiltinAI, RandomAI
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from training_center.env_factory import make_vec_env
from training_center.eval.elo import evaluate_model
from training_center.metadata import get_experiment_metadata


def _record_video(model_path: str, side: str, opponent: str, output_path: str) -> None:
    """Record a sample game video using pika-zoo's play script."""
    from pika_zoo.scripts.play import play

    p1 = model_path if side == "player_1" else opponent
    p2 = opponent if side == "player_1" else model_path
    play(p1=p1, p2=p2, winning_score=5, render=False, record=output_path, seed=0)


class WandbMetricsCallback(BaseCallback):
    """Forward SB3 training metrics to wandb."""

    def _on_step(self) -> bool:
        if self.logger is not None and hasattr(self.logger, "name_to_value"):
            metrics = {k: v for k, v in self.logger.name_to_value.items()}
            if metrics:
                wandb.run.log(metrics, step=self.num_timesteps)
        return True


class EloEvalCallback(BaseCallback):
    """Periodically evaluate the model's ELO during training."""

    def __init__(self, eval_freq: int, save_path: Path, eval_games: int = 20, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.eval_games = eval_games

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            tmp_path = self.save_path.parent / (self.save_path.name + "_tmp")
            self.model.save(str(tmp_path))

            results, elo = evaluate_model(
                str(tmp_path),
                opponents=("random", "builtin"),
                games=self.eval_games,
                winning_score=5,
            )

            log_data = {"eval/elo": elo, "eval/step": self.num_timesteps}
            for opp_name, (wins, losses) in results.items():
                log_data[f"eval/win_rate_{opp_name}"] = wins / (wins + losses)

            wandb.run.log(log_data, step=self.num_timesteps)

            if self.verbose:
                print(f"\n[Eval @ {self.num_timesteps} steps] ELO: {elo:.0f}")
                for opp_name, (wins, losses) in results.items():
                    print(f"  vs {opp_name}: {wins}W {losses}L")

            tmp_path.with_suffix(".zip").unlink()

        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline PPO training against a fixed opponent")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--save-path", required=True, help="Path to save the trained model")
    parser.add_argument("--side", default="player_1", choices=["player_1", "player_2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noisy", action="store_true", help="Add random perturbation to ball initial state")
    parser.add_argument("--opponent", default="random", choices=["random", "builtin"])
    parser.add_argument("--eval-freq", type=int, default=0, help="ELO eval frequency in steps (0=disabled)")
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity (user or team)")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    meta = get_experiment_metadata()

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
            "noisy": args.noisy,
            "eval_freq": args.eval_freq,
            **meta,
        },
    )

    opponent_policy = BuiltinAI() if args.opponent == "builtin" else RandomAI()

    env = make_vec_env(
        n_envs=args.num_envs,
        agent=args.side,
        opponent_policy=opponent_policy,
        use_subproc=True,
        seed=args.seed,
        noisy=args.noisy,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    callbacks = [WandbMetricsCallback()]
    if args.eval_freq > 0:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            EloEvalCallback(
                eval_freq=args.eval_freq // args.num_envs,
                save_path=save_path,
            )
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

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
        _record_video(model_zip, args.side, opp, video_path)
        run.log({f"video/vs_{opp}": wandb.Video(video_path, fps=25, format="mp4")})

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
