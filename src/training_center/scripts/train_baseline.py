"""Baseline training script: PPO against a fixed opponent (random or builtin).

Usage:
  uv run tc-train-baseline --timesteps 1000000
  uv run tc-train-baseline --opponent builtin --timesteps 1000000 --eval-freq 50000
"""

from __future__ import annotations

import argparse
import os

from pika_zoo.ai import BuiltinAI, RandomAI
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from training_center.env_factory import make_vec_env
from training_center.eval.elo import evaluate_model
from training_center.metadata import get_experiment_metadata


class EloEvalCallback(BaseCallback):
    """Periodically evaluate the model's ELO during training."""

    def __init__(self, eval_freq: int, save_path: str, eval_games: int = 20, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.eval_games = eval_games

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            tmp_path = self.save_path + "_tmp"
            self.model.save(tmp_path)

            results, elo = evaluate_model(
                tmp_path,
                opponents=("random", "builtin"),
                games=self.eval_games,
                winning_score=5,
            )

            self.logger.record("eval/elo", elo)
            for opp_name, (wins, losses) in results.items():
                win_pct = wins / (wins + losses)
                self.logger.record(f"eval/win_rate_{opp_name}", win_pct)

            if self.verbose:
                print(f"\n[Eval @ {self.num_timesteps} steps] ELO: {elo:.0f}")
                for opp_name, (wins, losses) in results.items():
                    print(f"  vs {opp_name}: {wins}W {losses}L")

            os.remove(tmp_path + ".zip")

        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline PPO training against a fixed opponent")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--save-path", default="models/checkpoints/ppo_pikazoo")
    parser.add_argument("--side", default="player_1", choices=["player_1", "player_2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random", choices=["random", "builtin"])
    parser.add_argument("--eval-freq", type=int, default=0, help="ELO eval frequency in steps (0=disabled)")
    parser.add_argument("--tensorboard-log", default=None)
    args = parser.parse_args()

    meta = get_experiment_metadata()
    print(f"Experiment metadata: {meta}")

    opponent_policy = BuiltinAI() if args.opponent == "builtin" else RandomAI()

    env = make_vec_env(
        n_envs=args.num_envs,
        agent=args.side,
        opponent_policy=opponent_policy,
        use_subproc=True,
        seed=args.seed,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device="cpu",
        tensorboard_log=args.tensorboard_log,
    )

    callbacks = []
    if args.eval_freq > 0:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        callbacks.append(
            EloEvalCallback(
                eval_freq=args.eval_freq // args.num_envs,
                save_path=args.save_path,
            )
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    env.close()


if __name__ == "__main__":
    main()
