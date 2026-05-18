"""Curriculum training: PPO against progressively harder opponents.

Starts with easy opponents, unlocks harder ones as the model improves.
Uses PFP sampling within the unlocked pool. With --side both, the
ladder may include the special spec "self" for past-checkpoint
self-play (universal models only).

Usage:
  uv run train-curriculum --save-dir experiments/010 --total-iterations 200
  uv run train-curriculum --save-dir experiments/010 --unlock-threshold 0.8
  uv run train-curriculum --side both \\
      --ladder random builtin duckll:0 ... duckll:7 self duckll:5 ... duckll:10
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import tempfile
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import wandb
from pika_zoo.ai import BuiltinAI, DuckllAI, RandomAI, StoneAI
from pika_zoo.ai.protocol import AIPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from training_center.env_factory import ensure_stack_size, make_vec_env, set_opponent_policy
from training_center.game import make_player, play_game
from training_center.metadata import get_experiment_metadata
from training_center.model_config import ModelConfig, load_model_config, save_model
from training_center.policy_config import (
    add_policy_args,
    ensure_policy_config_matches_init,
    policy_request_is_explicit,
    resolve_policy_config,
)
from training_center.pool import CurriculumPool, make_opponent_policy
from training_center.pool.curriculum import SELF_ENTRY
from training_center.scripts.utils import (
    SERVE_RULES,
    EvalBatch,
    EvalResult,
    build_eval_chart_log_data,
    build_train_chart_log_data,
    build_video_log_data,
    combine_per_side_results,
    extend_curriculum_chart_history,
    extend_eval_chart_history,
    extend_train_chart_history,
    parse_noise,
    record_video,
    setup_graceful_shutdown,
    shutdown_executor,
    worker_init,
)

# ELO ladder from experiment 009 — batch Bradley-Terry (ascending difficulty),
# with stone removed from the default gate after S009 showed it acts as a
# brittle early P2 receive trap for universal noisy curriculum runs. Stone is
# still available when explicitly passed via --ladder or evaluated post-hoc.
# The special spec "self" can be inserted (after duckll:7 mastery, e.g.) when
# training a universal model with --side both — see module docstring.
CURRICULUM_LADDER = [
    "random",
    "builtin",
    "duckll:0",
    "duckll:1",
    "duckll:2",
    "duckll:3",
    "duckll:7",
    "duckll:5",
    "duckll:4",
    "duckll:6",
    "duckll:8",
    "duckll:9",
    "duckll:10",
]


def _make_rule_opponent(spec: str) -> AIPolicy:
    """Instantiate a rule-based AI policy from a spec string."""
    if spec == "builtin":
        return BuiltinAI()
    elif spec == "builtin_bugfix":
        return BuiltinAI(bugfix=True)
    elif spec == "stone":
        return StoneAI()
    elif spec == "stone_random":
        return StoneAI(random_position=True)
    elif spec == "duckll" or spec.startswith("duckll:"):
        preset = int(spec.split(":")[1]) if ":" in spec else None
        return DuckllAI(preset=preset) if preset is not None else DuckllAI()
    return RandomAI()


def _make_self_policy(self_pool: deque[str], current_model: PPO) -> Callable[[np.ndarray], int]:
    """Build an opponent callable from the self-play checkpoint pool.

    Falls back to the current learner when the pool is empty (e.g. before
    the first checkpoint is saved).
    """
    if self_pool:
        path = random.choice(list(self_pool))
        opponent_model = PPO.load(path, device="cpu")
    else:
        opponent_model = current_model
    return make_opponent_policy(opponent_model)


class OpponentShuffleCallback(BaseCallback):
    """Reshuffle opponents per-env at the start of each rollout."""

    def __init__(
        self,
        envs,
        pool: CurriculumPool,
        self_pool: deque[str],
        current_model: PPO,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.envs = envs
        self.pool = pool
        self.self_pool = self_pool
        self.current_model = current_model

    def _on_rollout_start(self) -> None:
        for env in self.envs.envs:
            opp_spec = self.pool.sample_opponent()
            if opp_spec == SELF_ENTRY:
                policy = _make_self_policy(self.self_pool, self.current_model)
            else:
                policy = _make_rule_opponent(opp_spec)
            set_opponent_policy(env, policy)
            env.reset()

    def _on_step(self) -> bool:
        return True


def _eval_matchup_worker(
    model_path: str,
    model_side: str,
    opp_name: str,
    games: int,
    winning_score: int,
    serve: str,
    simplify_observation: bool,
    seed: int,
) -> tuple[str, EvalResult]:
    """Worker: evaluate model vs one opponent. Returns (opp_name, EvalResult)."""
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
            episode = play_game(
                model_player,
                opp_player,
                winning_score=winning_score,
                serve=serve,
                seed=game_seed,
                record_frames=True,
                simplify_observation=simplify_observation,
            )
        else:
            episode = play_game(
                opp_player,
                model_player,
                winning_score=winning_score,
                serve=serve,
                seed=game_seed,
                record_frames=True,
                simplify_observation=simplify_observation,
            )
        all_episodes.append(episode)

    return opp_name, EvalResult.from_episodes(
        all_episodes,
        model_name=Path(model_path).parent.name,
        opponent_name=opp_name,
        model_side=model_side,
        opponent_side="player_2" if model_side == "player_1" else "player_1",
        model_path=model_path,
        opponent_spec=opp_name,
        winning_score=winning_score,
        seed=seed,
    )


def main() -> None:
    ensure_stack_size()

    default_ladder_str = " → ".join(CURRICULUM_LADDER)
    parser = argparse.ArgumentParser(
        description="Curriculum training: progressive difficulty",
        epilog=(
            f"Default ladder (ELO order from experiment 009):\n  {default_ladder_str}\n"
            "  Source: https://api.wandb.ai/links/ootzk/8za7h3er\n\n"
            "Add 'self' to the ladder (requires --side both) for past-checkpoint self-play."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--total-iterations", type=int, default=50)
    parser.add_argument("--steps-per-iter", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--side",
        default="player_1",
        choices=["player_1", "player_2", "both"],
        help="Which side(s) the learner trains on. 'both' = universal model "
        "(envs split half P1 / half P2; auto-enables --simplify-observation).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-level", "--noise_level", type=int, default=None, choices=[0, 1, 2, 3, 4, 5], help="Noise preset level"
    )
    parser.add_argument("--noise-x", type=int, default=None, help="Ball x position noise ±N pixels")
    parser.add_argument("--noise-x-vel", type=int, default=None, help="Ball x velocity noise ±N")
    parser.add_argument("--noise-y-vel", type=int, default=None, help="Ball y velocity noise ±N")
    parser.add_argument("--serve", choices=SERVE_RULES, default="winner", help="Serve rule for training games")
    parser.add_argument(
        "--eval-serve",
        choices=SERVE_RULES,
        default=None,
        help="Serve rule for eval games (default: same as --serve)",
    )
    parser.add_argument(
        "--video-serve",
        choices=SERVE_RULES,
        default=None,
        help="Serve rule for sample videos (default: same as --eval-serve)",
    )
    parser.add_argument("--simplify-observation", action="store_true", help="Mirror player_2 x-axis observations")
    parser.add_argument(
        "--frame-stack", type=int, default=1, help="Number of recent observations to stack (1=disabled)"
    )
    parser.add_argument(
        "--ladder",
        nargs="+",
        default=CURRICULUM_LADDER,
        metavar="SPEC",
        help="Ordered list of opponent specs. Use 'self' (requires --side both) "
        "for past-checkpoint self-play. Default: experiment-009 ELO ladder.",
    )
    parser.add_argument("--unlock-threshold", type=float, default=0.75, help="Min win rate to unlock next opponent")
    parser.add_argument("--initial-unlocked", type=int, default=2, help="Number of opponents unlocked at start")
    parser.add_argument(
        "--selfplay-pool-size",
        type=int,
        default=20,
        help="Max number of past checkpoints kept in the self-play pool (FIFO). "
        "Only used when 'self' is in the ladder.",
    )
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluate every N iterations")
    parser.add_argument("--eval-games", type=int, default=10, help="Games per opponent per evaluation")
    parser.add_argument("--eval-score", type=int, default=5, help="Winning score for eval games")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--init-model", default=None, help="Pretrained model path to resume from")
    add_policy_args(parser)
    parser.add_argument("--wandb-entity", default="ootzk", help="W&B entity")
    parser.add_argument("--wandb-project", default="alphachu-volleyball", help="W&B project")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    args = parser.parse_args()

    ladder = list(args.ladder)
    if SELF_ENTRY in ladder and args.side != "both":
        print(f"Warning: '{SELF_ENTRY}' in ladder requires --side both (universal model); stripping it and proceeding.")
        ladder = [s for s in ladder if s != SELF_ENTRY]

    if args.side == "both" and not args.simplify_observation:
        print("Note: --side both auto-enables --simplify-observation")
        args.simplify_observation = True
    if args.eval_serve is None:
        args.eval_serve = args.serve
    if args.video_serve is None:
        args.video_serve = args.eval_serve

    save_dir = Path(args.save_dir)
    meta = get_experiment_metadata()

    noise = parse_noise(args.noise_level, args.noise_x, args.noise_x_vel, args.noise_y_vel)

    # Curriculum pool
    pool = CurriculumPool(ladder, unlock_threshold=args.unlock_threshold)
    for i in range(min(args.initial_unlocked, len(ladder))):
        pool.force_unlock(i)

    # Self-play checkpoint pool (paths). FIFO ring; populated by save_model below.
    selfplay_pool: deque[str] = deque(maxlen=args.selfplay_pool_size)

    requested_policy, requested_policy_kwargs = resolve_policy_config(
        policy=args.policy,
        net_arch=args.net_arch,
        policy_kwargs_json=args.policy_kwargs_json,
    )
    policy_explicit = policy_request_is_explicit(
        policy=args.policy,
        net_arch=args.net_arch,
        policy_kwargs_json=args.policy_kwargs_json,
    )
    if args.init_model:
        _, init_config = load_model_config(args.init_model)
        if policy_explicit:
            ensure_policy_config_matches_init(
                init_policy=init_config.policy,
                init_policy_kwargs=init_config.policy_kwargs,
                requested_policy=requested_policy,
                requested_policy_kwargs=requested_policy_kwargs,
            )
        training_policy = init_config.policy
        training_policy_kwargs = init_config.policy_kwargs
    else:
        training_policy = requested_policy
        training_policy_kwargs = requested_policy_kwargs

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
            "serve": args.serve,
            "eval_serve": args.eval_serve,
            "video_serve": args.video_serve,
            "simplify_observation": args.simplify_observation,
            "frame_stack": args.frame_stack,
            "unlock_threshold": args.unlock_threshold,
            "initial_unlocked": args.initial_unlocked,
            "eval_freq": args.eval_freq,
            "ent_coef": args.ent_coef,
            "policy": training_policy,
            "policy_kwargs": training_policy_kwargs,
            "ladder": ladder,
            "selfplay_pool_size": args.selfplay_pool_size if SELF_ENTRY in ladder else None,
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
        frame_stack=args.frame_stack,
        serve=args.serve,
        noise=noise,
    )

    # Initialize model
    model_cfg = ModelConfig(
        side=args.side,
        observation_simplified=args.simplify_observation,
        frame_stack=args.frame_stack,
        policy=training_policy,
        policy_kwargs=training_policy_kwargs,
    )
    ppo_kwargs = dict(device="cpu", verbose=0, ent_coef=args.ent_coef)
    if args.init_model:
        model = PPO.load(args.init_model, env=envs, seed=args.seed, **ppo_kwargs)
        print(f"Loaded from {args.init_model}")
    else:
        model = PPO(training_policy, envs, seed=args.seed, policy_kwargs=training_policy_kwargs, **ppo_kwargs)

    mp_context = multiprocessing.get_context("forkserver")
    eval_executor = ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=mp_context, initializer=worker_init)

    print(f"Curriculum training: {args.total_iterations} iterations x {args.steps_per_iter} steps")
    print(f"Envs: {args.num_envs} (DummyVecEnv), side={args.side}")
    print(f"Serve: train={args.serve}, eval={args.eval_serve}, video={args.video_serve}")
    print(f"Policy: {training_policy}, policy_kwargs={training_policy_kwargs}")
    print(f"Unlock threshold: {args.unlock_threshold:.0%}")
    print(f"Initial pool: {pool.unlocked}")
    print(f"Ladder: {ladder}")
    if SELF_ENTRY in ladder:
        print(f"Self-play: enabled (pool size={args.selfplay_pool_size})")

    setup_graceful_shutdown()

    # Eval/pool-update sides. For universal models we run eval on both sides,
    # splitting args.eval_games half/half so the combined eval still covers
    # ~args.eval_games games per opponent (not 2x).
    eval_sides = ["player_1", "player_2"] if args.side == "both" else [args.side]
    per_side_eval_games = args.eval_games // len(eval_sides)
    eval_chart_history: dict[str, list[EvalResult]] = {}
    train_chart_history: list[dict] = []
    curriculum_chart_history: list[dict] = []
    eval_checkpoint_tmp = tempfile.TemporaryDirectory(prefix="training-center-curriculum-eval-")
    eval_checkpoint_root = Path(eval_checkpoint_tmp.name)

    try:
        for iteration in range(args.total_iterations):
            # --- TRAIN ---
            model.learn(
                total_timesteps=args.steps_per_iter,
                reset_num_timesteps=False,
                callback=OpponentShuffleCallback(envs, pool, selfplay_pool, model),
            )

            # Log SB3 metrics
            if model.logger is not None and hasattr(model.logger, "name_to_value"):
                metrics = {k: v for k, v in model.logger.name_to_value.items()}
                if metrics:
                    history_len = len(train_chart_history)
                    extend_train_chart_history(train_chart_history, metrics, step=model.num_timesteps)
                    if len(train_chart_history) > history_len:
                        run.log(
                            build_train_chart_log_data(
                                train_chart_history,
                                curriculum_history=curriculum_chart_history,
                            ),
                            step=model.num_timesteps,
                        )

            # Save periodic self-play snapshots only when self-play is active.
            if SELF_ENTRY in pool.unlocked and iteration % args.save_interval == 0:
                iter_dir = save_model(model, save_dir / "selfplay" / f"iter_{iteration:06d}", model_cfg)
                selfplay_pool.append(str(iter_dir / "model.zip"))

            # --- EVALUATE ---
            is_last = iteration == args.total_iterations - 1
            if (iteration + 1) % args.eval_freq == 0 or is_last:
                step = model.num_timesteps
                ckpt_dir = save_model(model, eval_checkpoint_root / f"checkpoint_{iteration:06d}", model_cfg)
                model_path = str(ckpt_dir / "model.zip")

                artifact = wandb.Artifact(f"curriculum-checkpoint-{iteration:06d}", type="model")
                artifact.add_dir(str(ckpt_dir))
                run.log_artifact(artifact)

                # Evaluate vs all unlocked opponents (parallel). Self is excluded:
                # self vs self ~= 50% provides no signal and can't be loaded as an
                # opponent spec via make_player. For universal models we eval on
                # both sides, then combine for the main `eval/...` metrics and
                # pool-stat updates while also exposing `eval/p1/...` and
                # `eval/p2/...` so per-side asymmetry is visible.
                rng = np.random.default_rng()
                futures = {}
                eval_opponents = [opp for opp in pool.unlocked if opp != SELF_ENTRY]
                for side in eval_sides:
                    for opp in eval_opponents:
                        seed = int(rng.integers(0, 2**31))
                        f = eval_executor.submit(
                            _eval_matchup_worker,
                            model_path,
                            side,
                            opp,
                            per_side_eval_games,
                            args.eval_score,
                            args.eval_serve,
                            args.simplify_observation,
                            seed,
                        )
                        futures[f] = (side, opp)

                results_per_side: dict[str, dict[str, EvalResult]] = {s: {} for s in eval_sides}
                for f in as_completed(futures):
                    side, _ = futures[f]
                    opp_name, result = f.result()
                    results_per_side[side][opp_name] = result

                if len(eval_sides) == 2:
                    results_by_opponent = {
                        opp: combine_per_side_results(
                            results_per_side["player_1"][opp],
                            results_per_side["player_2"][opp],
                        )
                        for opp in eval_opponents
                    }
                else:
                    results_by_opponent = results_per_side[eval_sides[0]]

                eval_batch = EvalBatch(
                    [results_by_opponent[opp] for opp in eval_opponents],
                    iteration=iteration,
                    step=step,
                )

                # Pool stats: use combined per-opponent win rate (already
                # spans both sides for universal models — see results above).
                for result in eval_batch.results:
                    pool.set_win_rate(result.opponent_name, result.win_rate)

                # Try unlock
                newly_unlocked = pool.try_unlock()

                # Log
                log_data: dict = {}
                status = pool.status()
                extend_curriculum_chart_history(
                    curriculum_chart_history,
                    status,
                    iteration=iteration,
                    step=step,
                    selfplay_pool_size=len(selfplay_pool) if SELF_ENTRY in pool.unlocked else None,
                )
                log_data.update(
                    build_train_chart_log_data(
                        train_chart_history,
                        curriculum_history=curriculum_chart_history,
                    )
                )

                eval_chart_batches = {"combined": eval_batch}
                if len(eval_sides) == 2:
                    p1_batch = EvalBatch(
                        list(results_per_side["player_1"].values()),
                        iteration=iteration,
                        step=step,
                    )
                    p2_batch = EvalBatch(
                        list(results_per_side["player_2"].values()),
                        iteration=iteration,
                        step=step,
                    )
                    eval_chart_batches["p1"] = p1_batch
                    eval_chart_batches["p2"] = p2_batch
                log_data.update(
                    build_eval_chart_log_data(
                        extend_eval_chart_history(eval_chart_history, eval_chart_batches),
                        unlock_threshold=args.unlock_threshold,
                    )
                )

                run.log(log_data, step=step)

                # Print
                print(f"\n[Iter {iteration + 1}/{args.total_iterations}, step={step}]", flush=True)
                print(f"  Pool ({status['pool_size']}): {pool.unlocked}", flush=True)
                for line in eval_batch.format_score_frame_lines():
                    print(line, flush=True)
                if newly_unlocked:
                    print(f"  >>> UNLOCKED: {newly_unlocked}!", flush=True)

        # Final save
        final_dir = save_model(model, save_dir / "final", model_cfg)
        final_zip = str(final_dir / "model.zip")
        artifact = wandb.Artifact("curriculum-final", type="model")
        artifact.add_dir(str(final_dir))
        run.log_artifact(artifact)

        # Record sample videos vs unlocked opponents (skip self — needs a checkpoint
        # opponent spec, not meaningful as a final demo). For universal models,
        # record one video per side.
        video_sides = ["player_1", "player_2"] if args.side == "both" else [args.side]
        video_opponents = [opp for opp in pool.unlocked if opp != SELF_ENTRY]
        video_samples = []
        print(f"Recording sample videos ({len(video_opponents) * len(video_sides)} games)...", flush=True)
        for opp in video_opponents:
            for video_side in video_sides:
                tag = "p1" if video_side == "player_1" else "p2"
                suffix = f"_as_{tag}" if args.side == "both" else ""
                video_path = str(save_dir / f"vs_{opp}{suffix}.mp4")
                video_result = record_video(final_zip, video_side, opp, video_path, serve=args.video_serve)
                video_samples.append(
                    {
                        "opponent": opp,
                        "model_side": tag,
                        "serve": args.video_serve,
                        **video_result,
                        "video": wandb.Video(video_path, format="mp4"),
                    }
                )
        if video_samples:
            print(f"Sample videos recorded ({len(video_samples)} games).", flush=True)
            run.log(build_video_log_data(video_samples))

        print(f"\nTraining complete. Final pool: {pool.unlocked}")
        print(f"Model saved to {final_dir}")
    finally:
        shutdown_executor(eval_executor)
        envs.close()
        run.finish()
        eval_checkpoint_tmp.cleanup()


if __name__ == "__main__":
    main()
