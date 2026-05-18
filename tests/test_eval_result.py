"""Tests for evaluation result records and batch helpers."""

from training_center.scripts.utils import (
    EvalBatch,
    EvalResult,
    EvalSummary,
    _parse_video_result,
    build_eval_chart_log_data,
    build_eval_chart_table,
    build_train_chart_log_data,
    build_video_log_data,
    combine_per_side_results,
    combine_per_side_summaries,
    extend_curriculum_chart_history,
    extend_eval_chart_history,
    extend_train_chart_history,
    model_won_per_game,
)


def _result(
    side: str,
    *,
    wins: int,
    losses: int,
    winners: list[str],
    p1_scores: list[int] | None = None,
    p2_scores: list[int] | None = None,
    game_frames: list[int] | None = None,
    metrics: dict[str, float] | None = None,
) -> EvalResult:
    return EvalResult(
        model_name="model",
        opponent_name="builtin",
        model_side=side,
        opponent_side="player_2" if side == "player_1" else "player_1",
        summary=EvalSummary(
            wins=wins,
            losses=losses,
            game_winners=winners,
            p1_scores=p1_scores or [],
            p2_scores=p2_scores or [],
            game_frames=game_frames or [],
            metrics=metrics or {},
        ),
    )


def test_combine_per_side_results_basic_counts():
    p1 = _result("player_1", wins=8, losses=2, winners=["player_1"] * 8 + ["player_2"] * 2)
    p2 = _result("player_2", wins=4, losses=6, winners=["player_2"] * 4 + ["player_1"] * 6)
    combined = combine_per_side_results(p1, p2)
    assert combined.wins == 12
    assert combined.losses == 8
    assert abs(combined.win_rate - 0.6) < 1e-9
    assert combined.model_side == "both"
    assert len(combined.summary.game_winners) == 20


def test_combine_per_side_summaries_averages_metric_keys():
    p1 = EvalSummary(
        wins=5,
        losses=5,
        game_winners=["player_1"] * 5 + ["player_2"] * 5,
        metrics={"action_entropy": 2.0, "power_hit_rate": 0.8},
    )
    p2 = EvalSummary(
        wins=5,
        losses=5,
        game_winners=["player_2"] * 5 + ["player_1"] * 5,
        metrics={"action_entropy": 3.0, "power_hit_rate": 0.6},
    )
    combined = combine_per_side_summaries(p1, p2)
    assert abs(combined.metric("action_entropy") - 2.5) < 1e-9
    assert abs(combined.metric("power_hit_rate") - 0.7) < 1e-9


def test_combine_per_side_results_combines_score_and_frame_samples():
    p1 = _result(
        "player_1",
        wins=1,
        losses=1,
        winners=["player_1", "player_2"],
        p1_scores=[5, 5],
        p2_scores=[1, 3],
        game_frames=[100, 200],
    )
    p2 = _result(
        "player_2",
        wins=1,
        losses=1,
        winners=["player_2", "player_1"],
        p1_scores=[2, 4],
        p2_scores=[5, 5],
        game_frames=[300, 400],
    )
    combined = combine_per_side_results(p1, p2)
    assert combined.summary.p1_scores == [5, 5, 2, 4]
    assert combined.summary.p2_scores == [1, 3, 5, 5]
    assert abs(combined.summary.metric("avg_score") - 5.0) < 1e-9
    assert abs(combined.summary.metric("std_score") - 0.0) < 1e-9
    assert abs(combined.summary.metric("avg_opp_score") - 2.5) < 1e-9
    assert abs(combined.summary.metric("std_opp_score") - 1.118033988749895) < 1e-9
    assert abs(combined.summary.avg_p1_score - 4.0) < 1e-9
    assert abs(combined.summary.var_p1_score - 1.5) < 1e-9
    assert abs(combined.summary.std_p1_score - 1.224744871391589) < 1e-9
    assert abs(combined.summary.avg_p2_score - 3.5) < 1e-9
    assert abs(combined.summary.var_p2_score - 2.75) < 1e-9
    assert abs(combined.summary.std_p2_score - 1.6583123951777) < 1e-9
    assert abs(combined.summary.avg_game_frames - 250.0) < 1e-9
    assert abs(combined.summary.var_game_frames - 12500.0) < 1e-9
    assert abs(combined.summary.std_game_frames - 111.80339887498948) < 1e-9
    assert combined.format_score_frame_line() == "    vs builtin: 2W 2L (5.0±0.0 vs 2.5±1.1, frames: 250±112)"


def test_eval_summary_formats_score_frame_line():
    summary = EvalSummary(
        wins=3,
        losses=1,
        game_winners=["player_1", "player_1", "player_2", "player_1"],
        p1_scores=[5, 5, 3, 5],
        p2_scores=[1, 2, 5, 0],
        game_frames=[100, 120, 140, 160],
    )
    assert summary.format_score_frame_line("random") == ("    vs random: 3W 1L (4.5±0.9 vs 2.0±1.9, frames: 130±22)")


def test_eval_result_carries_identity_and_formats_without_external_label():
    summary = EvalSummary(
        wins=2,
        losses=0,
        game_winners=["player_1", "player_1"],
        p1_scores=[5, 5],
        p2_scores=[1, 2],
        game_frames=[100, 120],
    )
    result = EvalResult(
        model_name="checkpoint_000010",
        opponent_name="builtin",
        model_side="player_1",
        opponent_side="player_2",
        summary=summary,
        model_path="experiments/run/checkpoint_000010/model.zip",
        opponent_spec="builtin",
        winning_score=5,
        step=10_000,
    )
    assert result.opponent_name == "builtin"
    assert result.wins == 2
    assert result.to_record()["step"] == 10_000
    assert result.to_record()["winning_score"] == 5
    assert result.to_record()["opponent_spec"] == "builtin"
    assert result.format_score_frame_line() == "    vs builtin: 2W 0L (5.0±0.0 vs 1.5±0.5, frames: 110±10)"


def test_eval_batch_indexes_results():
    first = EvalResult(
        model_name="model",
        opponent_name="random",
        model_side="player_1",
        opponent_side="player_2",
        summary=EvalSummary(wins=1, losses=0, game_winners=["player_1"]),
    )
    second = EvalResult(
        model_name="model",
        opponent_name="builtin",
        model_side="player_1",
        opponent_side="player_2",
        summary=EvalSummary(wins=0, losses=1, game_winners=["player_2"]),
    )
    batch = EvalBatch([first, second], iteration=3, step=100)
    assert batch.by_opponent()["random"] is first
    assert first.iteration == 3
    assert second.step == 100
    assert [record["opponent_name"] for record in batch.to_records()] == ["random", "builtin"]


def test_eval_chart_table_is_single_long_form_source():
    batch = EvalBatch(
        [
            _result(
                "player_1",
                wins=1,
                losses=1,
                winners=["player_1", "player_2"],
                p1_scores=[5, 3],
                p2_scores=[1, 5],
                game_frames=[100, 120],
                metrics={
                    "avg_score": 4.0,
                    "std_score": 1.0,
                    "avg_opp_score": 3.0,
                    "std_opp_score": 2.0,
                },
            )
        ],
        iteration=7,
        step=1234,
    )

    table = build_eval_chart_table({"p1": batch})
    rows = {(row[3], row[4]): row for row in table.data}

    assert table.columns == [
        "step",
        "iteration",
        "opponent",
        "eval_side",
        "metric",
        "value",
        "std",
        "ci95_low",
        "ci95_high",
        "n",
        "wins",
        "losses",
    ]
    assert {row[3] for row in table.data} == {"combined", "p1"}
    assert rows[("combined", "win_rate")][:6] == [1234, 7, "builtin", "combined", "win_rate", 0.5]
    assert rows[("combined", "model_score")][:6] == [1234, 7, "builtin", "combined", "model_score", 4.0]
    assert rows[("combined", "opponent_score")][:6] == [1234, 7, "builtin", "combined", "opponent_score", 3.0]
    assert rows[("combined", "round_frames")][:6] == [1234, 7, "builtin", "combined", "round_frames", 0.0]
    assert rows[("combined", "game_frames")][:6] == [1234, 7, "builtin", "combined", "game_frames", 110.0]
    assert rows[("p1", "model_score")][:6] == [1234, 7, "builtin", "p1", "model_score", 4.0]


def test_eval_win_rate_chart_table_uses_nonzero_wilson_ci_at_extremes():
    batch = EvalBatch(
        [
            _result(
                "player_1",
                wins=0,
                losses=10,
                winners=["player_2"] * 10,
                p1_scores=[0] * 10,
                p2_scores=[5] * 10,
            )
        ],
        iteration=7,
        step=1234,
    )

    table = build_eval_chart_table({"p1": batch})
    win_rate_row = next(row for row in table.data if row[4] == "win_rate")
    _, _, _, _, _, win_rate, _, ci_low, ci_high, n, wins, losses = win_rate_row

    assert win_rate == 0.0
    assert ci_low == 0.0
    assert ci_high > 0.0
    assert (wins, losses, n) == (0, 10, 10)


def test_eval_chart_log_data_includes_immediate_plotly_panels():
    second = _result(
        "player_1",
        wins=2,
        losses=0,
        winners=["player_1", "player_1"],
        p1_scores=[5, 5],
        p2_scores=[1, 2],
        game_frames=[100, 120],
        metrics={
            "avg_score": 5.0,
            "std_score": 0.0,
            "avg_opp_score": 1.5,
            "std_opp_score": 0.5,
        },
    )
    second.opponent_name = "random"
    batch = EvalBatch(
        [
            _result(
                "player_1",
                wins=1,
                losses=1,
                winners=["player_1", "player_2"],
                p1_scores=[5, 3],
                p2_scores=[1, 5],
                game_frames=[100, 120],
                metrics={
                    "avg_score": 4.0,
                    "std_score": 1.0,
                    "avg_opp_score": 3.0,
                    "std_opp_score": 2.0,
                },
            ),
            second,
        ],
        iteration=7,
        step=1234,
    )

    log_data = build_eval_chart_log_data({"p1": batch})

    assert set(log_data) == {
        "eval/table",
        "eval/dashboard/builtin",
        "eval/dashboard/random",
    }
    fills = [trace.get("fill") for trace in log_data["eval/dashboard/builtin"].to_plotly_json()["data"]]
    assert "toself" in fills
    assert "tonexty" not in fills
    dashboard = log_data["eval/dashboard/builtin"].to_plotly_json()
    assert "updatemenus" not in dashboard["layout"]
    assert dashboard["layout"]["title"]["text"] == "Eval vs builtin"
    assert [annotation["text"] for annotation in dashboard["layout"]["annotations"]] == [
        "Win rate",
        "Model score",
        "Opponent score",
        "Round frames",
    ]
    assert dashboard["layout"]["paper_bgcolor"] == "rgba(255, 255, 255, 0)"
    assert dashboard["layout"]["plot_bgcolor"] == "rgba(255, 255, 255, 0)"
    assert dashboard["layout"]["autosize"] is True
    assert "height" not in dashboard["layout"]
    assert "0.75 threshold" not in [trace.get("name") for trace in dashboard["data"]]
    assert log_data["eval/dashboard/random"].to_plotly_json()["layout"]["title"]["text"] == "Eval vs random"


def test_train_chart_log_data_compacts_selected_sb3_metrics():
    history = []
    extend_train_chart_history(
        history,
        {
            "train/loss": 1.2,
            "train/entropy_loss": -0.3,
            "train/explained_variance": 0.4,
            "train/approx_kl": 0.01,
            "train/value_loss": 9.9,
        },
        step=100,
    )

    log_data = build_train_chart_log_data(history)
    dashboard = log_data["train/dashboard"].to_plotly_json()

    assert list(log_data) == ["train/dashboard"]
    assert dashboard["layout"]["autosize"] is True
    assert "height" not in dashboard["layout"]
    assert {trace["name"] for trace in dashboard["data"]} == {
        "Loss",
        "Entropy loss",
        "Explained variance",
        "Approx KL",
    }
    assert "Value loss" not in {trace["name"] for trace in dashboard["data"]}


def test_train_chart_log_data_can_append_curriculum_pool_size_subplot():
    train_history = []
    extend_train_chart_history(train_history, {"train/loss": 1.2}, step=100)
    curriculum_history = []
    extend_curriculum_chart_history(
        curriculum_history,
        {"pool_size": 3, "min_win_rate": 0.5, "avg_win_rate": 0.7},
        iteration=2,
        step=200,
        selfplay_pool_size=1,
    )

    log_data = build_train_chart_log_data(
        train_history,
        curriculum_history=curriculum_history,
    )
    dashboard = log_data["train/dashboard"].to_plotly_json()

    assert list(log_data) == ["train/dashboard"]
    assert dashboard["layout"]["autosize"] is True
    assert "height" not in dashboard["layout"]
    assert {
        "Loss",
        "unlocked pool",
        "self-play pool",
    }.issubset({trace["name"] for trace in dashboard["data"]})
    assert "Self-play pool size" in [annotation["text"] for annotation in dashboard["layout"]["annotations"]]
    assert [
        trace["showlegend"]
        for trace in dashboard["data"]
        if trace["name"] in {"unlocked pool", "self-play pool"}
    ] == [False, False]
    assert "min win rate" not in {trace["name"] for trace in dashboard["data"]}
    assert "avg win rate" not in {trace["name"] for trace in dashboard["data"]}
    assert "unlock threshold (0.75)" not in {trace["name"] for trace in dashboard["data"]}


def test_train_chart_hides_single_curriculum_pool_size_legend():
    train_history = []
    extend_train_chart_history(train_history, {"train/loss": 1.2}, step=100)
    curriculum_history = []
    extend_curriculum_chart_history(
        curriculum_history,
        {"pool_size": 3},
        iteration=2,
        step=200,
    )

    log_data = build_train_chart_log_data(
        train_history,
        curriculum_history=curriculum_history,
    )
    dashboard = log_data["train/dashboard"].to_plotly_json()

    pool_trace = next(trace for trace in dashboard["data"] if trace["name"] == "unlocked pool")
    assert pool_trace["showlegend"] is False


def test_video_log_data_includes_result_summary():
    log_data = build_video_log_data(
        [
            {
                "opponent": "builtin",
                "model_side": "p1",
                "serve": "random",
                "winner": "model",
                "score": "5-4",
                "frames": 828,
                "video": "video-placeholder",
            }
        ]
    )
    table = log_data["video/samples"]

    assert table.columns == ["opponent", "model_side", "serve", "winner", "score", "frames", "video"]
    assert table.data[0][:6] == ["builtin", "p1", "random", "model", "5-4", 828]


def test_parse_video_result_maps_player_winner_to_model_side():
    output = "Game over! Player 2 wins 0-5 (879 frames)\n"

    assert _parse_video_result(output, model_side="player_2") == {
        "winner": "model",
        "score": "0-5",
        "frames": 879,
    }
    assert _parse_video_result(output, model_side="player_1")["winner"] == "opponent"


def test_eval_dashboard_uses_dynamic_unlock_threshold():
    batch = EvalBatch(
        [
            _result(
                "player_1",
                wins=1,
                losses=1,
                winners=["player_1", "player_2"],
                p1_scores=[5, 3],
                p2_scores=[1, 5],
                game_frames=[100, 120],
            )
        ],
        iteration=7,
        step=1234,
    )

    dashboard = build_eval_chart_log_data({"p1": batch}, unlock_threshold=0.8)[
        "eval/dashboard/builtin"
    ].to_plotly_json()
    threshold_trace = next(trace for trace in dashboard["data"] if trace.get("name") == "unlock threshold (0.80)")

    assert threshold_trace["y"] == [0.8, 0.8]


def test_eval_dashboard_defaults_to_combined_when_available():
    combined_result = _result(
        "player_1",
        wins=1,
        losses=1,
        winners=["player_1", "player_2"],
        p1_scores=[5, 3],
        p2_scores=[1, 5],
        metrics={"avg_score": 4.0, "std_score": 1.0, "avg_opp_score": 3.0, "std_opp_score": 2.0},
    )
    combined_result.model_side = "both"
    combined = EvalBatch(
        [combined_result],
        iteration=0,
        step=100,
    )
    p1 = EvalBatch(
        [
            _result(
                "player_1",
                wins=1,
                losses=0,
                winners=["player_1"],
                p1_scores=[5],
                p2_scores=[1],
                metrics={"avg_score": 5.0, "std_score": 0.0, "avg_opp_score": 1.0, "std_opp_score": 0.0},
            )
        ],
        iteration=0,
        step=100,
    )
    p2 = EvalBatch(
        [
            _result(
                "player_2",
                wins=0,
                losses=1,
                winners=["player_1"],
                p1_scores=[5],
                p2_scores=[2],
                metrics={"avg_score": 2.0, "std_score": 0.0, "avg_opp_score": 5.0, "std_opp_score": 0.0},
            )
        ],
        iteration=0,
        step=100,
    )

    dashboard = build_eval_chart_log_data({"combined": combined, "p1": p1, "p2": p2})[
        "eval/dashboard/builtin"
    ].to_plotly_json()

    visible_by_group: dict[str, set[bool | str]] = {}
    for trace in dashboard["data"]:
        group = trace.get("legendgroup")
        if group:
            visible_by_group.setdefault(group, set()).add(trace.get("visible"))

    assert visible_by_group["combined"] == {True}
    assert visible_by_group["p1"] == {"legendonly"}
    assert visible_by_group["p2"] == {"legendonly"}


def test_eval_dashboard_does_not_duplicate_combined_when_combined_batch_exists():
    combined_result = _result(
        "player_1",
        wins=1,
        losses=1,
        winners=["player_1", "player_2"],
        p1_scores=[5, 3],
        p2_scores=[1, 5],
        metrics={"avg_score": 4.0, "std_score": 1.0, "avg_opp_score": 3.0, "std_opp_score": 2.0},
    )
    combined_result.model_side = "both"
    table = build_eval_chart_table(
        {
            "combined": EvalBatch([combined_result], iteration=0, step=100),
            "p1": EvalBatch(
                [
                    _result(
                        "player_1",
                        wins=1,
                        losses=0,
                        winners=["player_1"],
                        p1_scores=[5],
                        p2_scores=[1],
                        metrics={
                            "avg_score": 5.0,
                            "std_score": 0.0,
                            "avg_opp_score": 1.0,
                            "std_opp_score": 0.0,
                        },
                    )
                ],
                iteration=0,
                step=100,
            ),
            "p2": EvalBatch(
                [
                    _result(
                        "player_2",
                        wins=0,
                        losses=1,
                        winners=["player_1"],
                        p1_scores=[5],
                        p2_scores=[2],
                        metrics={
                            "avg_score": 2.0,
                            "std_score": 0.0,
                            "avg_opp_score": 5.0,
                            "std_opp_score": 0.0,
                        },
                    )
                ],
                iteration=0,
                step=100,
            ),
        }
    )

    assert [row for row in table.data if row[2] == "builtin" and row[3] == "combined" and row[4] == "win_rate"] == [
        [100, 0, "builtin", "combined", "win_rate", 0.5, None, 0.09452865480086614, 0.9054713451991339, 2, 1, 1]
    ]


def test_eval_dashboard_includes_combined_for_single_side_models():
    batch = EvalBatch(
        [
            _result(
                "player_2",
                wins=1,
                losses=0,
                winners=["player_2"],
                p1_scores=[1],
                p2_scores=[5],
                metrics={"avg_score": 5.0, "std_score": 0.0, "avg_opp_score": 1.0, "std_opp_score": 0.0},
            )
        ],
        iteration=0,
        step=100,
    )

    dashboard = build_eval_chart_log_data({"p2": batch})["eval/dashboard/builtin"].to_plotly_json()
    visible_by_group: dict[str, set[bool | str]] = {}
    for trace in dashboard["data"]:
        group = trace.get("legendgroup")
        if group:
            visible_by_group.setdefault(group, set()).add(trace.get("visible"))

    assert visible_by_group["combined"] == {True}
    assert visible_by_group["p2"] == {"legendonly"}


def test_extend_eval_chart_history_returns_cumulative_batches():
    history: dict[str, list[EvalResult]] = {}
    first = EvalBatch(
        [
            _result(
                "player_1",
                wins=1,
                losses=0,
                winners=["player_1"],
                p1_scores=[5],
                p2_scores=[1],
                metrics={"avg_score": 5.0, "std_score": 0.0, "avg_opp_score": 1.0, "std_opp_score": 0.0},
            )
        ],
        iteration=0,
        step=100,
    )
    second = EvalBatch(
        [
            _result(
                "player_1",
                wins=0,
                losses=1,
                winners=["player_2"],
                p1_scores=[2],
                p2_scores=[5],
                metrics={"avg_score": 2.0, "std_score": 0.0, "avg_opp_score": 5.0, "std_opp_score": 0.0},
            )
        ],
        iteration=1,
        step=200,
    )

    cumulative = extend_eval_chart_history(history, {"p1": first})
    assert [result.step for result in cumulative["p1"].results] == [100]

    cumulative = extend_eval_chart_history(history, {"p1": second})
    table = build_eval_chart_table(cumulative)

    assert [row[0] for row in table.data if row[3] == "combined" and row[4] == "model_score"] == [100, 200]
    assert [row[0] for row in table.data if row[3] == "p1" and row[4] == "model_score"] == [100, 200]


def test_model_won_per_game():
    summary = EvalSummary(
        wins=2,
        losses=2,
        game_winners=["player_1", "player_2", "player_1", "player_2"],
    )
    assert model_won_per_game(summary, "player_1") == [True, False, True, False]
    assert model_won_per_game(summary, "player_2") == [False, True, False, True]
