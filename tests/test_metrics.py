"""Tests for eval metrics computation."""

from pika_zoo.records.types import FrameRecord, GameRecord, GamesRecord, RoundRecord

from training_center.metrics import compute_eval_metrics


def _frame(action1: int = 0, action2: int = 0, ball_x: int = 100, **kwargs) -> FrameRecord:
    defaults = dict(
        frame=1,
        round_number=1,
        player1_action=action1,
        player1_x=50,
        player1_y=244,
        player1_state=0,
        player2_action=action2,
        player2_x=350,
        player2_y=244,
        player2_state=0,
        ball_x=ball_x,
        ball_y=100,
        ball_x_velocity=0,
        ball_y_velocity=0,
        ball_is_power_hit=False,
    )
    defaults.update(kwargs)
    return FrameRecord(**defaults)


def _round(frames: list[FrameRecord], server: str = "player_1", scorer: str = "player_1") -> RoundRecord:
    return RoundRecord(
        round_number=1,
        server=server,
        scorer=scorer,
        reward={"player_1": 1.0, "player_2": -1.0},
        start_frame=1,
        end_frame=len(frames),
        frames=frames,
    )


def test_action_entropy_uniform():
    """Uniform distribution over 4 actions → entropy = 2.0 bits."""
    frames = [_frame(action1=i % 4) for i in range(100)]
    game = GameRecord(num_frames=100, rounds=[_round(frames)])
    m = compute_eval_metrics(GamesRecord(games=[game]), "player_1")
    assert abs(m["action_entropy"] - 2.0) < 0.01


def test_action_entropy_single():
    """Single action → entropy = 0."""
    frames = [_frame(action1=0) for _ in range(50)]
    game = GameRecord(num_frames=50, rounds=[_round(frames)])
    m = compute_eval_metrics(GamesRecord(games=[game]), "player_1")
    assert m["action_entropy"] == 0.0


def test_power_hit_rate():
    frames = [
        _frame(p1_touch_ball=True, p1_power_hit=True),
        _frame(p1_touch_ball=True, p1_power_hit=False),
        _frame(p1_touch_ball=True, p1_power_hit=True),
        _frame(),
    ]
    game = GameRecord(num_frames=4, rounds=[_round(frames)])
    m = compute_eval_metrics(GamesRecord(games=[game]), "player_1")
    assert abs(m["power_hit_rate"] - 2 / 3) < 1e-9


def test_ball_own_side_ratio():
    # player_1 side: ball_x < 216
    frames = [
        _frame(ball_x=100),  # own side
        _frame(ball_x=300),  # opponent side
        _frame(ball_x=100),  # own side
        _frame(ball_x=100),  # own side
    ]
    game = GameRecord(num_frames=4, rounds=[_round(frames)])
    m = compute_eval_metrics(GamesRecord(games=[game]), "player_1")
    assert abs(m["ball_own_side_ratio"] - 0.75) < 1e-9


def test_std_round_frames():
    r1 = _round([_frame() for _ in range(10)], server="player_1")
    r1.start_frame = 1
    r1.end_frame = 10
    r2 = _round([_frame() for _ in range(20)], server="player_2")
    r2.start_frame = 11
    r2.end_frame = 30
    game = GameRecord(num_frames=30, rounds=[r1, r2])
    m = compute_eval_metrics(GamesRecord(games=[game]), "player_1")
    # mean=15, std = sqrt(((10-15)^2 + (20-15)^2)/2) = sqrt(25) = 5
    assert abs(m["std_round_frames"] - 5.0) < 1e-9
    assert abs(m["serve_avg_round_frames"] - 10.0) < 1e-9
    assert abs(m["receive_avg_round_frames"] - 20.0) < 1e-9
