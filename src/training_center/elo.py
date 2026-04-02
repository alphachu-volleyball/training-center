"""ELO rating calculation using Bradley-Terry MLE (batch, order-independent)."""

from __future__ import annotations

import numpy as np

INITIAL_ELO = 1500


def compute_elo(
    results: dict[tuple[str, str], tuple[int, int]],
    base_elo: float = INITIAL_ELO,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute ELO ratings from pairwise win counts using Bradley-Terry MLE.

    Args:
        results: {(player_a, player_b): (wins_a, wins_b), ...}
        base_elo: ELO assigned to geometric-mean strength (default 1500).
        max_iter: Maximum iterations for convergence.
        tol: Convergence tolerance (max relative change in strengths).

    Returns:
        {player_name: elo_rating, ...} — order-independent ratings.
    """
    # Collect players
    players: list[str] = []
    seen: set[str] = set()
    for a, b in results:
        for p in (a, b):
            if p not in seen:
                players.append(p)
                seen.add(p)

    if len(players) < 2:
        return {p: base_elo for p in players}

    idx = {p: i for i, p in enumerate(players)}
    n = len(players)

    # Total wins per player and pairwise game counts
    wins = np.zeros(n)
    # Pairs: list of (i, j, n_ij) where n_ij = total games between i and j
    pairs: list[tuple[int, int, int]] = []
    for (a, b), (wa, wb) in results.items():
        i, j = idx[a], idx[b]
        wins[i] += wa
        wins[j] += wb
        total = wa + wb
        if total > 0:
            pairs.append((i, j, total))

    # Bradley-Terry iterative update
    gamma = np.ones(n)
    for _ in range(max_iter):
        gamma_new = np.zeros(n)
        for i, j, n_ij in pairs:
            denom = gamma[i] + gamma[j]
            gamma_new[i] += n_ij / denom
            gamma_new[j] += n_ij / denom

        # Avoid division by zero for players with no games
        mask = gamma_new > 0
        gamma_new[mask] = wins[mask] / gamma_new[mask]
        gamma_new[~mask] = gamma[~mask]

        # Normalize to geometric mean = 1
        gamma_new /= np.exp(np.mean(np.log(np.maximum(gamma_new, 1e-300))))

        # Check convergence
        rel_change = np.max(np.abs(gamma_new - gamma) / np.maximum(gamma, 1e-300))
        gamma = gamma_new
        if rel_change < tol:
            break

    # Convert to ELO scale: elo = 400 * log10(gamma) + base_elo
    elos = 400 * np.log10(np.maximum(gamma, 1e-300)) + base_elo

    return {p: float(elos[idx[p]]) for p in players}


def _load_records(path: str) -> list[dict]:
    """Load records from CSV or JSON file."""
    import csv
    import json

    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "columns" in data and "data" in data:
            # W&B table format
            return [dict(zip(data["columns"], row)) for row in data["data"]]
        return data
    else:
        with open(path) as f:
            return list(csv.DictReader(f))


def main() -> None:
    import argparse
    import csv
    import sys

    parser = argparse.ArgumentParser(description="Compute ELO ratings from pairwise results (CSV/JSON)")
    parser.add_argument("input", help="Input file (CSV or JSON)")
    parser.add_argument("--p1", required=True, help="Column name for player 1")
    parser.add_argument("--p2", required=True, help="Column name for player 2")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--p1-wins", help="Column name for player 1 win count (requires --p2-wins)")
    group.add_argument("--win-rate", help="Column name for player 1 win rate (requires --games)")

    parser.add_argument("--p2-wins", help="Column name for player 2 win count")
    parser.add_argument("--games", type=int, help="Games per pair (used with --win-rate)")
    parser.add_argument("-o", "--output", help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    if args.p1_wins and not args.p2_wins:
        parser.error("--p1-wins requires --p2-wins")
    if args.win_rate and not args.games:
        parser.error("--win-rate requires --games")

    records = _load_records(args.input)

    win_counts: dict[tuple[str, str], tuple[int, int]] = {}
    for row in records:
        p1, p2 = row[args.p1], row[args.p2]
        if p1 == p2:
            continue
        if args.p1_wins:
            w1, w2 = int(row[args.p1_wins]), int(row[args.p2_wins])
        else:
            rate = float(row[args.win_rate])
            w1 = round(rate * args.games)
            w2 = args.games - w1
        win_counts[(p1, p2)] = (w1, w2)

    elos = compute_elo(win_counts)

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.writer(out)
    writer.writerow(["agent", "elo"])
    for name, elo in sorted(elos.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([name, f"{elo:.1f}"])
    if args.output:
        out.close()
        print(f"Wrote {len(elos)} ratings to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()