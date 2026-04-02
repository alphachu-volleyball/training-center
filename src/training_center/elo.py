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