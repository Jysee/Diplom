# solve_with_rnn.py
"""Inference script for the RNN‑based knapsack solver.

Now also prints an *exact* 0/1‑knapsack solution obtained by classic
Dynamic Programming (DP) so that you can directly compare the quality of
RNN and optimal answers on the same instance.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from train import KnapsackRNN  # Сама модель описана в train.py

# ---------------------------- КОНФИГ ------------------------------------ #
DEVICE = "cpu"  # set to "cuda" if you have a GPU model file and suitable HW
MODEL_PATH = Path("models/Large_Weak_a50.pt")
# ------------------------------------------------------------------------ #


# ------------------------- ПРИМЕР ЗАДАЧИ ----------------------------- #
weights = [
    32, 26, 14, 16, 3, 4, 1, 9, 41, 33, 46, 26, 31, 49, 37, 32, 28, 28, 47, 14,
    41, 34, 1, 20, 43, 28, 2
]
values = [
    39, 37, 43, 9, 5, 44, 2, 28, 5, 15, 25, 22, 21, 2, 1, 7, 1, 34, 27, 33, 13,
    31, 39, 20, 24, 50, 41
]
capacity = 171
# ------------------------------------------------------------------------ #


# ---------------------------- РЕШЕНИЕ ДП ----------------------------------- #

def knapsack_dp(w: list[int] | np.ndarray, v: list[int] | np.ndarray, cap: int):
    """Classic O(n·C) DP for the 0/1 knapsack problem.

    Returns
    -------
    indices : list[int]
        Indices of selected items giving optimal value.
    total_value : int | float
        Objective value of the selected set.
    """
    n = len(w)
    w_arr = np.asarray(w, dtype=int)
    v_arr = np.asarray(v, dtype=np.float32)

    dp = np.zeros((n + 1, cap + 1), dtype=v_arr.dtype)
    keep = np.zeros_like(dp, dtype=np.int8)

    for i in range(1, n + 1):
        wi, vi = w_arr[i - 1], v_arr[i - 1]
        for c in range(cap + 1):
            if wi <= c and dp[i - 1, c - wi] + vi > dp[i - 1, c]:
                dp[i, c] = dp[i - 1, c - wi] + vi
                keep[i, c] = 1
            else:
                dp[i, c] = dp[i - 1, c]

    # back‑track
    c = cap
    chosen = []
    for i in range(n, 0, -1):
        if keep[i, c]:
            chosen.append(i - 1)  # original index
            c -= w_arr[i - 1]
    chosen.reverse()
    return chosen, dp[n, cap]


# --------------------------- РЕШЕНИЕ RNN ------------------------------ #

def solve_with_rnn(model: KnapsackRNN, w: list[float], v: list[float], cap: int):
    """Greedy sampling based on RNN output probabilities (same as before)."""
    n = len(w)
    w_arr = np.asarray(w, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)

    # prepare 3‑channel input
    x = np.stack([
        w_arr / cap,  # channel 0: w_i / C
        v_arr / v_arr.max(),  # channel 1: v_i / max(v)
        np.ones_like(w_arr)  # channel 2: constant 1
    ], axis=1)

    X = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # (1, n, 3)
    mask = torch.ones(1, n, dtype=torch.bool).to(DEVICE)

    with torch.no_grad():
        probs = model(X, mask)[0].cpu().numpy()  # (n,)

    idx = np.argsort(-probs)
    chosen = []
    cap_left = cap
    for i in idx:
        if w_arr[i] <= cap_left:
            chosen.append(i)
            cap_left -= w_arr[i]
    total_w = w_arr[chosen].sum()
    total_v = v_arr[chosen].sum()
    return chosen, total_w, total_v


# ------------------------------ main ------------------------------------ #

def main():
    # 1. Загрузка модели
    model = KnapsackRNN()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    # 2. RNN решение
    t0 = time.perf_counter()
    rnn_idx, rnn_w, rnn_v = solve_with_rnn(model, weights, values, capacity)
    rnn_idx = [int(i) for i in rnn_idx]

    # 3. DP оптимальное решение
    t0 = time.perf_counter()
    dp_idx, dp_v = knapsack_dp(weights, values, capacity)
    dp_w = sum(weights[i] for i in dp_idx)

    # 4. Вывод
    print("\n==== RNN решение ====")
    print("Выбранные индексы:", rnn_idx)
    print(f"Общий вес    : {rnn_w:.0f} / {capacity}")
    print(f"Общая ценность     : {rnn_v:.0f}")

    print("\n==== DP оптимальное решение ====")
    print("Выбранные индексы:", dp_idx)
    print(f"Общий вес    : {dp_w} / {capacity}")
    print(f"Общая ценность     : {dp_v:.0f}")

    # 5. Насколько «хуже»  решение, полученное RNN, по сравнению с оптимальным, вычисленным динамическим программированием
    if dp_v > 0:
        gap = 100.0 * (dp_v - rnn_v) / dp_v
        print(f"\nOptimality gap  : {gap:.2f}%")


if __name__ == "__main__":
    main()
