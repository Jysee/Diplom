# generate.py  (обновлённый)

from pathlib import Path
import json, random, argparse, yaml
from typing import List, Dict
import numpy as np


# ---------- 1. Генерация одного экземпляра ----------
def generate_instance(n_min: int, n_max: int,
                      w_max: int, v_max: int,
                      correlation: str = 'uncorrelated',
                      delta: int = 10,
                      alpha: float = 0.5) -> Dict:
    n = random.randint(n_min, n_max)
    weights = np.random.randint(1, w_max + 1, size=n).tolist()

    if correlation == 'uncorrelated':
        values = np.random.randint(1, v_max + 1, size=n).tolist()
    elif correlation == 'weakly':
        values = [w + random.randint(-delta, delta) for w in weights]
        values = [max(1, min(v_max, v)) for v in values]
    elif correlation == 'strongly':
        values = [w + delta for w in weights]
    elif correlation == 'inverse':
        values = [max(1, v_max - w) for w in weights]
    elif correlation == 'subset-sum':
        values = weights.copy()
    else:
        raise ValueError(f"Unknown correlation type: {correlation}")

    capacity = int(alpha * sum(weights))
    return {"weights": weights, "values": values, "capacity": capacity}


# ---------- 2. Пакетная генерация ----------
def generate_dataset(out_path: Path,
                     n_instances: int,
                     n_min: int, n_max: int,
                     w_max: int, v_max: int,
                     corr: str,
                     delta: int,
                     alpha: float) -> None:

    data = [generate_instance(n_min, n_max, w_max, v_max,
                              correlation=corr, delta=delta, alpha=alpha)
            for _ in range(n_instances)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[data] Saved {len(data)} instances → {out_path}")


# ---------- 3. CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Generate knapsack dataset")
    parser.add_argument("--config", help="YAML-config of experiment")
    # fallback quick CLI
    parser.add_argument("--out",  default="data/knapsack_dataset.json")
    parser.add_argument("--n",    type=int, default=100)
    parser.add_argument("--n_min", type=int, default=20)
    parser.add_argument("--n_max", type=int, default=40)
    parser.add_argument("--w_max", type=int, default=100)
    parser.add_argument("--v_max", type=int, default=100)
    parser.add_argument("--corr",  default="uncorrelated",
                        choices=['uncorrelated','weakly','strongly',
                                 'inverse','subset-sum'])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--delta", type=int,   default=10)
    args = parser.parse_args()

    # --- если указан YAML ---
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        tag  = cfg["tag"]
        seed = cfg.get("seed", 42)          # единый seed
        random.seed(seed); np.random.seed(seed)

        g = cfg["gen"]
        out_path = Path(f"data/{tag}.json")
        generate_dataset(out_path,
                         n_instances=g["n_instances"],
                         n_min=g["n_min"], n_max=g["n_max"],
                         w_max=g["w_max"], v_max=g["v_max"],
                         corr=g["corr"],   delta=g.get("delta", 10),
                         alpha=g["alpha"])
    else:
        # --- быстрая генерация из CLI ---
        generate_dataset(Path(args.out), args.n,
                         args.n_min, args.n_max,
                         args.w_max, args.v_max,
                         args.corr, args.delta, args.alpha)


if __name__ == "__main__":
    main()
