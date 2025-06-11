"""
Оценка обученной модели.
• Может работать либо с парой --model/--data, либо c YAML-конфигом (--config).
• Сохраняет итоговые метрики в CSV (если указан --outfile).
"""

import argparse, json, os, csv, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score

from train import KnapsackDataset, KnapsackRNN, pad_collate


# ---------- greedy-fix пост-обработка ----------------------------------------
def pack_until_full(prob: torch.Tensor, w_ratio: torch.Tensor) -> torch.Tensor:
    """
    Возвращает бинарную маску (0/1) такой же длины, что и prob,
    набирая предметы по убыванию вероятности, пока суммарный вес
    (w_ratio) не превысит 1.0.
    prob, w_ratio : shape (T,), на CPU
    """
    idx_sorted = torch.argsort(prob, descending=True)
    take = torch.zeros_like(prob, dtype=torch.float32)
    capacity_left = 1.0
    for i in idx_sorted:
        wi = float(w_ratio[i])
        if wi <= capacity_left:
            take[i] = 1.0
            capacity_left -= wi
    return take




# ---------- базовая функция --------------------------------------------------
def evaluate(model_path: str,
             dataset_path: str,
             batch_size: int = 128,
             device: str = "cpu") -> dict:
    # --- данные
    with open(dataset_path, "r") as f:
        instances = json.load(f)
    loader = DataLoader(KnapsackDataset(instances),
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=pad_collate)

    # --- модель
    device = torch.device(device)
    model = KnapsackRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_pred, all_true, all_value = [], [], []

    with torch.no_grad():
        for X, Y_true, mask in loader:
            X, Y_true, mask = X.to(device), Y_true.to(device), mask.to(device)
            Y_prob = model(X, mask)  # вероятности (B,T)
            Y_pred_list = []
            for b in range(X.size(0)):
                # канал 0 содержит w_i / C
                w_ratio = X[b, :, 0].cpu()
                prob = Y_prob[b].cpu()
                Y_pred_list.append(pack_until_full(prob, w_ratio))
            Y_pred = torch.stack(Y_pred_list).to(device)  # (B,T) бинарная маска

            # --- плоские метки (без паддинга)
            all_pred.append(Y_pred[mask].cpu().numpy())
            all_true.append(Y_true[mask].cpu().numpy())

            # --- «финансовая» ценность решения
            values_norm = X.cpu().numpy()[..., 1]          # канал 1 — v_norm
            value_pred  = (values_norm * Y_pred.cpu().numpy() * mask.cpu().numpy()) \
                               .sum(axis=1)                # сумма по предметам
            all_value.append(value_pred)

    preds = np.concatenate(all_pred)
    truths = np.concatenate(all_true)

    return dict(
        accuracy=float(accuracy_score(truths, preds)),
        recall=float(recall_score(truths, preds)),
        precision=float(precision_score(truths, preds, zero_division=0)),
        avg_predicted_value=float(np.concatenate(all_value).mean())
    )


# ---------- CLI --------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML-конфиг эксперимента")
    p.add_argument("--model", help="путь к .pt файлу")
    p.add_argument("--data",  help="путь к .json файлу")
    p.add_argument("--outfile", help="куда сохранить CSV с метриками")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # --- если передан YAML ---
    if args.config:
        cfg  = yaml.safe_load(open(args.config))
        tag  = cfg["tag"]
        model_path = args.model or f"models/{tag}.pt"
        data_path  = args.data  or f"data/{tag}.json"
    else:
        if not (args.model and args.data):
            raise SystemExit("Нужно либо --config, либо --model и --data.")
        model_path, data_path = args.model, args.data

    metrics = evaluate(model_path, data_path,
                       batch_size=args.batch_size,
                       device=args.device)

    # --- вывод
    print("\nEvaluation results:")
    for k, v in metrics.items():
        print(f"{k:>18}: {v:.4f}")

    # --- запись в CSV, если нужно
    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        write_header = not os.path.exists(args.outfile)
        with open(args.outfile, "a", newline="") as f:
            wr = csv.writer(f)
            if write_header:
                wr.writerow(["model", *metrics.keys()])
            wr.writerow([os.path.basename(model_path), *metrics.values()])


if __name__ == "__main__":
    main()
