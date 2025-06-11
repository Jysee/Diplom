# run_experiments.py
"""
Менеджер полного цикла: генерация данных → обучение → оценка →
сводная таблица.  Работает со всеми YAML-конфигами в experiments/configs/.

Usage:
    python run_experiments.py
"""

import subprocess, yaml, glob, time, os, csv, sys
from pathlib import Path

CONFIG_DIR   = Path("experiments/configs")
RESULTS_DIR  = Path("experiments/results")
LOG_DIR      = Path("experiments/logs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PY = sys.executable



summary_csv  = RESULTS_DIR / "summary.csv"

cfg_files = sorted(CONFIG_DIR.glob("*.yaml"))
if not cfg_files:
    sys.exit(f"‣ Нет файлов конфигурации в {CONFIG_DIR}")

# ---------- заголовок итоговой таблицы ----------
with summary_csv.open("w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["tag", "acc", "recall", "precision",
                 "avg_pred_val", "epochs", "train_sec"])

# ---------- проход по всем конфигам ----------
for cfg_path in cfg_files:
    cfg = yaml.safe_load(cfg_path.read_text())
    tag = cfg["tag"]
    print(f"\n================ {tag} ================")

    # 1) -- генерация данных ---------------------------------
    subprocess.run([PY, "generate.py",
                    "--config", str(cfg_path)], check=True)

    # 2) -- обучение ----------------------------------------
    t0 = time.time()
    subprocess.run([PY, "train.py",
                    "--config", str(cfg_path)],
                   check=True)
    train_sec = round(time.time() - t0, 1)

    # 3) -- оценка ------------------------------------------
    eval_out = RESULTS_DIR / f"{tag}_eval.csv"
    subprocess.run([PY, "evaluation.py",
                    "--config",  str(cfg_path),
                    "--outfile", str(eval_out)],
                   check=True)

    # 4) -- извлекаем метрики -------------------------------
    with eval_out.open() as f:
        _header, *rows = csv.reader(f)
        last_row = rows[-1]                      # берём последнюю запись
        acc, rec, prec, pval = map(float, last_row[1:])

    # 5) -- кол-во эпох (строк) ------------------------------
    epoch_csv = RESULTS_DIR / f"{tag}.csv"
    if epoch_csv.exists():
        epochs = sum(1 for _ in open(epoch_csv)) - 1  # минус заголовок
    else:
        epochs = "?"

    # 6) -- записываем в summary -----------------------------
    with summary_csv.open("a", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([tag, acc, rec, prec, pval, epochs, train_sec])

print("\n✔ Все эксперименты завершены.")
print("Сводная таблица:", summary_csv)
