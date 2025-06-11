# analyse_4_4_1.py
"""
Собирает итоговые метрики из *_eval.csv,
строит bar-chart Accuracy / Recall по группам
и сохраняет PNG в figures/4_4_1/.
"""

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, glob, os, re

EVAL_DIR   = "experiments/results"
OUT_DIR    = "figures/4_4_1"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- 1. Собираем все eval-файлы --------
rows = []
for path in glob.glob(f"{EVAL_DIR}/*_eval.csv"):
    df = pd.read_csv(path)
    # df: одна строка (model, acc, recall, ...)
    tag = re.sub(r"\.pt$", "", df.loc[0, "model"])   # "Large_Inverse_a25"
    group, corr, alpha = tag.split("_")
    rows.append({
        "tag":   tag,
        "group": group,
        "corr":  corr,
        "alpha": int(alpha[1:]) / 100,
        "accuracy": df.loc[0, "accuracy"],
        "recall":   df.loc[0, "recall"],
        "precision":df.loc[0, "precision"],
        "value":    df.loc[0, "avg_predicted_value"]
    })

meta = pd.DataFrame(rows)
meta.to_csv(f"{OUT_DIR}/summary_raw.csv", index=False)
print(f"Собрано строк: {len(meta)}")

# -------- 2. Bar-chart Accuracy / Recall по группам --------
for metric in ["accuracy", "recall"]:
    plt.figure(figsize=(6,4))
    sns.barplot(x="group", y=metric, data=meta,  errorbar=('ci', 95))
    plt.ylim(0, 1)
    plt.title(f"{metric.capitalize()} by Group")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Group")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{metric}_by_group.png")
    plt.close()

# -------- 3. Box-plot распределения стоимости --------
plt.figure(figsize=(6,4))
sns.boxplot(x="group", y="value", data=meta)
plt.title("Average predicted value by group")
plt.ylabel("Avg predicted value (norm.)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/value_box.png")
plt.close()

print("Графики сохранены в", OUT_DIR)
