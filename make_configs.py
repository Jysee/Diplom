# make_configs.py
"""
Генерирует 45 YAML-файлов с конфигурациями экспериментов:
  - три диапазона размера (Small / Medium / Large)
  - пять типов корреляции
  - три значения плотности alpha
Каждый файл кладётся в experiments/configs/
"""
import yaml, os, itertools

os.makedirs("experiments/configs", exist_ok=True)

groups = {
    "Small":  dict(n_min=20,  n_max=40,  w_max=100, v_max=100),
    "Medium": dict(n_min=60,  n_max=100, w_max=100, v_max=100),
    "Large":  dict(n_min=150, n_max=250, w_max=50,  v_max=100),  # w_max=50 по плану
}

corr_types = {
    "Uncorrelated": "uncorrelated",
    "Weak":         "weakly",
    "Strong":       "strongly",
    "Inverse":      "inverse",
    "SubsetSum":    "subset-sum",
}

alphas = [0.25, 0.50, 0.75]

# базовые гиперпараметры модели/обучения
model_defaults = dict(layers=2, hidden=256, dropout=0.3)
train_defaults = dict(lr=3e-4, batch_size=32, patience=15)

for g_name, c_name, a in itertools.product(groups, corr_types, alphas):
    tag = f"{g_name}_{c_name}_a{int(a*100):02d}"
    cfg = {
        "tag": tag,
        "seed": 42,

        # параметры генератора
        "gen": {
            **groups[g_name],
            "corr": corr_types[c_name],
            "alpha": a,
            "n_instances": 100        # 100 экземпляров на комбинацию
        },

        # модель и обучение
        "model": model_defaults,
        "train": train_defaults
    }

    out_path = f"experiments/configs/{tag}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"Создан {out_path}")
