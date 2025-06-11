# solve_with_rnn.py
import torch
import numpy as np
from train import KnapsackRNN           # класс модели из вашего train.py

# ---------- 1. загрузка модели ----------
DEVICE = "cpu"                          # 'cuda' если есть GPU-сборка
model_path = "models/Large_Inverse_a75.pt"

model = KnapsackRNN()                  # параметры по умолчанию
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE).eval()

# ---------- 2. подготавливаем задачу ----------
weights   = [12,  7, 11,  8,  9, 13]    # пример
values    = [24, 13, 23, 15, 16, 28]
capacity  = 35

n = len(weights)
w = np.array(weights, dtype=np.float32)
v = np.array(values,  dtype=np.float32)

x = np.stack([
        w / capacity,           # канал 0: wᵢ / C
        v / v.max(),            # канал 1: vᵢ / max(v)
        np.ones_like(w)         # канал 2: константа 1
     ], axis=1)                 # shape → (n, 3)

X = torch.from_numpy(x).unsqueeze(0).to(DEVICE)   # (1, n, 3)
mask = torch.ones(1, n, dtype=torch.bool).to(DEVICE)

# ---------- 3. инференс ----------
with torch.no_grad():
    probs = model(X, mask)[0].cpu().numpy()       # shape (n,)

# бинаризуем порогом 0.5
y_pred = (probs > 0.5).astype(int)

idx = np.argsort(-probs)
cap = capacity
y_pred = np.zeros_like(probs, dtype=int)
for i in idx:
    if weights[i] <= cap:
        y_pred[i] = 1
        cap -= weights[i]


# ---------- 4. вывод результата ----------
chosen = [i for i, flag in enumerate(y_pred) if flag]
total_w = sum(w[i] for i in chosen)
total_v = sum(v[i] for i in chosen)

print("Выбранные индексы :", chosen)
print("Суммарный вес     :", total_w, "/", capacity)
print("Суммарная ценность:", total_v)
