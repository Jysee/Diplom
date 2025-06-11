# ---------- train.py (финальная версия для модуля 4.2.2) ----------
# Особенности:
#   • Генерация «золотых» меток через точный DP-решатель 0/1-рюкзака;
#   • LSTM-модель с параметрами (hidden=256, layers=2, dropout=0.3);
#   • AdamW (lr=3·10⁻⁴, weight_decay=5·10⁻⁴) + CosineAnnealingWarmRestarts;
#   • Клиппинг градиента, TensorBoard-логирование, ранняя остановка;
#   • Автоматическое создание каталогов models/ и runs/.
# ------------------------------------------------------------------
from pathlib import Path
import json, math, csv, os
import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# --- точный DP-решатель для 0/1 KP (нужен для «золотой» маски) ---
def solve_knapsack_dp(w, v, C):
    n = len(w)
    dp = [0]*(C+1); keep = [[0]*(C+1) for _ in range(n)]
    for i,(wi,vi) in enumerate(zip(w,v)):
        for c in range(C, wi-1, -1):
            if dp[c-wi] + vi > dp[c]:
                dp[c] = dp[c-wi] + vi
                keep[i][c] = 1
    mask = np.zeros(n, dtype=np.float32); c=C
    for i in range(n-1, -1, -1):
        if keep[i][c]:
            mask[i] = 1; c -= w[i]
    return mask


# ---------------- Dataset & collate_fn ----------------
class KnapsackDataset(Dataset):
    def __init__(self, instances):
        self.items = []
        for inst in instances:
            w = np.array(inst["weights"], dtype=np.int32)
            v = np.array(inst["values"],  dtype=np.int32)
            cap = int(inst["capacity"])
            y = solve_knapsack_dp(w, v, cap)              # оптимальная маска
            # канал 0:  вес / capacity    (0…1)
            # канал 1:  ценность / max(v) (0…1)
            # канал 2:  «константа 1» — маркер доступной ёмкости
            w_ratio = w.astype(np.float32) / cap
            v_ratio = v.astype(np.float32) / v.max()
            cap_feat = np.ones_like(w_ratio, dtype=np.float32)
            x = np.stack([w_ratio, v_ratio, cap_feat], 1)  # (n, 3)

            self.items.append((x.astype(np.float32), y))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def pad_collate(batch):
    lengths = [len(x) for x,_ in batch]; L = max(lengths)
    X = torch.zeros(len(batch), L, 3);  Y = torch.zeros(len(batch), L)
    M = torch.zeros(len(batch), L, dtype=torch.bool)
    for i,(x,y) in enumerate(batch):
        n=len(x); X[i,:n]=torch.from_numpy(x); Y[i,:n]=torch.from_numpy(y); M[i,:n]=1
    return X, Y, M


# ------------------ Модель ------------------
class KnapsackRNN(nn.Module):
    def __init__(self, inp=3, hid=256, layers=2, drop=0.3):
        super().__init__()
        self.rnn = nn.LSTM(inp, hid, layers, batch_first=True, dropout=drop)
        self.fc  = nn.Linear(hid, 1)

    def forward(self, x, mask):
        h,_ = self.rnn(x)                         # (B,T,H)
        logits = self.fc(h).squeeze(-1)           # (B,T)
        prob   = torch.sigmoid(logits) * mask.float()
        return prob


# ------------------ Тренировка ------------------
def train():
    # ---------- ЧТЕНИЕ YAML-КОНФИГА ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="путь к YAML-файлу эксперимента")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tag = cfg["tag"]  # Small_Uncorrelated_a25
    seed = cfg["seed"]  # 42
    gcfg = cfg["gen"]  # блок генерации (для путей)
    mcfg = cfg["model"]  # layers, hidden, dropout
    tcfg = cfg["train"]  # lr, batch_size, patience

    # ---------- ФИКСИРУЕМ SIT ----------
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_p = Path(f"data/{tag}.json")  # >>> REPLACE
    inst = json.loads(data_p.read_text())
    tr, te = train_test_split(inst, test_size=0.15, random_state=42)
    tr, va = train_test_split(tr,  test_size=0.1765, random_state=42)  # 70/15/15

    dl = lambda xs,shuf: DataLoader(KnapsackDataset(xs), tcfg["batch_size"], shuffle=shuf,
                                    collate_fn=pad_collate, num_workers=2)
    train_loader, val_loader = dl(tr,True), dl(va,False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = KnapsackRNN(
        inp=3,
        hid=mcfg["hidden"],
        layers=mcfg["layers"],
        drop=mcfg["dropout"]).to(dev)  # >>> REPLACE дефолты

    opt = optim.AdamW(net.parameters(), lr=tcfg["lr"], weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
             T_0=len(train_loader)*5, T_mult=2)
    loss_fn = nn.BCELoss(reduction="none")

    writer = SummaryWriter(f"runs/{tag}")
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)

    best, patience, wait = math.inf, tcfg["patience"], 0
    for epoch in range(1, tcfg.get("epochs", 200) + 1):
        # ---- train ----
        net.train(); tl=0
        for step,(X,Y,M) in enumerate(train_loader):
            X,Y,M = X.to(dev),Y.to(dev),M.to(dev)
            opt.zero_grad()
            pred = net(X,M)
            loss = (loss_fn(pred,Y)*M.float()).sum()/M.sum()
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0)
            opt.step(); sched.step(epoch-1+step/len(train_loader))
            tl += loss.item()*X.size(0)
        tl /= len(train_loader.dataset)

        # ---- val ----
        net.eval(); vl=0
        with torch.no_grad():
            for X,Y,M in val_loader:
                X,Y,M = X.to(dev),Y.to(dev),M.to(dev)
                pred = net(X,M)
                loss = (loss_fn(pred,Y)*M.float()).sum()/M.sum()
                vl += loss.item()*X.size(0)
        vl /= len(val_loader.dataset)

        writer.add_scalars("Loss",{"train":tl,"val":vl},epoch)
        print(f"Epoch {epoch:03d} | train {tl:.4f} | val {vl:.4f}")

        os.makedirs("results", exist_ok=True)
        csv_path = f"results/{tag}.csv"
        write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="") as f:
            wr = csv.writer(f)
            if write_header:
                wr.writerow(["epoch", "train_loss", "val_loss"])
            wr.writerow([epoch, tl, vl])


        if vl < best - 1e-4:
            best, wait = vl, 0
            torch.save(net.state_dict(), models_dir/f"{tag}.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping."); break
    writer.close()




if __name__ == "__main__":
    train()
