"""Train / evaluate one configuration on one split. Saves per-image predictions so every
reported number can be recomputed (see metrics.py / analyze.py).

Normally called through run_experiments.py, not directly.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # in case an op (e.g. digamma) lacks an MPS kernel

import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from model import QCNet, loss_fn, n_trainable, predict

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------- augmentation (Section 2.5.7)

class Rot90:
    def __call__(self, x):
        return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))


class MotionBlur:
    def __init__(self, kmin=3, kmax=7):
        self.kmin, self.kmax = kmin, kmax

    def __call__(self, x):
        k = random.choice(range(self.kmin, self.kmax + 1, 2))
        ker = torch.zeros(k, k)
        if random.random() < 0.5:
            ker[k // 2, :] = 1.0 / k
        else:
            ker[:, k // 2] = 1.0 / k
        ker = ker.expand(x.shape[0], 1, k, k).contiguous()
        return F.conv2d(x.unsqueeze(0), ker, padding=k // 2, groups=x.shape[0]).squeeze(0)


class GaussNoise:
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma


class MultNoise:
    def __init__(self, lo=0.8, hi=1.2):
        self.lo, self.hi = lo, hi

    def __call__(self, x):
        return x * torch.empty_like(x).uniform_(self.lo, self.hi)


class Apply:
    """Apply fn with probability p (plain callable; avoids container-type quirks)."""
    def __init__(self, fn, p):
        self.fn, self.p = fn, p

    def __call__(self, x):
        return self.fn(x) if random.random() < self.p else x


class Compose:
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, x):
        for f in self.fns:
            x = f(x)
        return x


def build_transforms(train):
    base = [TF.to_tensor, T.Resize((224, 224), antialias=True)]          # PIL -> float [0,1] tensor
    norm = [lambda x: x.clamp(0, 1), T.Normalize(MEAN, STD)]
    if not train:
        return Compose(base + norm)
    # Augmentation probabilities used for all experiments.
    aug = [
        Rot90(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        Apply(T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), 0.5),
        Apply(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15), 0.5),
        Apply(T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)), 0.2),
        Apply(MotionBlur(3, 7), 0.1),
        Apply(GaussNoise(0.02), 0.2),
        Apply(MultNoise(0.8, 1.2), 0.2),
    ]
    return Compose(base + aug + norm)


class ScaffoldDS(Dataset):
    def __init__(self, df, root, train, harmonize):
        self.df = df.reset_index(drop=True)
        self.tf = build_transforms(train)
        self.imgs = []
        for p in self.df.path:
            im = Image.open(os.path.join(root, p)).convert("RGB")
            if harmonize and im.size != (256, 256):
                # 108+72 square-grid images are 512 px, all others 256 px. Bring everything to a common
                # resolution first so image scale/sharpness cannot act as a shortcut for the label.
                im = im.resize((256, 256), Image.BOX)
            self.imgs.append(im)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(self.imgs[i]), int(r.geometry_id), torch.tensor(float(r.label), dtype=torch.float32)


def balanced_sampler(df):
    """Section 2.5.8: batches balanced over geometry x quality combinations."""
    key = df.geometry_id.astype(str) + "_" + df.label.astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df), replacement=True)


@torch.no_grad()
def run_eval(model, loader, cfg, device):
    model.eval()
    ps, us, als, bes, ys, losses = [], [], [], [], [], []
    for x, g, y in loader:
        x, g, y = x.to(device), g.to(device), y.float().to(device)
        out = model(x, g)
        losses.append(loss_fn(out, y, cfg["head"]).item() * len(y))
        p, u, a, b = predict(out, cfg["head"])
        ps.append(p.cpu()); ys.append(y.cpu())
        if u is not None:
            us.append(u.cpu()); als.append(a.cpu()); bes.append(b.cpu())
    cat = lambda z: torch.cat(z).numpy() if z else None
    y, p = cat(ys), cat(ps)
    return dict(loss=sum(losses) / len(y), y=y, p=p, u=cat(us), alpha=cat(als), beta=cat(bes))


def train_one(df, root, train_idx, sel_idx, eval_idx, cfg, out_dir, seed=42):
    """train on train_idx, select checkpoint / drive LR schedule on sel_idx, report on eval_idx."""
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = get_device()
    hz = cfg.get("harmonize", True)
    dtr = df.iloc[train_idx]
    ds_tr = ScaffoldDS(dtr, root, True, hz)
    ds_sel = ScaffoldDS(df.iloc[sel_idx], root, False, hz)
    ds_ev = ds_sel if list(sel_idx) == list(eval_idx) else ScaffoldDS(df.iloc[eval_idx], root, False, hz)
    bs = cfg.get("batch_size", 32)
    dl_tr = DataLoader(ds_tr, batch_size=bs, sampler=balanced_sampler(dtr), num_workers=0)
    dl_sel = DataLoader(ds_sel, batch_size=64, shuffle=False, num_workers=0)
    dl_ev = DataLoader(ds_ev, batch_size=64, shuffle=False, num_workers=0)

    model = QCNet(geom=cfg["geom"], head=cfg["head"], dropout=cfg.get("dropout", 0.3),
                  freeze_backbone=cfg.get("freeze", False)).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, min_lr=1e-6)

    epochs = cfg.get("epochs", 50)
    kl_w, kl_anneal = cfg.get("kl_weight", 0.0), cfg.get("kl_anneal_epochs", 10)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        lam = kl_w * min(1.0, ep / kl_anneal) if kl_w > 0 else 0.0
        tl, n, tp, ty = 0.0, 0, [], []
        for x, g, y in dl_tr:
            x, g, y = x.to(device), g.to(device), y.float().to(device)
            out = model(x, g)
            loss = loss_fn(out, y, cfg["head"], lam)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(y); n += len(y)
            tp.append(predict(out, cfg["head"])[0].detach().cpu()); ty.append(y.cpu())
        tp, ty = torch.cat(tp).numpy(), torch.cat(ty).numpy()
        ev = run_eval(model, dl_sel, cfg, device)
        sched.step(ev["loss"])
        h = dict(epoch=ep, lr=opt.param_groups[0]["lr"], kl_lambda=lam,
                 train_loss=tl / n, train_acc=float(((tp >= .5) == ty).mean()),
                 train_auc=roc_auc_score(ty, tp) if 0 < ty.sum() < len(ty) else np.nan,
                 sel_loss=ev["loss"], sel_acc=float(((ev["p"] >= .5) == ev["y"]).mean()),
                 sel_auc=roc_auc_score(ev["y"], ev["p"]) if 0 < ev["y"].sum() < len(ev["y"]) else np.nan)
        hist.append(h)
        if ev["loss"] < best:
            best, best_state, h["best"] = ev["loss"], copy.deepcopy(model.state_dict()), True
        print(f"  ep {ep:3d} train_loss {h['train_loss']:.4f} sel_loss {h['sel_loss']:.4f} "
              f"sel_acc {h['sel_acc']:.3f} sel_auc {h['sel_auc']:.3f} lr {h['lr']:.1e} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    ev = run_eval(model, dl_ev, cfg, device)
    d = df.iloc[eval_idx].reset_index(drop=True)
    pred = d[["path", "label", "geometry", "tip", "prefix", "print_id", "session"]].copy()
    pred["p"] = ev["p"]
    for k in ["u", "alpha", "beta"]:
        if ev[k] is not None:
            pred[k] = ev[k]
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    best_ep = int(pd.DataFrame(hist).sel_loss.idxmin()) + 1
    meta = dict(cfg=cfg, seed=seed, device=str(device), n_train=len(train_idx), n_sel=len(sel_idx),
                n_eval=len(eval_idx), best_epoch=best_ep, trainable_params=n_trainable(model),
                total_params=sum(p.numel() for p in model.parameters()),
                torch=torch.__version__, minutes=(time.time() - t0) / 60)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    pred.to_csv(os.path.join(out_dir, "preds.csv"), index=False)  # written last = run complete
    return pred
