"""Geometry-conditioned evidential quality-control network, plus the ablation variants.

Variants are selected with two switches:
  geom : "embed32" (paper), "embed8", "onehot", "none"
  head : "beta"  -> 2 outputs, alpha = softplus(.)+1, beta = softplus(.)+1   (paper)
         "bce"   -> 1 logit, ordinary sigmoid classifier (conventional baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

N_GEOM = 4


class QCNet(nn.Module):
    def __init__(self, geom="embed32", head="beta", dropout=0.3, pretrained=True, freeze_backbone=False):
        super().__init__()
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=w)
        self.backbone.fc = nn.Identity()  # 512-d global-average-pooled features
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

        self.geom, self.head = geom, head
        if geom.startswith("embed"):
            dim = int(geom[5:])
            self.embed = nn.Embedding(N_GEOM, dim)
        elif geom == "onehot":
            dim = N_GEOM
        elif geom == "none":
            dim = 0
        else:
            raise ValueError(geom)

        self.mlp = nn.Sequential(
            nn.Linear(512 + dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(256, 2 if head == "beta" else 1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:  # keep frozen BatchNorm statistics fixed too
            self.backbone.eval()
        return self

    def forward(self, x, g):
        v = self.backbone(x)
        if self.geom.startswith("embed"):
            z = torch.cat([v, self.embed(g)], 1)
        elif self.geom == "onehot":
            z = torch.cat([v, F.one_hot(g, N_GEOM).float()], 1)
        else:
            z = v
        o = self.out(self.mlp(z))
        if self.head == "beta":
            ab = F.softplus(o) + 1.0
            return ab[:, 0], ab[:, 1]          # alpha, beta
        return o[:, 0]                          # logit


def predict(output, head):
    """-> (p_good, uncertainty or None, alpha or None, beta or None)"""
    if head == "beta":
        a, b = output
        s = a + b
        return a / s, 2.0 / s, a, b
    return torch.sigmoid(output), None, None, None


# ---------------------------------------------------------------- losses

def beta_bernoulli_nll(a, b, y):
    """-log p(y | alpha, beta) with p(y=1) = E[theta] = alpha / (alpha + beta).

    NOTE: depends only on the ratio alpha/beta. Scaling both by any constant leaves the loss
    unchanged, so on its own it does not train the total evidence alpha+beta (i.e. u = 2/(alpha+beta)).
    """
    s = a + b
    return -(y * torch.log(a / s) + (1 - y) * torch.log(b / s)).mean()


def kl_to_uniform(a, b):
    """KL( Beta(a,b) || Beta(1,1) ) = -lnB(a,b) + (a-1)psi(a) + (b-1)psi(b) - (a+b-2)psi(a+b)."""
    s = a + b
    return (torch.lgamma(s) - torch.lgamma(a) - torch.lgamma(b)
            + (a - 1) * torch.digamma(a) + (b - 1) * torch.digamma(b) - (s - 2) * torch.digamma(s))


def evidential_kl(a, b, y):
    """Sensoy et al. (2018) regulariser, binary case: remove the evidence for the TRUE class and
    penalise any remaining (misleading) evidence by pulling it toward the uniform Beta(1,1)."""
    a_t = y + (1 - y) * a          # y=1: alpha -> 1 (correct-class evidence removed), keep beta
    b_t = (1 - y) + y * b          # y=0: beta  -> 1, keep alpha
    return kl_to_uniform(a_t, b_t).mean()


def loss_fn(output, y, head, kl_weight=0.0):
    if head == "beta":
        a, b = output
        loss = beta_bernoulli_nll(a, b, y)
        if kl_weight > 0:
            loss = loss + kl_weight * evidential_kl(a, b, y)
        return loss
    return F.binary_cross_entropy_with_logits(output, y)


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
