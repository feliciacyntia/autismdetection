"""E2 — fusion citra 3D + tabel: backbone = CNN 3D+SE full_v2 (lapisan sama dengan pilot_models.Scratch3DSE).

Mode: image (tanpa tabel, juga dasar late fusion), concat (embedding + tabel -> head), film (Perez 2018), daft (Wolf 2022).
FiLM/DAFT dipasang pada masukan stage terakhir. Implementasi sendiri dari paper, bukan salinan kode ai-med/DAFT (GPL-3.0).
Modulasi: x * (1 + a) + b agar mulai dekat identitas pada data kecil.
Backbone dibuat lebih dulu sehingga inisialisasinya identik antarmode pada seed yang sama.
"""
import torch
import torch.nn as nn

from pilot_models import Head, SE3d


def _stage(cin, c, se_ratio):
    return [nn.Conv3d(cin, c, 3, padding=1, bias=False), nn.BatchNorm3d(c), nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False), nn.BatchNorm3d(c), nn.ReLU(inplace=True), SE3d(c, se_ratio)]


class Fusion3D(nn.Module):
    def __init__(self, channels, se_ratio, dropout, mode, d_tab, bottleneck=7):
        super().__init__()
        assert mode in ("image", "concat", "film", "daft"), mode
        self.mode = mode
        layers, cin = [], 1
        for c in channels[:-1]:
            layers += _stage(cin, c, se_ratio) + [nn.MaxPool3d(2)]
            cin = c
        self.early = nn.Sequential(*layers)
        self.last = nn.Sequential(*_stage(cin, channels[-1], se_ratio))
        d_mod = {"film": d_tab, "daft": cin + d_tab}.get(mode)
        if d_mod is not None:
            h = max(d_mod // bottleneck, 8)
            self.mod = nn.Sequential(nn.Linear(d_mod, h), nn.ReLU(inplace=True), nn.Linear(h, 2 * cin))
        self.head = Head(channels[-1] + (d_tab if mode == "concat" else 0), dropout=dropout)

    def forward(self, x):
        v, t = x
        f = self.early(v)
        if self.mode in ("film", "daft"):
            inp = t if self.mode == "film" else torch.cat([f.mean(dim=(2, 3, 4)), t], 1)
            a, b = self.mod(inp).chunk(2, 1)
            f = f * (1 + a[:, :, None, None, None]) + b[:, :, None, None, None]
        f = self.last(f).mean(dim=(2, 3, 4))
        if self.mode == "concat":
            f = torch.cat([f, t], 1)
        e = self.head.emb(f)
        return self.head.out(e).squeeze(1), e

    def param_groups(self, hp):
        return [{"params": list(self.parameters()), "lr": hp["lr"]}]
