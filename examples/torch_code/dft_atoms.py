import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def fold_freq(m, q):
    return (m - (torch.gt(m, (q / 2)) * q))

def structure_factor(nn1, nn2, nn3, cc, qx, qy, qz):
    phase = (cc * (((nn1 * qx) + (nn2 * qy)) + (nn3 * qz)))
    return torch.exp(((-torch.tensor(1j)) * phase) if isinstance(((-torch.tensor(1j)) * phase), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * phase))))

# === Program ===
a = 16.0
ecut = 16.0
s1 = 60
s2 = 60
s3 = 60
n = ((s1 * s2) * s3)
px = 0.0
py = 0.0
pz = 0.0
ms = torch.arange(int(n))
m1 = torch.remainder(torch.floor((ms / (s3 * s2)) if isinstance((ms / (s3 * s2)), torch.Tensor) else torch.tensor(float((ms / (s3 * s2))))), s1)
m2 = torch.remainder(torch.floor((ms / s3) if isinstance((ms / s3), torch.Tensor) else torch.tensor(float((ms / s3)))), s2)
m3 = torch.remainder(ms, s3)
n1 = fold_freq(m1, s1)
n2 = fold_freq(m2, s2)
n3 = fold_freq(m3, s3)
c = ((2 * 3.141592653589793) / a)
G2 = ((c * c) * (((n1 * n1) + (n2 * n2)) + (n3 * n3)))
active = torch.le(G2, (2 * ecut))
G2c = torch.masked_select(G2, active)
Sf = structure_factor(n1, n2, n3, c, px, py, pz)
Omega = ((a * a) * a)
count = torch.sum(active if isinstance(active, torch.Tensor) else torch.tensor(float(active)))
physika_print(n)
physika_print(Omega)
physika_print(count)
physika_print(G2[int(0)])
physika_print(Sf[int(0)])