import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def coulomb(Z_nuc, G2, Sf, s1, s2, s3):
    pi = 3.141592653589793
    nonzero = torch.gt(G2, 0.0)
    nonzero_f = (nonzero * 1.0)
    safe_G2 = (G2 + ((1.0 - nonzero_f) * 1.0))
    Vcoul = (((((-4.0) * pi) * Z_nuc) / safe_G2) * nonzero_f)
    return op_J((Vcoul * Sf), s1, s2, s3)

def op_J(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    cube = torch.reshape(W, (int(s1), int(s2), int(s3),))
    spec = torch.fft.fftn(cube if isinstance(cube, torch.Tensor) else torch.tensor(float(cube)))
    flat = torch.reshape(spec, (int(n),))
    return (flat / n)

# === Program ===
s1 = 2
s2 = 2
s3 = 2
n = 8
ms = torch.arange(int(n))
m1 = torch.remainder(torch.floor((ms / 4.0) if isinstance((ms / 4.0), torch.Tensor) else torch.tensor(float((ms / 4.0)))), 2.0)
m2 = torch.remainder(torch.floor((ms / 2.0) if isinstance((ms / 2.0), torch.Tensor) else torch.tensor(float((ms / 2.0)))), 2.0)
m3 = torch.remainder(ms, 2.0)
pi = 3.141592653589793
G2 = ((pi * pi) * (((m1 * m1) + (m2 * m2)) + (m3 * m3)))
Sf = torch.stack([torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64)])
Vcoul_g = coulomb(1.0, G2, Sf, s1, s2, s3)
physika_print(Vcoul_g)