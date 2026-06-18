import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def op_O(W, Omega):
    return (Omega * W)

def op_L(W, G2c, Omega):
    return (((-Omega) * G2c) * W)

def op_Linv(W, G2c, Omega):
    nonzero = torch.gt(G2c, 0.0)
    nonzero_f = (nonzero * 1.0)
    safe_G2c = (G2c + ((1.0 - nonzero_f) * 1.0))
    return (((W / safe_G2c) / (-Omega)) * nonzero_f)

def op_J(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    cube = torch.reshape(W, (int(s1), int(s2), int(s3),))
    spec = torch.fft.fftn(cube if isinstance(cube, torch.Tensor) else torch.tensor(float(cube)))
    flat = torch.reshape(spec, (int(n),))
    return (flat / n)

def op_I(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    cube = torch.reshape(W, (int(s1), int(s2), int(s3),))
    field = torch.fft.ifftn(cube if isinstance(cube, torch.Tensor) else torch.tensor(float(cube)))
    flat = torch.reshape(field, (int(n),))
    return (flat * n)

def op_Idag(W, active, s1, s2, s3):
    n = ((s1 * s2) * s3)
    F = op_J(W, s1, s2, s3)
    return (torch.masked_select(F, active) * n)

def op_Jdag(W, active, n, s1, s2, s3):
    W_full = torch.zeros(int(n), dtype=W.dtype).masked_scatter(active.bool(), W)
    return (op_I(W_full, s1, s2, s3) / n)

# === Program ===
s1 = 2
s2 = 2
s3 = 2
n = 8
Omega = 8.0
ms = torch.arange(int(n))
m1 = torch.remainder(torch.floor((ms / 4.0) if isinstance((ms / 4.0), torch.Tensor) else torch.tensor(float((ms / 4.0)))), 2.0)
m2 = torch.remainder(torch.floor((ms / 2.0) if isinstance((ms / 2.0), torch.Tensor) else torch.tensor(float((ms / 2.0)))), 2.0)
m3 = torch.remainder(ms, 2.0)
c = 3.141592653589793
G2 = ((c * c) * (((m1 * m1) + (m2 * m2)) + (m3 * m3)))
active = torch.le(G2, 10.0)
G2c = torch.masked_select(G2, active)
W_real = torch.stack([torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((2 + 1j), dtype=torch.complex64), torch.as_tensor((0 + 3j), dtype=torch.complex64), torch.as_tensor(((-1) + 0j), dtype=torch.complex64), torch.as_tensor((2 + 0j), dtype=torch.complex64), torch.as_tensor((1 - 1j), dtype=torch.complex64), torch.as_tensor((0 + 0j), dtype=torch.complex64), torch.as_tensor((3 + 2j), dtype=torch.complex64)])
W_spec = op_J(W_real, s1, s2, s3)
W_back = op_I(W_spec, s1, s2, s3)
physika_print(W_real)
physika_print(W_back)
physika_print(op_O(W_real, Omega))
W_active = torch.stack([torch.as_tensor((0 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64)])
LW = op_L(W_active, G2c, Omega)
LinvLW = op_Linv(LW, G2c, Omega)
physika_print(W_active)
physika_print(LW)
physika_print(LinvLW)
W_ones = torch.stack([torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64), torch.as_tensor((1 + 0j), dtype=torch.complex64)])
W_idag = op_Idag(W_ones, active, s1, s2, s3)
W_jdag = op_Jdag(W_idag, active, n, s1, s2, s3)
physika_print(W_idag)
physika_print(W_jdag)