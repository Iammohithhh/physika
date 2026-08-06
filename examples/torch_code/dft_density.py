import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def mask_embed(values, mask, n):
    empty = (torch.arange(n) * 0j)
    return torch.masked_scatter(empty, mask, values)

def vec_size(x):
    return len(x)

def orth(atoms, W):
    norm = torch.sqrt((atoms.volume() * torch.sum((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0) if isinstance((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0))))) if isinstance((atoms.volume() * torch.sum((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0) if isinstance((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0))))), torch.Tensor) else torch.tensor(float((atoms.volume() * torch.sum((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0) if isinstance((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(W if isinstance(W, torch.Tensor) else torch.tensor(float(W))) ** 2.0))))))))
    return (W / norm)

def get_n_total(atoms, Y):
    active = atoms.active()
    n_full = vec_size(active)
    Y_full = mask_embed(Y, active, n_full)
    Yrs = op_I(atoms, Y_full)
    f0 = atoms.f[int(0)]
    return (f0 * (torch.abs(Yrs if isinstance(Yrs, torch.Tensor) else torch.tensor(float(Yrs))) ** 2.0))

def get_phi(atoms, n):
    n_c = (n * (1.0 + 0j))
    n_recip = op_J(atoms, n_c)
    return (((-4.0) * π) * op_Linv(atoms, op_O(atoms, n_recip)))

def op_I(atoms, W):
    s1 = atoms.s1
    s2 = atoms.s2
    s3 = atoms.s3
    Ngrid = ((s1 * s2) * s3)
    active = atoms.active()
    if vec_size(W) == vec_size(active):
        reciprocal_grid = W
    else:
        reciprocal_grid = mask_embed(W, active, Ngrid)
    reciprocal_grid3d = torch.reshape(reciprocal_grid, (int(s1), int(s2), int(s3),))
    real_grid = torch.fft.ifftn(reciprocal_grid3d)
    return (torch.reshape(real_grid, (int(Ngrid),)) * Ngrid)

def op_J(atoms, W):
    s1 = atoms.s1
    s2 = atoms.s2
    s3 = atoms.s3
    Ngrid = ((s1 * s2) * s3)
    real_grid = torch.reshape(W, (int(s1), int(s2), int(s3),))
    reciprocal_grid = torch.fft.fftn(real_grid)
    return (torch.reshape(reciprocal_grid, (int(Ngrid),)) / Ngrid)

def op_O(atoms, W):
    return (atoms.volume() * W)

def op_Linv(atoms, W):
    G2 = atoms.g2()
    nonzero = (torch.gt(G2, 0.0) * 1.0)
    safe_G2 = (G2 + (1.0 - nonzero))
    return (((W / safe_G2) / (-atoms.volume())) * nonzero)

# === Classes ===
class Atoms(nn.Module):
    def __init__(self, a, ecut, s1, s2, s3, Natoms, px, py, pz, Nstate, Z_nuc, f):
        super().__init__()
        self.a = torch.as_tensor(a).float()
        self.ecut = torch.as_tensor(ecut).float()
        self.s1 = int(s1)
        self.s2 = int(s2)
        self.s3 = int(s3)
        self.Natoms = int(Natoms)
        self.px = torch.as_tensor(px).float()
        self.py = torch.as_tensor(py).float()
        self.pz = torch.as_tensor(pz).float()
        self.Nstate = int(Nstate)
        self.Z_nuc = torch.as_tensor(Z_nuc).float()
        self.f = torch.as_tensor(f).float()
        self.learnable_params = [self.a, self.ecut, self.px, self.py, self.pz, self.Z_nuc, self.f]

    def volume(self):
        this = self
        return ((self.a * self.a) * self.a)

    def flat_index(self):
        this = self
        return torch.arange(((self.s1 * self.s2) * self.s3))

    def m1(self):
        this = self
        return torch.remainder(torch.floor((self.flat_index() / (self.s3 * self.s2)) if isinstance((self.flat_index() / (self.s3 * self.s2)), torch.Tensor) else torch.tensor(float((self.flat_index() / (self.s3 * self.s2))))), self.s1)

    def m2(self):
        this = self
        return torch.remainder(torch.floor((self.flat_index() / self.s3) if isinstance((self.flat_index() / self.s3), torch.Tensor) else torch.tensor(float((self.flat_index() / self.s3)))), self.s2)

    def m3(self):
        this = self
        return torch.remainder(self.flat_index(), self.s3)

    def fold_freq(self, m, axis_length):
        this = self
        m = torch.as_tensor(m, device=DEVICE).float()
        return (m - (torch.gt(m, (axis_length / 2)) * axis_length))

    def sample_coord(self, m, s):
        this = self
        m = torch.as_tensor(m, device=DEVICE).float()
        return (m * (self.a / s))

    def recip_scale(self):
        this = self
        return ((2 * π) / self.a)

    def structure_factor(self, n1, n2, n3, c, px, py, pz):
        this = self
        n1 = torch.as_tensor(n1, device=DEVICE).float()
        n2 = torch.as_tensor(n2, device=DEVICE).float()
        n3 = torch.as_tensor(n3, device=DEVICE).float()
        c = torch.as_tensor(c, device=DEVICE).float()
        px = torch.as_tensor(px, device=DEVICE).float()
        py = torch.as_tensor(py, device=DEVICE).float()
        pz = torch.as_tensor(pz, device=DEVICE).float()
        phase = (c * (((n1 * px) + (n2 * py)) + (n3 * pz)))
        return torch.exp(((-torch.tensor(1j)) * phase) if isinstance(((-torch.tensor(1j)) * phase), torch.Tensor) else torch.tensor(float(((-torch.tensor(1j)) * phase))))

    def freq_x(self):
        this = self
        return self.fold_freq(self.m1(), self.s1)

    def freq_y(self):
        this = self
        return self.fold_freq(self.m2(), self.s2)

    def freq_z(self):
        this = self
        return self.fold_freq(self.m3(), self.s3)

    def coord_x(self):
        this = self
        return self.sample_coord(self.m1(), self.s1)

    def coord_y(self):
        this = self
        return self.sample_coord(self.m2(), self.s2)

    def coord_z(self):
        this = self
        return self.sample_coord(self.m3(), self.s3)

    def gx(self):
        this = self
        return (self.recip_scale() * self.freq_x())

    def gy(self):
        this = self
        return (self.recip_scale() * self.freq_y())

    def gz(self):
        this = self
        return (self.recip_scale() * self.freq_z())

    def g2(self):
        this = self
        c = self.recip_scale()
        n1 = self.freq_x()
        n2 = self.freq_y()
        n3 = self.freq_z()
        return ((c * c) * (((n1 * n1) + (n2 * n2)) + (n3 * n3)))

    def active(self):
        this = self
        return torch.le(self.g2(), (2 * self.ecut))

    def g2c(self):
        this = self
        return torch.masked_select(self.g2(), self.active())

    def sf(self):
        this = self
        n1 = self.freq_x()
        n2 = self.freq_y()
        n3 = self.freq_z()
        c = self.recip_scale()
        natoms = self.Natoms
        Sf = (self.structure_factor(n1, n2, n3, c, self.px[int(0)], self.py[int(0)], self.pz[int(0)]) * 0.0)
        for a in range(int(0), int(natoms)):
            Sf = Sf + self.structure_factor(n1, n2, n3, c, self.px[int(a)], self.py[int(a)], self.pz[int(a)])
        return Sf

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
π = 3.141592653589793
a = 16.0
ecut = 16.0
s = 60
Natoms = 1
Nstate = 1
px = torch.tensor([0.0], device=DEVICE)
py = torch.tensor([0.0], device=DEVICE)
pz = torch.tensor([0.0], device=DEVICE)
Z_nuc = torch.tensor([1.0], device=DEVICE)
f = torch.tensor([1.0], device=DEVICE)
H_atom = Atoms(a, ecut, s, s, s, Natoms, px, py, pz, Nstate, Z_nuc, f).to(DEVICE)
W0 = (torch.exp(((-0.5) * H_atom.g2c()) if isinstance(((-0.5) * H_atom.g2c()), torch.Tensor) else torch.tensor(float(((-0.5) * H_atom.g2c())))) * (1.0 + 0j))
Y0 = orth(H_atom, W0)
norm_check = (H_atom.volume() * torch.sum((torch.abs(Y0 if isinstance(Y0, torch.Tensor) else torch.tensor(float(Y0))) ** 2.0) if isinstance((torch.abs(Y0 if isinstance(Y0, torch.Tensor) else torch.tensor(float(Y0))) ** 2.0), torch.Tensor) else torch.tensor(float((torch.abs(Y0 if isinstance(Y0, torch.Tensor) else torch.tensor(float(Y0))) ** 2.0)))))
physika_print(norm_check)
n0 = get_n_total(H_atom, Y0)
phi0 = get_phi(H_atom, n0)
physika_print(phi0[int(0)])
physika_print(phi0[int(1)])