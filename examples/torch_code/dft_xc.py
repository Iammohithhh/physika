import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def lda_x(n):
    pi = 3.141592653589793
    slater = (((-3.0) / 4.0) * ((3.0 / (2.0 * pi)) ** (2.0 / 3.0)))
    rs = ((3.0 / ((4.0 * pi) * n)) ** (1.0 / 3.0))
    ex = (slater / rs)
    vx = ((4.0 / 3.0) * ex)
    return torch.stack([torch.as_tensor(ex), torch.as_tensor(vx)])

def lda_c_chachiyo(n):
    pi = 3.141592653589793
    a = (-0.01554535)
    b = 20.4562557
    rs = ((3.0 / ((4.0 * pi) * n)) ** (1.0 / 3.0))
    ec = (a * torch.log(((1.0 + (b / rs)) + (b / (rs ** 2.0))) if isinstance(((1.0 + (b / rs)) + (b / (rs ** 2.0))), torch.Tensor) else torch.tensor(float(((1.0 + (b / rs)) + (b / (rs ** 2.0)))))))
    vc = (ec + (((a * b) * (2.0 + rs)) / (3.0 * ((b + (b * rs)) + (rs ** 2.0)))))
    return torch.stack([torch.as_tensor(ec), torch.as_tensor(vc)])

def lda_c_vwn(n):
    pi = 3.141592653589793
    A = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = (-0.10498)
    rs = ((3.0 / ((4.0 * pi) * n)) ** (1.0 / 3.0))
    sqrt_rs = torch.sqrt(rs if isinstance(rs, torch.Tensor) else torch.tensor(float(rs)))
    denom = ((rs + (b * sqrt_rs)) + c)
    Q = torch.sqrt(((4.0 * c) - (b ** 2.0)) if isinstance(((4.0 * c) - (b ** 2.0)), torch.Tensor) else torch.tensor(float(((4.0 * c) - (b ** 2.0)))))
    f_at_x0 = ((b * x0) / (((x0 ** 2.0) + (b * x0)) + c))
    f3 = ((2.0 * ((2.0 * x0) + b)) / Q)
    lin = ((2.0 * sqrt_rs) + b)
    atan_term = torch.atan((Q / lin) if isinstance((Q / lin), torch.Tensor) else torch.tensor(float((Q / lin))))
    ec = (A * ((torch.log((rs / denom) if isinstance((rs / denom), torch.Tensor) else torch.tensor(float((rs / denom)))) + (((2.0 * b) / Q) * atan_term)) - (f_at_x0 * (torch.log((((sqrt_rs - x0) ** 2.0) / denom) if isinstance((((sqrt_rs - x0) ** 2.0) / denom), torch.Tensor) else torch.tensor(float((((sqrt_rs - x0) ** 2.0) / denom)))) + (f3 * atan_term)))))
    tt = ((lin ** 2.0) + (Q ** 2.0))
    vc = (ec - (((sqrt_rs * A) / 6.0) * ((((2.0 / sqrt_rs) - (lin / denom)) - ((4.0 * b) / tt)) - (f_at_x0 * (((2.0 / (sqrt_rs - x0)) - (lin / denom)) - ((4.0 * ((2.0 * x0) + b)) / tt))))))
    return torch.stack([torch.as_tensor(ec), torch.as_tensor(vc)])

# === Program ===
n_test = torch.stack([torch.as_tensor((0.5 + 0j), dtype=torch.complex64), torch.as_tensor((1.0 + 0j), dtype=torch.complex64), torch.as_tensor((2.0 + 0j), dtype=torch.complex64), torch.as_tensor((4.0 + 0j), dtype=torch.complex64)])
ex_vx = lda_x(n_test)
ec_vc = lda_c_chachiyo(n_test)
ec_vc_vwn = lda_c_vwn(n_test)
physika_print(ex_vx)
physika_print(ec_vc)
physika_print(ec_vc_vwn)