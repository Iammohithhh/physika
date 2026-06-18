import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06
PI = 3.141592653589793


@pytest.fixture(scope="module")
def xc_ns():
    """Execute examples/dft_xc.phyk and return its namespace."""
    return exec_phyk("dft_xc")


def _rs(n):
    return (3.0 / (4.0 * PI * n))**(1.0 / 3.0)


SAMPLE = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.complex64)


class TestLdaX:
    """Slater exchange: [ex, vx] stacked, vx = 4/3 ex, ex < 0, ex ~ n^(1/3)."""

    def test_returns_stacked_pair(self, xc_ns):
        assert xc_ns["ex_vx"].shape == (2, 4)

    def test_dtype_complex(self, xc_ns):
        assert xc_ns["ex_vx"].dtype == torch.complex64

    def test_matches_slater_formula(self, xc_ns):
        f = -3.0 / 4.0 * (3.0 / (2.0 * PI))**(2.0 / 3.0)
        rs = _rs(SAMPLE)
        ex = f / rs
        vx = 4.0 / 3.0 * ex
        out = xc_ns["lda_x"](SAMPLE)
        assert torch.allclose(out[0], ex, rtol=r_tol, atol=a_tol)
        assert torch.allclose(out[1], vx, rtol=r_tol, atol=a_tol)

    def test_golden_value_at_unit_density(self, xc_ns):
        # Independent hand-computed reference at n = 1.
        out = xc_ns["lda_x"](torch.tensor([1.0], dtype=torch.complex64))
        assert out[0].real.item() == pytest.approx(-0.738443, abs=1e-3)
        assert out[1].real.item() == pytest.approx(-0.984591, abs=1e-3)

    def test_vx_is_four_thirds_ex(self, xc_ns):
        out = xc_ns["ex_vx"]
        assert torch.allclose(out[1], (4.0 / 3.0) * out[0],
                              rtol=r_tol,
                              atol=a_tol)

    def test_exchange_is_negative(self, xc_ns):
        out = xc_ns["ex_vx"]
        assert (out[0].real < 0).all()
        assert (out[1].real < 0).all()

    def test_density_scaling_law(self, xc_ns):
        # rs ~ n^(-1/3)  =>  ex = f/rs ~ n^(1/3).
        lda_x = xc_ns["lda_x"]
        n = torch.tensor([1.0, 3.0], dtype=torch.complex64)
        ex = lda_x(n)[0]
        ratio = (ex[1] / ex[0]).real.item()
        assert ratio == pytest.approx(3.0**(1.0 / 3.0), rel=1e-4)


class TestLdaCChachiyo:
    """Chachiyo correlation: single smooth expression, ec < 0."""

    def test_shape_and_dtype(self, xc_ns):
        assert xc_ns["ec_vc"].shape == (2, 4)
        assert xc_ns["ec_vc"].dtype == torch.complex64

    def test_matches_chachiyo_formula(self, xc_ns):
        a = -0.01554535
        b = 20.4562557
        rs = _rs(SAMPLE)
        ec = a * torch.log(1.0 + b / rs + b / rs**2.0)
        vc = ec + a * b * (2.0 + rs) / (3.0 * (b + b * rs + rs**2.0))
        out = xc_ns["lda_c_chachiyo"](SAMPLE)
        assert torch.allclose(out[0], ec, rtol=r_tol, atol=a_tol)
        assert torch.allclose(out[1], vc, rtol=r_tol, atol=a_tol)

    def test_correlation_is_negative(self, xc_ns):
        assert (xc_ns["ec_vc"][0].real < 0).all()

    def test_finite(self, xc_ns):
        out = xc_ns["ec_vc"]
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()


class TestLdaCVwn:
    """VWN correlation (reference-only upstream); exercises atan + sqrt."""

    def test_shape_and_dtype(self, xc_ns):
        assert xc_ns["ec_vc_vwn"].shape == (2, 4)
        assert xc_ns["ec_vc_vwn"].dtype == torch.complex64

    def test_matches_vwn_formula(self, xc_ns):
        A = 0.0310907
        b = 3.72744
        c = 12.9352
        x0 = -0.10498
        rs = _rs(SAMPLE)
        x = torch.sqrt(rs)
        X = rs + b * x + c
        Q = torch.sqrt(torch.tensor(4.0 * c - b**2.0))
        fx0 = b * x0 / (x0**2.0 + b * x0 + c)
        f3 = 2.0 * (2.0 * x0 + b) / Q
        tx = 2.0 * x + b
        tanx = torch.atan(Q / tx)
        ec = A * (torch.log(rs / X) + 2.0 * b / Q * tanx - fx0 * (torch.log(
            (x - x0)**2.0 / X) + f3 * tanx))
        tt = tx**2.0 + Q**2.0
        vc = ec - x * A / 6.0 * (2.0 / x - tx / X - 4.0 * b / tt - fx0 *
                                 (2.0 / (x - x0) - tx / X - 4.0 *
                                  (2.0 * x0 + b) / tt))
        out = xc_ns["lda_c_vwn"](SAMPLE)
        assert torch.allclose(out[0], ec, rtol=r_tol, atol=a_tol)
        assert torch.allclose(out[1], vc, rtol=r_tol, atol=a_tol)

    def test_correlation_is_negative(self, xc_ns):
        assert (xc_ns["ec_vc_vwn"][0].real < 0).all()

    def test_finite(self, xc_ns):
        out = xc_ns["ec_vc_vwn"]
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()
