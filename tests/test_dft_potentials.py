import pytest
import torch
from tests.conftest import exec_phyk

r_tol = 1e-05
a_tol = 1e-06
PI = 3.141592653589793


@pytest.fixture(scope="module")
def pot_ns():
    """Execute examples/dft_potentials.phyk and return its namespace."""
    return exec_phyk("dft_potentials")


def _coulomb_ref(Z_nuc, G2, Sf, s1, s2, s3):
    nonzero = torch.gt(G2, 0.0)
    nonzero_f = nonzero * 1.0
    safe_G2 = G2 + (1.0 - nonzero_f) * 1.0
    Vcoul = (-4.0 * PI * Z_nuc / safe_G2) * nonzero_f
    n = s1 * s2 * s3
    cube = torch.reshape(Vcoul * Sf, (s1, s2, s3))
    spec = torch.fft.fftn(cube)
    return torch.reshape(spec, (n, )) / n


class TestCoulombSmoke:

    def test_output_shape(self, pot_ns):
        assert pot_ns["Vcoul_g"].shape == (8, )

    def test_output_complex(self, pot_ns):
        assert pot_ns["Vcoul_g"].dtype == torch.complex64

    def test_dc_mode_no_inf_or_nan(self, pot_ns):
        # The 1/|G|^2 singularity at G=0 must be masked, never leak inf/nan.
        v = pot_ns["Vcoul_g"]
        assert torch.isfinite(v.real).all()
        assert torch.isfinite(v.imag).all()


class TestCoulombSemantics:

    def test_grid_has_dc_mode(self, pot_ns):
        # G2[0] is exactly 0 — the case the mask must handle.
        assert pot_ns["G2"][0].item() == pytest.approx(0.0)

    def test_matches_reference(self, pot_ns):
        coulomb = pot_ns["coulomb"]
        G2, Sf = pot_ns["G2"], pot_ns["Sf"]
        out = coulomb(1.0, G2, Sf, 2, 2, 2)
        expected = _coulomb_ref(1.0, G2, Sf, 2, 2, 2)
        assert torch.allclose(out, expected, rtol=r_tol, atol=a_tol)

    def test_linear_in_charge(self, pot_ns):
        # Vcoul is linear in Z: coulomb(2Z) == 2 * coulomb(Z).
        coulomb = pot_ns["coulomb"]
        G2, Sf = pot_ns["G2"], pot_ns["Sf"]
        one = coulomb(1.0, G2, Sf, 2, 2, 2)
        two = coulomb(2.0, G2, Sf, 2, 2, 2)
        assert torch.allclose(two, 2.0 * one, rtol=r_tol, atol=a_tol)
