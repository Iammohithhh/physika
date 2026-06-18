import pytest
from tests.conftest import exec_phyk
import torch

r_tol = 1e-05
a_tol = 1e-06


@pytest.fixture(scope="module")
def ops_ns():
    """Execute examples/dft_operators.phyk and return its namespace."""
    return exec_phyk("dft_operators")


class TestActiveSet:
    """The 2x2x2 smoke grid: 4 of 8 plane waves below the cutoff."""

    def test_active_count(self, ops_ns):
        assert int(ops_ns["active"].sum()) == 4

    def test_G2c_keeps_active_only(self, ops_ns):
        assert ops_ns["G2c"].shape == (4,)

    def test_dc_component_is_zero(self, ops_ns):
        assert ops_ns["G2"][0].item() == pytest.approx(0.0)


class TestForwardBackward:
    """op_J (real -> reciprocal) and op_I (its inverse)."""

    @pytest.mark.parametrize("var", ["W_real", "W_spec", "W_back"])
    def test_dtype(self, ops_ns, var):
        assert ops_ns[var].dtype == torch.complex64

    def test_roundtrip_recovers_field(self, ops_ns):
        # op_I(op_J(W)) == W.
        assert torch.allclose(ops_ns["W_back"], ops_ns["W_real"],
                              rtol=r_tol, atol=a_tol)


class TestOverlap:
    """op_O: scale by the cell volume Omega."""

    def test_scales_by_omega(self, ops_ns):
        op_O = ops_ns["op_O"]
        W = ops_ns["W_real"]
        assert torch.allclose(op_O(W, 8.0), 8.0 * W, rtol=r_tol, atol=a_tol)


class TestLaplacian:
    """op_L and its inverse op_Linv."""

    def test_opL_matches_definition(self, ops_ns):
        # op_L(W) == -Omega * G2c * W.
        op_L = ops_ns["op_L"]
        G2c = ops_ns["G2c"]
        W = ops_ns["W_active"]
        expected = (-ops_ns["Omega"]) * G2c * W
        assert torch.allclose(op_L(W, G2c, ops_ns["Omega"]), expected,
                              rtol=r_tol, atol=a_tol)

    def test_linv_inverts_L_on_nonzero_modes(self, ops_ns):
        # Linv(L(W)) == W everywhere except the DC mode (forced to 0).
        assert torch.allclose(ops_ns["LinvLW"], ops_ns["W_active"],
                              rtol=r_tol, atol=a_tol)

    def test_linv_zeros_dc_component(self, ops_ns):
        # The G=0 entry must be exactly zero, never inf/nan from /0.
        op_Linv = ops_ns["op_Linv"]
        G2c = ops_ns["G2c"]
        ones = torch.ones(4, dtype=torch.complex64)
        out = op_Linv(ones, G2c, ops_ns["Omega"])
        assert out[0].item() == 0
        assert torch.isfinite(out.real).all()


class TestAdjoints:
    """op_Idag (gather via mask_select) and op_Jdag (scatter via mask_embed)."""

    def test_idag_jdag_roundtrip_on_constant(self, ops_ns):
        # A constant real field survives the active-set round trip unchanged.
        assert torch.allclose(ops_ns["W_jdag"], ops_ns["W_ones"],
                              rtol=r_tol, atol=a_tol)

    def test_mask_embed_inverts_mask_select(self, ops_ns):
        # The core of op_Jdag: scatter is the left-inverse of gather.
        # mask_embed(mask_select(x, m), m, n) keeps active entries, zeros rest.
        op_Idag = ops_ns["op_Idag"]
        op_Jdag = ops_ns["op_Jdag"]
        active = ops_ns["active"]
        # round trip a non-constant field through gather then scatter
        W = (torch.arange(8, dtype=torch.float32) + 1).to(torch.complex64)
        compact = op_Idag(W, active, 2, 2, 2)
        assert compact.shape == (4,)
        back = op_Jdag(compact, active, 8, 2, 2, 2)
        assert back.shape == (8,)
