import pytest
import torch
from tests.conftest import exec_phyk

S = 60


@pytest.fixture(scope="module")
def atoms_ns():
    return exec_phyk("dft_atoms")


class TestHAtomBasis:

    def test_grid_size(self, atoms_ns):
        assert int(atoms_ns["n"]) == S**3

    def test_cell_volume(self, atoms_ns):
        assert atoms_ns["Omega"] == pytest.approx(4096.0)

    def test_active_count_matches_julia(self, atoms_ns):
        assert int(atoms_ns["active"].sum()) == 12533
        assert int(atoms_ns["count"]) == 12533

    def test_active_is_bool_mask(self, atoms_ns):
        assert atoms_ns["active"].dtype == torch.bool
        assert atoms_ns["active"].shape == (S**3,)

    def test_G2c_keeps_active_only(self, atoms_ns):
        assert atoms_ns["G2c"].shape == (12533,)

    def test_dc_component_is_zero(self, atoms_ns):
        assert atoms_ns["G2"][0].item() == pytest.approx(0.0)

    def test_structure_factor_all_ones(self, atoms_ns):
        Sf = atoms_ns["Sf"]
        assert Sf.shape == (S**3,)
        assert torch.allclose(Sf, torch.ones_like(Sf), atol=1e-5)
