# tests/test_dft_run.py

import math
import pytest
import torch
from tests.conftest import exec_phyk


S = (2, 2, 2)
A = 2.0
ECUT = 5.0


@pytest.fixture(scope="module")
def scf_ns():
    """Execute examples/dft_scf.phyk and return its namespace."""
    return exec_phyk("dft_scf")


class TestRunSCFMultiAtom:
    """Tests multi-atom and non-unit nuclear charge systems."""

    def _h2_like(self, scf_ns):
        """Create a two-atom hydrogen-like system."""
        return scf_ns["Atoms"](
            A, ECUT, S[0], S[1], S[2],
            2,
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            1,
            torch.tensor([1.0, 1.0]),
            torch.tensor([2.0]),
        )

    def _he_like(self, scf_ns):
        """Create a helium-like system with Z = 2."""
        return scf_ns["Atoms"](
            A, ECUT, S[0], S[1], S[2],
            1,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            1,
            torch.tensor([2.0]),
            torch.tensor([2.0]),
        )

    def test_smoke_h2_like(self, scf_ns):
        # Check that the H2-like SCF run returns a finite energy.
        out = scf_ns["runSCF"](
            self._h2_like(scf_ns), 0.0, 5, 1e-4, 1e-8, 11
        )

        assert math.isfinite(float(out.detach()))

    def test_smoke_he_like(self, scf_ns):
        # Check that the He-like SCF run returns a finite energy.
        out = scf_ns["runSCF"](
            self._he_like(scf_ns), 0.0, 5, 1e-4, 1e-8, 11
        )

        assert math.isfinite(float(out.detach()))

    def test_reproducible_with_same_seed(self, scf_ns):
        # The same seed should give the same final energy.
        atoms = self._h2_like(scf_ns)

        e1 = float(
            scf_ns["runSCF"](atoms, 0.0, 5, 1e-4, 1e-8, 7).detach()
        )
        e2 = float(
            scf_ns["runSCF"](atoms, 0.0, 5, 1e-4, 1e-8, 7).detach()
        )

        assert e1 == pytest.approx(e2, rel=1e-6)

    def test_zero_iterations_returns_dummy_initial(self, scf_ns):
        # With zero iterations, runSCF returns the initial dummy energy.
        out = scf_ns["runSCF"](self._he_like(scf_ns), 0.0, 0, 1e-4, 1e-8, 11)
        val = out.detach() if hasattr(out, "detach") else out
        assert float(val) == 0.0
