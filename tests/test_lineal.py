import numpy as np
import pytest

from simtele.lineal import (cg, es_spd, gauss_seidel, hilbert, iter_matrix_radii, jacobi, lu_doolittle, lu_solve, tridiag)

A = tridiag(4, 4.0); XS = np.array([1., 2., 3., 4.]); B = A @ XS


def test_los_cuatro_metodos_resuelven_el_sistema_normal():
    for f in (jacobi, gauss_seidel, cg):
        H, ok = f(A, B, XS, tol=1e-10, maxit=1000)
        assert ok and np.linalg.norm(H[-1]["x"] - XS) < 1e-8
    assert np.linalg.norm(lu_solve(A, B)[0] - XS) < 1e-12


def test_gauss_seidel_mas_rapido_que_jacobi_y_cg_termina_en_n_pasos():
    assert len(gauss_seidel(A, B, XS)[0]) < len(jacobi(A, B, XS)[0])
    assert len(cg(A, B, XS)[0]) - 1 <= 5


def test_lu_reproduce_A_y_rechaza_pivote_nulo():
    L, U = lu_doolittle(A)
    assert np.allclose(L @ U, A, atol=1e-12)
    with pytest.raises(ValueError):
        lu_doolittle(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_cg_exige_spd():
    assert es_spd(A) and es_spd(hilbert(4))
    with pytest.raises(ValueError):
        cg(np.array([[1.0, 2.0], [3.0, 1.0]]), np.ones(2), np.ones(2))


def test_hilbert_jacobi_diverge_aunque_sea_spd():
    H = hilbert(4); xs = np.ones(4)
    assert iter_matrix_radii(H)[0] > 1
    assert not jacobi(H, H @ xs, xs)[1]


def test_no_divide_por_cero_cuando_la_referencia_es_nula():
    H, ok = jacobi(A, np.zeros(4), np.zeros(4))
    assert H[0]["erel"] is None and H[0]["eabs"] == 0.0
