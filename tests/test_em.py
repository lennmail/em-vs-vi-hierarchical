import numpy as np

from emvi.data import sample_known_groups
from emvi.em import EMKnownGroups, EMMixture, Priors


def exact_map_known(X, p):
    """Solve the (linear) MAP equations of model 1 directly."""
    n, K = X.shape
    A = np.zeros((K + 1, K + 1))
    b = np.zeros(K + 1)
    for k in range(K):
        A[k, k] = n / p.sigma2 + 1 / p.tau2
        A[k, K] = -1 / p.tau2
        b[k] = X[:, k].sum() / p.sigma2
    A[K, :K] = -1 / p.tau2
    A[K, K] = K / p.tau2 + 1 / p.tau_naught2
    b[K] = p.mu_naught / p.tau_naught2
    sol = np.linalg.solve(A, b)
    return sol[:K], sol[K]


def test_known_groups_matches_exact_map():
    rng = np.random.default_rng(0)
    p = Priors(sigma2=2.0, tau2=3.0, tau_naught2=5.0, mu_naught=1.0)
    s = sample_known_groups(
        rng, num_groups=4, num_obs_per_group=30, sigma2=2.0, tau2=3.0, tau_naught2=5.0, mu_naught=1.0
    )
    res = EMKnownGroups(p, tol=1e-12).fit(s.X, rng)
    theta, mu = exact_map_known(s.X, p)
    np.testing.assert_allclose(res.theta, theta, atol=1e-6)
    assert abs(res.mu - mu) < 1e-6
    assert np.all(np.diff(res.log_posterior) >= -1e-9)


def test_mixture_is_monotone_and_recovers_separated_means():
    rng = np.random.default_rng(1)
    p = Priors(sigma2=1.0, tau2=100.0, tau_naught2=100.0, mu_naught=0.0)
    theta = np.array([-6.0, 0.0, 7.0])
    counts = [300, 300, 400]
    x = np.concatenate([rng.normal(t, 1.0, c) for t, c in zip(theta, counts, strict=False)])
    res = EMMixture(p, num_groups=3, n_init=5, tol=1e-10).fit(x, rng)
    order = np.argsort(res.theta)
    assert np.all(np.diff(res.log_posterior) >= -1e-6)
    np.testing.assert_allclose(res.theta[order], theta, atol=0.2)
    np.testing.assert_allclose(res.weights[order], [0.3, 0.3, 0.4], atol=0.03)
