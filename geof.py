"""Gaussian entanglement of formation from covariance matrices.

Use interleaved (x1,p1,x2,p2,...) coordinates, hbar=1, and
V_jk=<{Delta R_j,Delta R_k}>, so the vacuum covariance is I.
Entropies are in nats. The calculator returns a feasible pure Gaussian
covariance, upper and lower bounds, and numerical residuals.
See README.md for the interface and the accompanying paper for the derivation.
"""
from __future__ import annotations

import warnings
from numbers import Integral
import numpy as np
from scipy.linalg import schur, block_diag
from scipy.optimize import minimize


def omega(n: int) -> np.ndarray:
    return np.kron(np.eye(n), np.array([[0., 1.], [-1., 0.]]))


def _sym(a):
    return (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T) / 2


def _power(a, t):
    w, u = np.linalg.eigh(_sym(a))
    if np.min(w) <= 0:
        raise ValueError("A positive-definite matrix is required.")
    return _sym((u * w**t) @ u.T)


def _indices(modes):
    return np.array([q for j in modes for q in (2*j, 2*j+1)], dtype=int)


def _partition(n, alice_modes):
    a = tuple(range(n//2)) if alice_modes is None else tuple(alice_modes)
    if any(not isinstance(j, Integral) or isinstance(j, (bool, np.bool_)) for j in a):
        raise ValueError("alice_modes must contain integer mode indices.")
    if len(set(a)) != len(a) or any(j < 0 or j >= n for j in a):
        raise ValueError("alice_modes must contain distinct valid mode indices.")
    return a, tuple(j for j in range(n) if j not in a)


def symplectic_eigenvalues(v):
    """Symplectic eigenvalues from a Hermitian matrix, in increasing order."""
    v = _sym(v)
    root = _power(v, .5)
    vals = np.linalg.eigvalsh(1j * root @ omega(len(v)//2) @ root)
    return np.sort(vals[len(v)//2:])


def bosonic_entropy(x):
    """g(x)=(x+1)log(x+1)-x log(x), with continuous value g(0)=0."""
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-10):
        raise ValueError("A nonnegative occupation is required.")
    x = np.maximum(x, 0.)
    out = np.zeros_like(x)
    positive = x > 0
    out[positive] = np.log1p(x[positive]) + x[positive]*np.log1p(1/x[positive])
    return float(out) if out.ndim == 0 else out


def gaussian_entropy(v):
    nus = symplectic_eigenvalues(v)
    if np.min(nus) < 1-1e-8:
        raise ValueError("The covariance is not physical in the vacuum-I convention.")
    return float(np.sum(bosonic_entropy(np.maximum(nus-1, 0)/2)))


def pure_entanglement(gamma, alice_modes):
    n = len(gamma)//2
    a, b = _partition(n, alice_modes)
    if not a or not b:
        return 0.
    ix = _indices(a)
    return gaussian_entropy(np.asarray(gamma)[np.ix_(ix, ix)])


def covariance_diagnostics(v, gamma=None):
    if np.iscomplexobj(v):
        raise ValueError("Quadrature covariance matrices must be real.")
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or len(v) == 0 or len(v) % 2:
        raise ValueError("A real 2n by 2n covariance is required.")
    if not np.all(np.isfinite(v)):
        raise ValueError("Covariance entries must be finite.")
    j = omega(len(v)//2)
    result = {
        "symmetry_residual": float(np.linalg.norm(v-v.T, 'fro')),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(_sym(v)))),
        "uncertainty_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(_sym(v)+1j*j)))
    }
    if gamma is not None:
        gamma = _sym(gamma)
        result.update({
            "purity_residual": float(np.linalg.norm(gamma @ j @ gamma-j, 'fro')),
            "feasibility_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(_sym(v)-gamma)))
        })
    return result


def _checked_covariance(v):
    diagnostics = covariance_diagnostics(v)
    v = np.asarray(v, dtype=float)
    scale = max(1., np.linalg.norm(v, 2))
    if diagnostics['symmetry_residual'] > 1e-9*scale:
        raise ValueError("Input covariance must be symmetric.")
    v = _sym(v)
    if diagnostics['minimum_eigenvalue'] <= 0:
        raise ValueError("Input covariance must be positive definite.")
    nus = symplectic_eigenvalues(v)
    if np.min(nus) < 1-2e-8:
        raise ValueError("Input covariance is not physical in the vacuum-I convention.")
    return v, diagnostics, nus


def williamson(v):
    """Return S, nu with V=S diag(nu1,nu1,...) S.T and symplectic S.

    The descending order puts mixed modes before possible pure normal modes.
    This is a floating decomposition; residuals are checked by solve_geof.
    """
    v = _sym(v)
    n = len(v)//2
    root = _power(v, .5)
    t, u = schur(root @ omega(n) @ root, output='real')
    nus = []
    for j in range(n):
        if t[2*j, 2*j+1] < 0:
            u[:, 2*j+1] *= -1
        nus.append(abs(t[2*j, 2*j+1]))
    order = np.argsort(nus)[::-1]
    ix = _indices(order)
    nus = np.array(nus)[order]
    if np.min(nus) <= 0:
        raise ValueError("Williamson decomposition failed: singular covariance.")
    s = root @ u[:, ix] @ np.diag(np.repeat(nus, 2)**(-.5))
    return s, nus


def pure_below(v):
    """Canonical pure covariance supplied by a Williamson decomposition."""
    s, _ = williamson(v)
    return _sym(s @ s.T)


def _Fprime(x):
    """Derivative of F(x)=g(sinh(x/4)^2), including F'(0)=0."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x)/2
    out = np.zeros_like(x)
    mid = (a > 1e-12) & (a < 30)
    out[mid] = .5*np.sinh(a[mid])*np.log1p(2/np.expm1(a[mid]))
    out[a >= 30] = .5
    # The omitted values are at most O(|x| log(1/|x|)); evaluating exactly
    # at x=0 implements the continuous derivative. No clipping is used for
    # resolved eigenvalues away from zero.
    tiny = (a > 0) & (a <= 1e-12)
    out[tiny] = .5*a[tiny]*np.log(2/a[tiny])
    return np.sign(x)*out


def entropy_gradient(gamma, alice_modes):
    """Analytic full-matrix gradient of the reflection-spectrum extension.

    Its restriction to the pure-covariance tangent space is the differential
    of pure-state entanglement, including at zero Schmidt squeezing.
    """
    gamma = _sym(gamma)
    n = len(gamma)//2
    a, b = _partition(n, alice_modes)
    if not a or not b:
        return np.zeros_like(gamma)
    signs = np.ones(2*n)
    signs[_indices(b)] = -1
    pi = np.diag(signs)
    invroot = _power(gamma, -.5)
    r = _sym(invroot @ pi @ gamma @ pi @ invroot)
    lam, u = np.linalg.eigh(r)
    if np.min(lam) <= 0:
        raise ValueError("Reflection spectrum lost positivity.")
    fp = _Fprime(np.log(lam))
    fpr = (u*(fp/lam)) @ u.T
    rfpr = (u*fp) @ u.T
    return _sym((pi @ invroot @ fpr @ invroot @ pi - invroot @ rfpr @ invroot)/4)


def tangent_projection(k):
    j = omega(len(k)//2)
    return _sym((k+j @ k @ j)/2)


def _psd_part(k):
    w, u = np.linalg.eigh(_sym(k))
    return _sym((u*np.maximum(w, 0)) @ u.T)


def lower_bound(v, gamma, alice_modes, optimise=True, z=None, solver=None):
    """Evaluate the global geodesic lower bound; optionally fit its PSD weight.

    For M=Gamma^-1/2 V Gamma^-1/2, K=Gamma^1/2 grad(E) Gamma^1/2,
    R=sqrt(2n) log(lambda_max(M)), and any Z>=0, the exact inequality is

      G(V) >= E(Gamma)-Tr[Z(M-I)]-R ||P(K+Z)||_F.

    Gamma must be pure and <=V. CVXPY is optional; without it, Z=0 is used.
    A PSD projection makes any returned Z an admissible weight in exact
    arithmetic. Reported evaluations are floating, not interval certified.
    """
    v, _, _ = _checked_covariance(v)
    gamma, _, gamma_nus = _checked_covariance(gamma)
    if gamma.shape != v.shape:
        raise ValueError("The seed and target must have the same dimension.")
    _partition(len(v)//2, alice_modes)
    if np.max(np.abs(gamma_nus-1)) > 2e-7:
        raise ValueError("The lower-bound formula requires a pure covariance.")
    d = len(v)
    root, invroot = _power(gamma, .5), _power(gamma, -.5)
    m = _sym(invroot @ v @ invroot)
    gap = _sym(m-np.eye(d))
    if np.min(np.linalg.eigvalsh(gap)) < -2e-8:
        raise ValueError("The lower-bound formula requires Gamma <= V.")
    radius = np.sqrt(d)*max(0., np.log(np.max(np.linalg.eigvalsh(m))))
    k = _sym(root @ entropy_gradient(gamma, alice_modes) @ root)
    status = 'supplied_weight' if z is not None else 'zero_weight'
    zvalue = np.zeros_like(v) if z is None else _psd_part(z)
    if optimise and z is None and radius > 1e-12:
        try:
            import cvxpy as cp
            zz = cp.Variable((d,d), symmetric=True)
            j = omega(d//2)
            tangent = (k+zz+j @ (k+zz) @ j)/2
            objective = cp.trace(zz @ gap)+radius*cp.norm(tangent, 'fro')
            problem = cp.Problem(cp.Minimize(objective), [zz >> 0])
            chosen = solver or ('CLARABEL' if 'CLARABEL' in cp.installed_solvers() else 'SCS')
            options = {'tol_gap_abs': 1e-9, 'tol_feas': 1e-9, 'tol_gap_rel': 1e-9, 'max_iter': 300} if chosen == 'CLARABEL' else {'eps': 1e-7, 'max_iters': 30000}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                problem.solve(solver=chosen, **options)
            status = str(problem.status)
            if zz.value is not None and np.all(np.isfinite(zz.value)):
                candidate = _psd_part(zz.value)
                penalty = lambda z_: max(0., float(np.trace(z_ @ gap)))+radius*np.linalg.norm(tangent_projection(k+z_), 'fro')
                if penalty(candidate) < penalty(zvalue):
                    zvalue = candidate
        except ImportError:
            status = 'cvxpy_unavailable_zero_weight'
        except Exception as exc:
            status = 'weight_fit_failed_'+type(exc).__name__
    residual = float(np.linalg.norm(tangent_projection(k+zvalue), 'fro'))
    complementarity = max(0., float(np.trace(zvalue @ gap)))
    upper = pure_entanglement(gamma, alice_modes)
    raw = upper-complementarity-radius*residual
    return {
        'lower_bound': max(0., float(raw)), 'raw_lower_bound': float(raw),
        'upper_bound': upper, 'gap': upper-max(0., float(raw)),
        'tangent_residual': residual, 'complementarity_penalty': complementarity,
        'geodesic_radius': float(radius), 'weight_fit_status': status,
        'interval_certified': False, 'z': zvalue,
        'weight_minimum_eigenvalue': float(np.min(np.linalg.eigvalsh(zvalue)))
    }


def separability_sdp(v, alice_modes, solver=None):
    """Find local physical covariances dominated by V, an SDP feasibility test.

    A feasible decomposition yields a product pure candidate. An unsuccessful
    numerical SDP is not by itself an entanglement certificate. Returns None
    if CVXPY is unavailable or no checked product candidate was obtained.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None
    v = _sym(v)
    n = len(v)//2
    a, b = _partition(n, alice_modes)
    if not a or not b:
        return {'gamma': pure_below(v), 'status': 'trivial_partition'}
    ix = np.concatenate([_indices(a), _indices(b)])
    vv = v[np.ix_(ix, ix)]
    na, nb = 2*len(a), 2*len(b)
    va, vb = cp.Variable((na,na), symmetric=True), cp.Variable((nb,nb), symmetric=True)
    delta = cp.Variable()
    ja, jb = omega(na//2), omega(nb//2)
    local = cp.bmat([[va,np.zeros((na,nb))],[np.zeros((nb,na)),vb]])
    constraints = [vv-local-delta*np.eye(2*n) >> 0,
                   cp.bmat([[va,-ja],[ja,va]]) >> 0,
                   cp.bmat([[vb,-jb],[jb,vb]]) >> 0]
    problem = cp.Problem(cp.Maximize(delta), constraints)
    chosen = solver or ('CLARABEL' if 'CLARABEL' in cp.installed_solvers() else 'SCS')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            problem.solve(solver=chosen)
        if va.value is None or vb.value is None or delta.value < -1e-7:
            return None
        gg = block_diag(pure_below(va.value), pure_below(vb.value))
        gamma = np.empty_like(v)
        gamma[np.ix_(ix,ix)] = gg
        if np.min(np.linalg.eigvalsh(v-gamma)) < -2e-8*max(1., np.linalg.norm(v,2)):
            return None
        return {'gamma':_sym(gamma), 'status':str(problem.status), 'slack':float(delta.value)}
    except Exception:
        return None


def _tangent_basis(n):
    basis = []
    qp_to_interleaved = np.array([j for i in range(n) for j in (i,n+i)])
    for kind in (0,1):
        for i in range(n):
            for j in range(i,n):
                a = np.zeros((n,n)); a[i,j] = a[j,i] = 1.
                if kind == 0:
                    h = block_diag(a,-a)
                else:
                    h = np.block([[np.zeros((n,n)),a],[a,np.zeros((n,n))]])
                basis.append(h[np.ix_(qp_to_interleaved,qp_to_interleaved)])
    return np.array(basis)


def solve_geof(v, alice_modes=None, n_starts=3, maxiter=500, seed=271828,
               fit_lower_bound=True, check_separability=True, pure_tolerance=1e-9):
    """Compute Gaussian entanglement of formation across a mode bipartition.

    Parameters
    ----------
    v : real symmetric (2n, 2n) array
        Physical covariance in interleaved quadrature order, with vacuum I.
    alice_modes : sequence of integers, optional
        Zero-based Alice mode indices; the remaining modes belong to Bob.
        The default assigns the first n//2 modes to Alice.
    n_starts, maxiter : int
        Number of feasible starting points and iterations per local search.
    seed : int
        Random seed for reproducible starting points.
    fit_lower_bound, check_separability : bool
        Use the optional CVXPY lower-bound fit and separability calculation.
    pure_tolerance : float
        Williamson eigenvalue tolerance for identifying fixed pure modes.

    Returns
    -------
    dict
        Candidate covariance ``gamma``, ``lower_bound``, ``upper_bound``,
        ``gap``, and residuals. Bounds use floating-point arithmetic;
        ``interval_certified`` is False. The input array is unchanged.
    """
    v, diagnostics, _ = _checked_covariance(v)
    scale = max(1., np.linalg.norm(v,2))
    n = len(v)//2
    a, b = _partition(n, alice_modes)
    s, nus = williamson(v)
    williamson_error = max(np.linalg.norm(s @ np.diag(np.repeat(nus,2)) @ s.T-v,'fro'),
                           np.linalg.norm(s @ omega(n) @ s.T-omega(n),'fro'))
    if williamson_error > 1e-7*scale:
        raise ValueError("Williamson decomposition residual is too large.")
    mixed = int(np.sum(nus > 1+pure_tolerance))
    product = separability_sdp(v,a) if check_separability and mixed else None
    search_runs = []
    if product is not None:
        gamma_best = product['gamma']
        method = 'separability_sdp_product_candidate'
    elif mixed == 0 or not a or not b:
        gamma_best = _sym(s @ s.T)
        method = 'pure_state_or_trivial_partition'
    else:
        dim = 2*mixed
        sm = s[:,:dim]
        fixed = s[:,dim:] @ s[:,dim:].T
        dd = np.diag(np.repeat(nus[:mixed],2))
        basis = _tangent_basis(mixed)
        count = len(basis)
        safety = min(1e-9*max(1.,np.max(nus)), (np.min(nus[:mixed])-1)/100)
        cache = {}

        def unpack(theta):
            if 'theta' in cache and np.array_equal(theta,cache['theta']):
                return cache['data']
            h = np.einsum('k,kij->ij',theta,basis)
            lam,u = np.linalg.eigh(h)
            ex = np.exp(lam)
            gm = _sym((u*ex) @ u.T)
            gamma = _sym(sm @ gm @ sm.T+fixed)
            val = pure_entanglement(gamma,a)
            gradg = _sym(sm.T @ entropy_gradient(gamma,a) @ sm)
            dl = lam[:,None]-lam[None,:]
            mid = (lam[:,None]+lam[None,:])/2
            ratio = np.ones_like(dl)
            nz = np.abs(dl) > 1e-8
            ratio[nz] = np.sinh(dl[nz]/2)/(dl[nz]/2)
            frechet = np.exp(mid)*ratio
            gradh = _sym(u @ (frechet*(u.T @ gradg @ u)) @ u.T)
            grad = np.einsum('ij,kij->k',gradh,basis)
            data = (gm,gamma,val,grad)
            cache['theta'],cache['data'] = np.array(theta,copy=True),data
            return data

        def feasible(theta):
            return np.linalg.eigvalsh(dd-unpack(theta)[0])-safety

        rng = np.random.default_rng(seed)
        bound = max(1e-4,float(np.log(np.max(nus[:mixed]))))
        best_value = np.inf
        for run in range(max(1,int(n_starts))):
            theta = np.zeros(count) if run == 0 else rng.normal(0,.15*bound,count)
            while np.min(feasible(theta)) < 0:
                theta *= .5
            result = minimize(lambda x:unpack(x)[2], theta, jac=lambda x:unpack(x)[3],
                              method='SLSQP', bounds=[(-bound,bound)]*count,
                              constraints=[{'type':'ineq','fun':feasible}],
                              options={'ftol':1e-11,'maxiter':int(maxiter),'disp':False})
            candidate = result.x
            # The path exp(tH) remains pure and joins the canonical feasible
            # covariance to the optimizer. Repair feasibility along that path.
            if np.min(feasible(candidate)) < 0:
                lo,hi = 0.,1.
                for _ in range(65):
                    mid = (lo+hi)/2
                    if np.min(feasible(mid*candidate)) >= 0:
                        lo=mid
                    else:
                        hi=mid
                candidate = lo*candidate
            gm,gamma,value,_ = unpack(candidate)
            search_runs.append({'success':bool(result.success),'iterations':int(result.nit),
                                'message':str(result.message),'upper_bound':float(value),
                                'minimum_slack':float(np.min(np.linalg.eigvalsh(dd-gm)))})
            if value < best_value:
                best_value = value
                gamma_best = gamma.copy()
        method = 'pure_logarithm_search'
    bound_result = lower_bound(v,gamma_best,a,optimise=fit_lower_bound)
    result = {**bound_result,
              'gamma':gamma_best,'alice_modes':list(a),'method':method,
              'normal_mode_occupations':((nus-1)/2).tolist(),
              'fixed_pure_normal_modes':n-mixed,'williamson_residual':float(williamson_error),
              'diagnostics':covariance_diagnostics(v,gamma_best),'search_runs':search_runs,
              'units':'nats','covariance_convention':'vacuum I; hbar=1',
              'interpretation':'Bounds evaluated in floating-point arithmetic; gap and matrix residuals describe numerical accuracy.'}
    return result


def json_ready(result):
    """Convert calculator output, including matrices, to JSON-compatible data."""
    if isinstance(result,np.ndarray):
        return result.tolist()
    if isinstance(result,dict):
        return {k:json_ready(v) for k,v in result.items()}
    if isinstance(result,(tuple,list)):
        return [json_ready(x) for x in result]
    if isinstance(result,np.generic):
        return result.item()
    return result

