"""Small examples with known exact Gaussian entanglement of formation.

Run ``python3 examples.py`` to compare the numerical bounds with exact values.
The two constructor functions can also be imported by other programs.
"""
import argparse
import numpy as np
from scipy.linalg import block_diag
from geof import solve_geof, lower_bound, bosonic_entropy


def two_mode_squeezed(r):
    eye, z = np.eye(2), np.diag([1.,-1.])
    return np.block([[np.cosh(2*r)*eye,np.sinh(2*r)*z],
                     [np.sinh(2*r)*z,np.cosh(2*r)*eye]])


def ppt_family(a, c):
    """The exactly solvable 2:2 PPT family in the manuscript, c>=0."""
    if c < 0 or a < np.sqrt(1+c*c+np.sqrt(2)*c)-1e-12:
        raise ValueError("Require c>=0 and a>=sqrt(1+c^2+sqrt(2)c).")
    eye, z, x = np.eye(2), np.diag([1.,-1.]), np.array([[0.,1.],[1.,0.]])
    xx = np.block([[a*eye,c*z],[c*z,a*eye]])
    pp = np.block([[a*eye,c*x],[c*x,a*eye]])
    delta = a*a-c*c
    h = max(0.,delta*delta+1-2*a*a)
    zz = max(0.,(np.sqrt(2)*c*(delta+1)-2*a*np.sqrt(h))/(2*(2*a*a-c*c)))
    q = np.block([[np.sqrt(1+zz*zz)*eye,zz*(z-x)/np.sqrt(2)],
                  [zz*(z-x)/np.sqrt(2),np.sqrt(1+zz*zz)*eye]])
    order = [0,4,1,5,2,6,3,7]
    v = block_diag(xx,pp)[np.ix_(order,order)]
    gamma = block_diag(q,np.linalg.inv(q))[np.ix_(order,order)]
    exact = 2*bosonic_entropy((np.sqrt(1+zz*zz)-1)/2)
    return v,gamma,exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='check the results against the exact values')
    args = parser.parse_args()
    r, nu = .6, 1.4
    v2 = nu*two_mode_squeezed(r)
    exact2 = bosonic_entropy(np.sinh(max(0., r-.5*np.log(nu)))**2)
    v4, gamma4, exact4 = ppt_family(1.9, 1.)
    cases = [('two-mode thermal state', v2, [0], exact2),
             ('four-mode PPT state', v4, [0, 1], exact4)]
    print('Entanglement in nats')
    for name, v, alice, exact in cases:
        result = solve_geof(v, alice, n_starts=2, maxiter=250)
        lo, hi = result['lower_bound'], result['upper_bound']
        print(f'{name}: [{lo:.12g}, {hi:.12g}], '
              f'gap={result["gap"]:.3g}, exact={exact:.12g}')
        if args.check:
            assert lo <= exact+2e-8
            assert hi >= exact-2e-8
            assert hi-exact < 3e-6
    if args.check:
        result = lower_bound(v4, gamma4, [0, 1])
        assert abs(result['upper_bound']-exact4) < 1e-10
        assert result['lower_bound'] <= exact4+2e-8
        print('Example checks passed.')


if __name__ == '__main__':
    main()
