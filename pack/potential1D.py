import numpy as np
from scipy.sparse.linalg import eigs
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, hyperu, hyp1f1

from .local_util import root_idx, timer,  basis, quad_break


# def bisect_multi(f, x0, ep=1e-4):
#     points = np.sort(x0)
#     res = []
#     for i in range(len(points)-1):
#         if f(points[i]) * f(points[i+1]) < 0:
#             ri = root_scalar(f, method='bisect',
#                              bracket=[points[i]+ep, points[i+1]-ep])
#             res.append(ri.root)
#     return np.array(res)


def hyp1f1_dz(a, b, z):
    """
    M(a, b, z) = 1F1(a, b, z)
    return d hyp1f1 / dz
    ref: https://dlmf.nist.gov/13.3
    """
    return a/b*hyp1f1(a+1, b+1, z)


def hyperu_dz(a, b, z):
    """
    return d hyperu / dz
    ref: https://dlmf.nist.gov/13.3
    """
    return -a*hyperu(a+1, b+1, z)


class Potential1D:

    def __init__(self, V: callable, N, w, points=[], verbose=1, zero='mid'):
        """
        Numercically solve the bound state and the eigen wave functions for a
        particle in a potential V.
        The solutions are by diagonalized the Hamiltonian in the momentum
        space, i.e. sin and cos basis.

        V: potential function.
        N: the number of basis, must be an odd number.
        w: the basis frequency.

        For example, N = 5, w=0.1, then the basis will be:
        sqrt(1/T), sqrt(2/T) cos(0.1x), sqrt(2/T) cos(0.2x),
                   sqrt(2/T) sin(0.1x), sqrt(2/T) sin(0.2x), where T=2pi/w.

        points: the points where the potential is not smooth. When calculate
        the Fourier transform of the potential, we will separate the integral
        region according these points.

        verbose: ther verbose level. The greatter this value, the more
        information will be printed.

        zero: The default is 'mid', which means the potential is centered at
        zero. If you set zero to 'left', the potential will be shifted, where
        x=0 is the left of the potential.
        """
        self.V = V
        self.N = N
        self.w = w
        self.T = 2*np.pi / w
        self.points = points
        self.verbose = verbose
        self.NN = (N-1)/2  # number of the sin or cos
        self.levels = 0  # the number of eigenvalues and eigenvectors desired.
        if zero == 'mid':
            self.a = -self.T/2
            self.b = self.T/2
        elif zero == 'left':
            self.a = 0
            self.b = self.T

        # the default region when generate the wave function
        self.xlist = np.linspace(self.a, self.b, 1000)

    def get_eigenvals(self, levels):
        if not hasattr(self, 'eigenvals') or self.levels != levels:
            self._gen_eigenvals(levels)
        return self.eigenvals

    @timer
    def _gen_eigenvals(self, levels):
        """
        levels: The number of eigenvalues and eigenvectors desired.
        This is the eigen energies are with smallest magnetidue, so the larege
        negative bound state will not be find. One can shift the potential
        positive in order to find the ground state.

        See: docs of scipy.sparse.linalg.eigs
        """
        self.levels = levels
        self.get_H()
        if self.verbose > 0:
            print('=========== digonal the Hamiltonian ... ===========')
        val, vec = eigs(self.H, k=levels, which='SM')
        self.eigenvals = np.real(val)
        self.vec = np.real(vec)
        if self.verbose > 0:
            print('=========== digonal the Hamiltonian FINISHED! =====')
        return self.eigenvals

    def get_eigenwf(self, xs=[]):
        if not hasattr(self, 'eigenwf'):
            if len(xs) == 0:
                xs = np.copy(self.xlist)
            self._gen_eigenwf(xs=xs)
        return self.eigenwf

    @timer
    def _gen_eigenwf(self, xs):
        """
        get eigen wave function.
        xs: the points where the wave function is calculated.
        """
        self.xs = xs
        if not hasattr(self, 'eigenvals'):
            raise ValueError('call get_eigenvals first!')
        # psi = np.zeros([len(self.eigenvals), len(xs)])
        self.eigenwf = []
        if self.verbose > 0:
            print('=========== generate wave function ... ============')
        for veci in self.vec.T:
            psi_i = np.zeros(len(xs))
            for j in range(self.N):
                basis_j_x = np.array([basis(xi, j, self.N, self.w)
                                      for xi in xs])
                psi_i += veci[j]*basis_j_x
            self.eigenwf.append(psi_i)
            if self.verbose > 0:
                print('check normalization: ', np.trapz(psi_i**2, xs))
        if self.verbose > 0:
            print('=========== generate wave function FINISHED! ======')

    def get_H(self):
        if not hasattr(self, 'H'):
            self._gen_H()
        return self.H

    @timer
    def _gen_H(self):
        """
        c: cos term
        s: sin term

        cosA cosB = (   cos(A+B) + cos(A-B)) / 2
        sinA sinB = ( - cos(A+B) + cos(A-B)) / 2
        cosA sinB = (   sin(A+B) - sin(A-B)) / 2
        """
        self._gen_sin_cos_list()
        if self.verbose > 0:
            print('=========== generate Hamiltonian ... ==============')
        self.H = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N)[::-1]:
                if i <= j:
                    self.H[i, j] = self._gen_Hnm(i, j)
                else:
                    self.H[i, j] = self.H[j, i]
        if self.verbose > 0:
            print('=========== generate Hamiltonian FINISHED! ========')

    def _gen_Hnm(self, n, m):
        """
        for n <= m only

        c: cos term
        s: sin term

        cosA cosB = (   cos(A+B) + cos(A-B)) / 2
        sinA sinB = ( - cos(A+B) + cos(A-B)) / 2
        cosA sinB = (   sin(A+B) - sin(A-B)) / 2
        """
        if n == m:
            if n <= self.NN:
                kin = (n*self.w)**2 # kin = 1/2 * (n*self.w)**2
            else:
                kin = ((n-self.NN) * self.w)**2 # kin = 1/2 * ((n-self.NN) * self.w)**2
        else:
            kin = 0

        fac = 1
        if n == 0:
            fac *= 1/np.sqrt(2)
        if m == 0:
            fac *= 1/np.sqrt(2)

        if (n <= self.NN) and (m <= self.NN):
            nn = n
            mm = m
            int0 = self.cos_list[int(nn+mm)]
            int1 = self.cos_list[int(np.abs(nn-mm))]
            potential = (int0 + int1) / self.T
        elif (n <= self.NN) and (m > self.NN):
            nn = n
            mm = m - self.NN
            int0 = self.sin_list[int(nn+mm)]
            int1 = self.sin_list[int(np.abs(nn-mm))] * np.sign(nn-mm)
            potential = (int0 - int1) / self.T
        elif (n > self.NN) and (m > self.NN):
            nn = n - self.NN
            mm = m - self.NN
            int0 = self.cos_list[int(nn+mm)]
            int1 = self.cos_list[int(np.abs(nn-mm))]
            potential = (- int0 + int1) / self.T
        else:
            raise ValueError('n and m must be <= N/2')

        return fac * potential + kin

    def get_sin_cos_list(self):
        if not hasattr(self, 'sin_list'):
            self._gen_sin_cos_list()
        return self.sin_list, self.cos_list

    @timer
    def _gen_sin_cos_list(self):
        self.sin_list = []
        self.cos_list = []
        for i in range(self.N):
            if self.verbose > 0:
                print('=== calculating cos and sin ===', i, '===', end='\r')
            self.cos_list.append(quad_break(self.V,
                                            self.a, self.b,
                                            points=self.points,
                                            weight='cos',
                                            wvar=i*self.w)[0])
            self.sin_list.append(quad_break(self.V, self.a, self.b,
                                            points=self.points,
                                            weight='sin',
                                            wvar=i*self.w)[0])
        if self.verbose > 0:
            print('=========== calculating cos and sin  FINISHED! ====')


def normallizeV(V: callable, w, N, energy_cut=0, right_cut=True):
    """
    cut the divergent part of the potential for 3D centrifuge potential.
    Assume the potential is from 0 to some value

    right cut: if we need to cut the right part of the potential.
    """
    T = 2*np.pi/w
    if energy_cut == 0:
        energy_cut = N*w*10
    r0 = root_scalar(lambda x: V(x)-energy_cut, method='newton', x0=1e-10).root
    if right_cut:
        r1 = root_scalar(lambda x: V(x)-energy_cut,
                         method='newton', x0=T).root
    else:
        r1 = np.inf

    def normedV(x):
        x += 1e-10
        get_outer = -np.sign((x-r0)*(r1-x))/2 + 1/2
        get_inner = 1 - get_outer
        return get_outer*energy_cut + get_inner*V(x)
    points = []
    for r in [r0, r1]:
        if r < T:
            points.append(r)
    return normedV, points


class StepedHarmonic:
    """
    Find the bound state energy of a 3D steped harmonic potential.
    The potential is
    V(r) = -V0 + r**2/2,  for r<a
    V(r) = r**2/2,  for r>a

    hyperu, hyp1f1 is M(a, b, z) and U(a, b, z)
    https://en.wikipedia.org/wiki/Confluent_hypergeometric_function
    ref: PRL 94, 023202 (2005)
    """
    def __init__(self, a, V0, L):
        """
        a: step potential width
        V0: step potential deepth
        L: angular momentum quantum number
        """
        self.a = a
        self.V0 = V0
        self.L = L

    def R_in(self, nu, r, derivative=False):
        """
        ref: PRL 94, 023202 (2005) Eq. (13).
        R_l^- in PRL 94, 023202 (2005) Eq. (13) is not the same as here.
        R_in satisfies the radial schrodinger equation.
        nu is the vibration quantum number.
        The total energy is E = 2*nu + l + 3/2
        """

        if derivative:
            res = (1+self.L-r**2) * hyp1f1(-nu, self.L+3/2, r**2)
            res += 2*r**2 * hyp1f1_dz(-nu, self.L+3/2, r**2)
            res *= r**self.L * np.exp(-r**2/2)
        else:
            res = r**(self.L+1) * np.exp(-r**2/2)
            res *= hyp1f1(-nu, self.L+3/2, r**2)
        return res

    def R_out(self, nu, r, derivative=False):
        """
        ref: PRL 94, 023202 (2005) Eq. (14). with a r difference.
        """
        if derivative:
            res = (1+self.L-r**2) * hyperu(-nu, self.L+3/2, r**2)
            res += 2*r**2 * hyperu_dz(-nu, self.L+3/2, r**2)
            res *= r**self.L * np.exp(-r**2/2)
        else:
            res = r**(self.L+1) * np.exp(-r**2/2)
            res *= hyperu(-nu, self.L+3/2, r**2)
        return res

    def log_dev(self, a, nu, V0, r_number=1000):
        """
        The log derivate difference between R_out and R_in.
        r_number: the number of step for numerical derivative.
        """
        nu_in = nu + V0/2
        res_in = self.R_in(nu_in, a, derivative=True)
        res_in /= self.R_in(nu_in, a, derivative=False)

        res_out = self.R_out(nu, a, derivative=True)
        res_out /= self.R_out(nu, a, derivative=False)
        res = (res_in - res_out)
        return res

    def get_bound_state_energy(self, check=True, precision=1e-4, nu_max=1,
                               swEb=-2):
        """Find the bound state energy."""
        nu = np.arange(swEb/2-self.L/2-3/4, nu_max, precision)
        E = 2*nu + self.L + 3/2
        log_devs = np.zeros(len(nu))
        for i, nui in enumerate(nu):
            print('Finding bound state energy ...', i, '/', len(nu), end='\r')
            log_devs[i] = self.log_dev(a=self.a, nu=nui, V0=self.V0)
        print('\n')
        idx = root_idx(log_devs)
        self.Eb = E[idx]
        if check:
            print('bound state energy is:', self.Eb)
            print('\n')
            plt.plot(E, self.R_in(nu, self.a), 'r--')
            plt.plot(E, self.R_out(nu, self.a), 'b--')
            plt.plot(E, log_devs)
            plt.plot(E, E*0)
            plt.show()
        return self.Eb


class StepWell:
    """
    Find the bound state energy of a 3D step well potential.
    The potential is
    V(r) = -V0,  for r<a
    V(r) = 0,  for r>a
    """
    def __init__(self, L):
        """
        a: step potential width
        V0: step potential deepth
        L: angular momentum quantum number
        """
        # self.a = a
        # self.V0 = V0
        self.L = L

    def R_out(self, r, kap, derivative=False):
        """
        The wave function R_out satisfies the radial schrodinger equation.
        When r > a, i.e. out the well, the wave function is the linear
        combination of the spherical Bessel function jn and yn.
        """
        tmp = 1j*spherical_jn(self.L, 1j*kap*r, derivative=False)
        tmp += - spherical_yn(self.L, 1j*kap*r, derivative=False)
        if derivative:
            tmp_prime = 1j*spherical_jn(self.L, 1j*kap*r, derivative=True)
            tmp_prime += - spherical_yn(self.L, 1j*kap*r, derivative=True)
            res = 1j*kap * tmp
            res += 1j*kap*r * 1j*kap*tmp_prime
        else:
            res = 1j*kap*r * tmp
        return res

    def R_in(self, r, k, derivative=False):
        """
        The wave function R_in satisfies the radial schrodinger equation.
        When r < a, i.e. in the well, the wave function is the spherical Bessel
        function jn.
        """
        if derivative:
            res = k*spherical_jn(self.L, k*r)
            res += k**2*r*spherical_jn(self.L, k*r, derivative=True)
        else:
            res = k*r*spherical_jn(self.L, k*r)
        return res

    def log_dev(self, r, kap, V0):
        """
        The log derivate difference between R_out and R_in.
        """
        k = np.sqrt(2*(V0 - kap**2/2))
        res_in = self.R_in(r, k, derivative=True)
        res_in /= self.R_in(r, k, derivative=False)
        res_out = self.R_out(r, kap, derivative=True)
        res_out /= self.R_out(r, kap, derivative=False)
        return (res_in - res_out).real

    def get_bound_state_energy(self, a, V0, check=True):
        """Find the bound state energy."""
        E = np.linspace(-V0+1e-9, 0, 1000)
        kaps = np.sqrt(2*(-E))
        log_devs = np.array([self.log_dev(r=a, kap=ki, V0=V0)
                             for ki in kaps])
        self.Eb = E[root_idx(log_devs)]
        if check:
            print('bound state energy is:', self.Eb)
            plt.plot(E, log_devs)
            plt.plot(E, E*0)
            plt.show()
        return self.Eb

    def get_width(self, V0, Eb, check=True):
        """For a given potential depth and bound state energy, find the
        potental width (the shallowets case)."""
        kap = np.sqrt(-2*Eb)
        res = root_scalar(lambda a: self.log_dev(a, kap, V0), x0=1e-3)
        width = res.root
        if check:
            print("if converged:", res.converged)
        return width
