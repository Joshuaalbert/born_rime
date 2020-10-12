import numpy as np
from scipy.special import kv
import astropy.units as au
import astropy.constants as constants
from astropy.units.quantity import Quantity
from typing import List, Tuple
import pylab as plt
# from mayavi import mlab
from born_rime.utils import _validate_type, _validate_unit_type
from born_rime.plotting import add_colorbar

Gauss = 1e-4 * au.T
au.set_enabled_equivalencies(au.dimensionless_angles())


class CartesianSpaceDomain(object):
    space_grid: Quantity
    index_grid: np.ndarray
    plot_slices: List[Tuple[Tuple[int, int], Tuple]]

    def __init__(self, box, shape):
        self.box = box
        self.shape = shape

    def build(self):
        self.space_grid = self.compute_space_grid()
        self.index_grid = self.compute_index_grid()
        self.plot_slices = self.compute_plot_slices()

    def fourier_space(self):
        """
        Build the Fourier space of this object.
        k = s L n/(n-1)
        -> s = k (n-1)/(n L) = k ds
        ds = (n-1)/(n L) = Ls/(n-1)
        -> Ls = (n-1)^2 / (n L)
        :return: CartesianSpaceDomain of the Fourier space
        """
        n = np.array(self.shape)
        Ls = (n - 1) ** 2 / (n * self.L)
        box = [Quantity([-ls / 2., ls / 2.]) for ls in Ls]
        return CartesianSpaceDomain(box, self.shape)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        for s in value:
            _validate_type('shape', s, int)
            if s % 2 != 1:
                raise ValueError("shape must be odd, got {}".format(s))
            if s < 3:
                raise ValueError("shape must be at least 3, got {}".format(s))
        self._shape = tuple(value)

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, value):
        for bounds in value:
            _validate_type('bounds', bounds, Quantity)
            if bounds[1] <= bounds[0]:
                raise ValueError('bounds must be different and ordered, got {}'.format(bounds))
        self._box = tuple(value)

    @property
    def extents(self):
        return [bounds for bounds in self.box]

    @property
    def origin_indices(self):
        return np.floor((-self.xmin / self.dx).value).astype(int)

    @property
    def xmin(self):
        return Quantity([b[0] for b in self.box])

    @property
    def xmax(self):
        return Quantity([b[1] for b in self.box])

    @property
    def L(self):
        return Quantity([b[1] - b[0] for b in self.box])

    @property
    def dx(self):
        return Quantity([L / (n - 1) for L, n in zip(self.L, self.shape)])

    def compute_space_grid(self):
        """
        Compute the grid of spatial locations of domain.
        :return: Quantity
            [N1, ..., ND, D]
        """
        vecs = [xmin + np.arange(n) * dx for (xmin, dx, n) in zip(self.xmin, self.dx, self.shape)]
        grid_locs = np.stack([v.flatten() for v in np.meshgrid(*vecs, indexing='ij')], axis=0).reshape(
            (-1,) + self.shape)
        return Quantity(grid_locs, copy=False)

    def compute_index_grid(self):
        """
        Compute the grid of indices.
        :return: Quantity
            [N1, ..., ND, D]
        """
        vecs = [np.arange(n) for n in self.shape]
        grid_locs = np.stack([v.flatten() for v in np.meshgrid(*vecs, indexing='ij')], axis=0).reshape(
            (-1,) + self.shape)
        return grid_locs.astype(int)

    def compute_plot_slices(self):
        """
        Compute the indexing slices that map a field over the space to slice in the pair-wise planes passing through origin.
        """
        res = []
        for d1 in range(len(self.shape)):
            for d2 in range(d1 + 1, len(self.shape)):
                slices = []
                for i in range(len(self.shape)):
                    if i not in [d1, d2]:
                        slices.append(slice(self.origin_indices[i], self.origin_indices[i] + 1, 1))
                    else:
                        slices.append(slice(None, None, None))
                res.append(((d1, d2), tuple(slices)))
        return res

    def __eq__(self, other):
        return np.all(other.L == self.L) and np.all(other.shape == self.shape)

    def __repr__(self):
        return "Space shape:{} box:{} lengths:{} dx:{}".format(self.shape, self.box, self.L, self.dx)


def test_cartesian_space_domain():
    box = [[-1.5, 1.] * au.m, [-1., 1.5] * au.m]
    shape = [5, 7]
    space = CartesianSpaceDomain(box, shape)
    print(space)
    assert space == space
    f_space = space.fourier_space()
    print(f_space)
    assert space.L.unit == 1. / f_space.L.unit
    space_2 = f_space.fourier_space()
    # assert space_2 == space
    space.build()
    assert space.space_grid.unit == space.L.unit
    assert space.space_grid.shape == tuple([2] + shape)
    assert isinstance(space.space_grid, Quantity)


class FEDModel(object):
    fed: Quantity

    def __init__(self, space_domain: CartesianSpaceDomain):
        self.space_domain = space_domain

    def build(self):
        self.fed = self.compute_fed(self.space_domain.space_grid)

    def plot(self):
        for (d1, d2), plot_slices in self.space_domain.plot_slices:
            img = self.fed[plot_slices].reshape((self.space_domain.shape[d1], self.space_domain.shape[d2])).to(
                1 / au.cm ** 3)
            y_label = "Dim{} [{}]".format(d1, self.space_domain.extents[d1].unit)
            x_label = "Dim{} [{}]".format(d2, self.space_domain.extents[d2].unit)
            extent = list(self.space_domain.extents[d2].value) + list(self.space_domain.extents[d1].value)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, aspect='auto')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.colorbar(sc, label=r'$n_e$ [{}]'.format(img.unit))
            plt.show()

    def compute_fed(self, space_coords: Quantity):
        """
        Compute the FED over the space.
        """
        raise NotImplementedError()


class CosExp(FEDModel):
    def __init__(self, mean: Quantity = 6e9 / au.m ** 3, amp: Quantity = 3e12 / au.m ** 3,
                 height: Quantity = 250 * au.km, thickness: Quantity = 20 * au.km, period: Quantity = 5 * au.km,
                 space_domain: CartesianSpaceDomain = None):
        super(CosExp, self).__init__(space_domain=space_domain)
        self.period = period
        self.height = height
        self.thickness = thickness
        self.amplitude = amp
        self.mean = mean

    def compute_fed(self, space_coords: Quantity):
        if len(self.space_domain.shape) == 2:
            return Quantity(
                self.mean + self.amplitude * np.cos(2 * np.pi * space_coords[-2, ...] / self.period) * np.exp(
                    -0.5 * np.square(space_coords[-1, ...] - self.height) / np.square(self.thickness)), copy=False)

        if len(self.space_domain.shape) == 3:
            return Quantity(
                self.mean + self.amplitude * np.cos(2 * np.pi * space_coords[-3, ...] / self.period) * np.exp(
                    -0.5 * np.square(space_coords[-1, ...] - self.height) / np.square(self.thickness)), copy=False)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        _validate_unit_type('amplitude', value, 1 / au.m ** 3)
        self._amplitude = value

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        _validate_unit_type('mean', value, 1 / au.m ** 3)
        self._mean = value

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        _validate_unit_type('period', value, au.m)
        self._period = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        _validate_unit_type('height', value, au.m)
        self._height = value

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        _validate_unit_type('thickness', value, au.m)
        self._thickness = value

    def __repr__(self):
        return "CosExp ionosphere, height:{} thickness:{} period:{} amplitude:{}".format(self.height, self.thickness,
                                                                                         self.period, self.amplitude)


def test_cosexp():
    box = [Quantity([-10, 20] * au.km), Quantity([-10, 20] * au.km), Quantity([0., 300] * au.km)]
    shape = [51, 21, 301]
    space = CartesianSpaceDomain(box, shape)
    space.build()
    fed_model = CosExp(space_domain=space)
    fed_model.build()
    assert fed_model.fed.shape == tuple(shape)
    assert au.get_physical_type(fed_model.fed.unit) == au.get_physical_type(au.Unit(1 / au.m ** (3)))
    fed_model.plot()


class ThinCircularHole(FEDModel):
    def __init__(self, hole: Quantity = 3e9 / au.m ** 3, layer: Quantity = 3e13 / au.m ** 3,
                 height: Quantity = 250 * au.km, thickness: Quantity = 20 * au.km, diameter: Quantity = 5 * au.km,
                 space_domain: CartesianSpaceDomain = None):
        super(ThinCircularHole, self).__init__(space_domain=space_domain)
        self.diameter = diameter
        self.height = height
        self.thickness = thickness
        self.hole = hole
        self.layer = layer

    def compute_fed(self, space_coords: Quantity):
        if len(self.space_domain.shape) == 2:
            where_hole = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                        np.abs(space_coords[-2, ...]) < self.diameter / 2.)

            where_layer = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                         np.abs(space_coords[-2, ...]) >= self.diameter / 2.)
            return Quantity(
                self.hole * where_hole + self.layer * where_layer, copy=False)

        if len(self.space_domain.shape) == 3:
            where_hole = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                        np.linalg.norm(space_coords[-3:-1, ...], axis=0) < self.diameter / 2.)
            where_layer = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                         np.linalg.norm(space_coords[-3:-1, ...], axis=0) >= self.diameter / 2.)
            return Quantity(
                self.hole * where_hole + self.layer * where_layer, copy=False)


class BModel(object):
    theta: Quantity
    B0: Quantity

    def __init__(self, space_domain: CartesianSpaceDomain):
        self.space_domain = space_domain

    def build(self):
        self.theta = self.compute_theta()
        self.B0 = self.compute_B0()

    def compute_B0(self):
        raise NotImplementedError()

    def compute_theta(self):
        raise NotImplementedError()


class ConstantParallelB(BModel):

    def __init__(self, constant_B=0.25 * Gauss, space_domain: CartesianSpaceDomain = None):
        super(ConstantParallelB, self).__init__(space_domain=space_domain)
        self.constant_B = constant_B

    def compute_B0(self):
        return self.constant_B  # * np.ones(self.space_domain.shape)

    def compute_theta(self):
        return 0. * au.rad


class RefractiveIndexModel(object):
    n_squared: Quantity
    n: Quantity

    def __init__(self, space_domain: CartesianSpaceDomain = None):
        self.space_domain = space_domain

    def build(self):
        self.n_squared = self.compute_n_squared()
        self.n = np.sqrt(self.n_squared)

    def compute_n_squared(self):
        raise NotImplementedError()


class DiffractionHole(RefractiveIndexModel):
    def __init__(self, hole_n: Quantity = 1., layer_n: Quantity = 2.,
                 height: Quantity = 250 * au.m, thickness: Quantity = 20 * au.m, diameter: Quantity = 5 * au.m,
                 space_domain: CartesianSpaceDomain = None):
        super(DiffractionHole, self).__init__(space_domain=space_domain)
        self.diameter = diameter
        self.height = height
        self.thickness = thickness
        self.hole_n = hole_n
        self.layer_n = layer_n

    def compute_n_squared(self):
        space_coords = self.space_domain.space_grid
        if len(self.space_domain.shape) == 2:
            where_hole = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                        np.abs(space_coords[-2, ...]) < self.diameter / 2.)

            where_layer = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                         np.abs(space_coords[-2, ...]) >= self.diameter / 2.)
            return Quantity(
                self.hole_n ** 2 * where_hole + self.layer_n ** 2 * where_layer, copy=False)

        if len(self.space_domain.shape) == 3:
            where_hole = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                        np.linalg.norm(space_coords[-3:-1, ...], axis=0) < self.diameter / 2.)
            where_layer = np.logical_and(np.abs(space_coords[-1, ...] - self.height) < self.thickness,
                                         np.linalg.norm(space_coords[-3:-1, ...], axis=0) >= self.diameter / 2.)
            return Quantity(
                self.hole_n ** 2 * where_hole + self.layer_n ** 2 * where_layer, copy=False)


class IonosphereModel(RefractiveIndexModel):
    '''
    A model of an ionosphere with constant magnetic field.
    '''

    def __init__(self, nu: float, space_domain: CartesianSpaceDomain, fed_model: FEDModel, B_model: BModel):
        super(IonosphereModel, self).__init__(space_domain=space_domain)
        self.fed_model = fed_model
        self.B_model = B_model
        self.nu = nu

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        _validate_unit_type('nu', value, 1 / au.s)
        self._nu = value

    def compute_n_squared(self):
        return self.compute_lassen_apple_hartree(self.fed_model.fed, self.B_model.B0, self.B_model.theta)

    def _plasma_freqency_squared(self, fed):
        omega_p_squared = fed * (constants.e.si ** 2 / constants.eps0 / constants.m_e)
        return omega_p_squared

    def _gyro_frequency(self, B0):
        return B0 * constants.e.si / constants.m_e

    def compute_lassen_apple_hartree(self, ne, B0, theta):
        omega_0_squared = self._plasma_freqency_squared(ne)
        # omega_H = self._gyro_frequency(B0)
        X = omega_0_squared / (2 * np.pi * self.nu) ** 2
        # Y = omega_H / np.sqrt(omega_0_squared)
        #
        # pol_term = np.sqrt((0.5*Y**2 * np.sin(theta)**2)**2 + (1. - X)**2 * Y**2 * np.cos(theta)**2)
        # A = 1. - X - 0.5*Y**2 * np.sin(theta)**2
        # n2_right = 1. - X*(1. - X)/(A - pol_term)
        # n2_left = 1. - X*(1. - X)/(A + pol_term)
        return 1. - X  # 0.5*(n2_left + n2_right)

    def plot(self):
        for (d1, d2), plot_slices in self.space_domain.plot_slices:
            img = self.n[plot_slices].reshape((self.space_domain.shape[d1], self.space_domain.shape[d2]))
            y_label = "Dim{} [{}]".format(d1, self.space_domain.extents[d1].unit)
            x_label = "Dim{} [{}]".format(d2, self.space_domain.extents[d2].unit)
            extent = list(self.space_domain.extents[d2].value) + list(self.space_domain.extents[d1].value)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, aspect='auto')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.colorbar(sc, label=r'$n$ [{}]'.format(img.unit))
            plt.show()


def test_ionosphere_model():
    box = [Quantity([-10, 20] * au.km), Quantity([0., 300] * au.km)]
    shape = [51, 301]
    space = CartesianSpaceDomain(box, shape)
    space.build()
    fed_model = CosExp(mean=1e12 / au.m ** 3, amp=1e11 / au.m ** 3, space_domain=space)
    fed_model.build()
    fed_model.plot()
    B_model = ConstantParallelB(space_domain=space)
    B_model.build()
    ionosphere_model = IonosphereModel(150e6 / au.s, space, fed_model, B_model)
    ionosphere_model.build()
    ionosphere_model.plot()


class ScatteringPotential(object):
    potential: Quantity
    preconditioner: Quantity
    delta: Quantity

    def __init__(self, nu: Quantity = 150e6 / au.s, space_domain: CartesianSpaceDomain = None,
                 refractive_index_model: RefractiveIndexModel = None,
                 boundary_thickness_lambda: float = 25.,
                 boundary_N: int = 4,
                 boundary_depth: float = 3.):
        self.space_domain = space_domain
        self.refractive_index_model = refractive_index_model
        self.nu = nu
        self.boundary_thickness_lambda = boundary_thickness_lambda
        self.boundary_N = boundary_N
        self.boundary_depth = boundary_depth

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        _validate_unit_type('nu', value, 1 / au.s)
        self._nu = value

    @property
    def wavelength(self):
        return constants.c / self.nu

    def build(self):
        self.k2 = 4. * np.pi ** 2 * self.refractive_index_model.n_squared / self.wavelength ** 2
        self.k0 = 0.5 * np.max(np.real(np.sqrt(self.k2))) + 0.5 * np.min(np.real(np.sqrt(self.k2)))
        self.delta = 1.2 * np.max(np.abs(self.k2 - self.k0 ** 2)) + 0.1 * self.k0 ** 2
        self.a_squared = -self.k0 ** 2 - 1j * self.delta

        self.potential = self.compute_potential()
        self.potential = self.compute_boundary_potential()
        self.preconditioner = self.compute_preconditioner()

    def compute_boundary_potential(self):
        """
        Compute
        """

        thickness = self.boundary_thickness_lambda * self.wavelength
        alpha = 1. / (thickness / self.boundary_depth)

        def _poly(x):
            ax = alpha * x
            res = np.ones_like(ax)
            for n in range(1, self.boundary_N + 1):
                res = res + res * ax / n
            return res

        potential = self.potential.copy()

        dx = self.space_domain.dx
        for d in range(len(dx)):
            potential_roll = np.moveaxis(potential, d, -1)
            num_blocks = max(3, int(thickness / dx[d]) + 1)
            x = np.arange(num_blocks) * dx[d]
            Px = _poly(x)
            k2_k02 = alpha ** 2 * (1. - alpha * x + 2j * self.k0 * x) * (alpha * x) ** (self.boundary_N - 1) / (
                        Px * np.math.factorial(self.boundary_N))
            potential_boundary = k2_k02 - 1j * self.delta
            potential_roll[..., :num_blocks] = potential_boundary[::-1]
            potential_roll[..., -num_blocks:] = potential_boundary
            potential = np.moveaxis(potential_roll, -1, d)

        return potential

    def compute_potential(self):
        """
        Compute k^2 - k0^2 - i delta
        """
        return self.k2 - self.k0 ** 2 - 1j * self.delta

    def compute_preconditioner(self):
        return 1j * (self.potential / self.delta)

    def plot(self):
        for (d1, d2), plot_slices in self.space_domain.plot_slices:
            img = np.abs(
                self.potential[plot_slices].reshape((self.space_domain.shape[d1], self.space_domain.shape[d2])))
            y_label = "Dim{} [{}]".format(d1, self.space_domain.extents[d1].unit)
            x_label = "Dim{} [{}]".format(d2, self.space_domain.extents[d2].unit)
            extent = list(self.space_domain.extents[d2].value) + list(self.space_domain.extents[d1].value)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, aspect='auto')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.colorbar(sc, label=r'|potential| [{}]'.format(img.unit))
            plt.show()

            img = np.angle(
                self.potential[plot_slices].reshape((self.space_domain.shape[d1], self.space_domain.shape[d2])))
            y_label = "Dim{} [{}]".format(d1, self.space_domain.extents[d1].unit)
            x_label = "Dim{} [{}]".format(d2, self.space_domain.extents[d2].unit)
            extent = list(self.space_domain.extents[d2].value) + list(self.space_domain.extents[d1].value)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, aspect='auto')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.colorbar(sc, label=r'Arg(potential) [{}]'.format(img.unit))
            plt.show()

            # img = np.abs(
            #     self.preconditioner[plot_slices].reshape((self.space_domain.shape[d1], self.space_domain.shape[d2])))
            # y_label = "Dim{} [{}]".format(d1, self.space_domain.extents[d1].unit)
            # x_label = "Dim{} [{}]".format(d2, self.space_domain.extents[d2].unit)
            # extent = list(self.space_domain.extents[d2].value) + list(self.space_domain.extents[d1].value)
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, aspect='auto')
            # ax.set_xlabel(x_label)
            # ax.set_ylabel(y_label)
            # plt.colorbar(sc, label=r'Abs(preconditioner) [{}]'.format(img.unit))
            # plt.show()


def test_potential():
    box = [Quantity([-500., 500] * au.m), Quantity([0., 1000] * au.m)]
    shape = [1001, 1001]
    space = CartesianSpaceDomain(box, shape)
    space.build()
    print(space)
    fed_model = CosExp(height=200 * au.m, thickness=10 * au.m, period=5 * au.m, space_domain=space)
    # fed_model = ThinCircularHole(height=200*au.corner_indices,thickness=10*au.corner_indices,diameter=1.*au.corner_indices,space_domain=space)
    fed_model.build()
    fed_model.plot()
    B_model = ConstantParallelB(space_domain=space)
    B_model.build()
    ionosphere_model = IonosphereModel(150e6 / au.s, space, fed_model, B_model)
    ionosphere_model.build()
    ionosphere_model.plot()
    potential = ScatteringPotential(150e6 / au.s, space, ionosphere_model, boundary_thickness_lambda=3.)
    potential.build()
    potential.plot()

    n_model = DiffractionHole(space_domain=space)
    n_model.build()
    potential = ScatteringPotential(150e6 / au.s, space, n_model, boundary_thickness_lambda=3.)
    potential.build()
    potential.plot()


class FunctionFourier(object):
    forward_translation_factor: Quantity

    def __init__(self, space_domain: CartesianSpaceDomain = None):
        self.space_domain = space_domain

    def _broadcast_prod(self, array):
        arrays = np.meshgrid(*[x * np.ones(s) for (x, s) in zip(array, self.space_domain.shape)], indexing='ij')
        X = arrays[0]
        for a in arrays[1:]:
            X = X * a
        return X

    def build(self):
        self.fourier_space = self.space_domain.fourier_space()
        self.fourier_space.build()
        D = self.fourier_space.space_grid.shape[0]
        # D, N1,...,Nd
        s = self.fourier_space.space_grid
        # N1,...,Nd
        sxmin = np.sum(np.stack(
            np.meshgrid(*[x * np.ones(n) for (x, n) in zip(self.space_domain.xmin, self.space_domain.shape)],
                        indexing='ij'), axis=0) * s, axis=0)
        # N1,...,Nd
        dx = self._broadcast_prod(self.space_domain.dx)
        # N1,...,Nd
        self.forward_translation_factor = np.exp(-1j * 2 * np.pi * sxmin) * dx

        # self.forward_translation_factor = self._broadcast_prod(self.space_domain.dx * np.exp(-2j*np.pi*))

        # N1,...,Nd
        xsmin = np.sum(np.stack(
            np.meshgrid(*[x * np.ones(n) for (x, n) in zip(self.fourier_space.xmin, self.fourier_space.shape)],
                        indexing='ij'), axis=0) * self.space_domain.space_grid, axis=0)

        # ds = self._broadcast_prod(self.fourier_space.dx)
        self.inverse_translation_factor = np.reciprocal(dx) * np.exp(1j * 2. * np.pi * xsmin)

    def fourier_transform(self, H):
        """
        Fourier transform `H`
        """
        shift_axes = list(range(-1, -len(self.space_domain.shape) - 1, -1))
        F = np.fft.fftn(H, s=self.space_domain.shape)
        return np.fft.fftshift(F, axes=shift_axes) * self.forward_translation_factor
        # return F * self.forward_translation_factor

    def inverse_fourier_transform_old(self, F):
        """
        Inverse Fourier transform of `F`
        """
        shift_axes = list(range(-1, -len(self.space_domain.shape) - 1, -1))
        F = F / self.forward_translation_factor
        F = np.fft.ifftshift(F, axes=shift_axes)
        H = np.fft.ifftn(F, s=self.space_domain.shape)
        return H

    def inverse_fourier_transform(self, F):
        """
        Inverse Fourier transform of `F`
        """
        shift_axes = list(range(-1, -len(self.space_domain.shape) - 1, -1))
        F = np.fft.ifftshift(F * self.inverse_translation_factor, axes=shift_axes)
        H = np.fft.ifftn(F, s=self.space_domain.shape)
        return H


def test_verify_fft():
    n = 300
    k = np.arange(n)
    for m in range(n):
        A = (np.arange(n) == m).astype(float)
        expect = np.exp(-1j * 2 * np.pi * k * m / n)
        assert np.all(np.isclose(np.fft.fft(A), expect))


def test_fourier():
    A = np.random.normal(size=(3, 3, 3))
    assert np.all(np.isclose(np.moveaxis(np.moveaxis(A, 1, -1), -1, 1), A))
    assert np.all(np.isclose(np.moveaxis(A[0, :, :], 0, -1), A[0, :, :].T))

    # box = [Quantity([-10, 20.] * au.k0), Quantity([-10, 10.] * au.k0)]
    # shape = [31, 31]
    # space = CartesianSpaceDomain(box, shape)
    # space.build()
    # H = np.exp(-0.5 * np.linalg.norm(space.space_grid, axis=0) ** 2 / (3 * au.k0) ** 2)
    # fourier = FunctionFourier(space_domain=space)
    # fourier.build()
    # F = fourier.fourier_transform(H)
    # assert F.unit == au.k0 ** 2
    # # plt.imshow(F.real)
    # # plt.colorbar()
    # # plt.show()
    #
    # k = np.linalg.norm(fourier.fourier_space.space_grid, axis=0)
    # analytic = 2 * np.pi * (3 * au.k0) ** 2 * np.exp(-2. * np.pi ** 2 * (3 * au.k0) ** 2 * k ** 2)
    #
    # assert np.all(np.isclose(F.real, analytic.real, atol=1e-1))
    # # plt.imshow(analytic.real.value  - F.real.value)
    # # plt.colorbar()
    # # plt.show()
    #
    # H_rec = fourier.inverse_fourier_transform(analytic)
    # assert np.all(np.isclose(H_rec.real, H, atol=1e-2))
    #
    # # plt.imshow(H)
    # # plt.colorbar()
    # # plt.show()
    #
    # # plt.imshow(H_rec.real - H)
    # # plt.colorbar()
    # # plt.show()

    def normal_dist(mu, var):
        def _func(x):
            return np.sqrt(2 * np.pi * var) ** (-1.) * np.exp(-0.5 * (x - mu) ** 2 / var)

        return _func

    def normal_fourier(mu, var):
        def _func(k):
            return np.exp(-2. * np.pi ** 2 * (var * k ** 2) - 2j * np.pi * mu * k)

        return _func

    f = normal_dist(0., 1.)
    g = normal_dist(5., 0.5)
    h = normal_dist(5., 1.5)
    H = normal_fourier(5., 1.5)

    box = [Quantity([-10., 10.])]
    shape = [201]
    space = CartesianSpaceDomain(box, shape)
    space.build()
    fourier = FunctionFourier(space_domain=space)
    fourier.build()
    # print(fourier.space_domain, fourier.fourier_space)
    F = fourier.fourier_transform(f(space.space_grid[0, :]))
    G = fourier.fourier_transform(g(space.space_grid[0, :]))
    H_fft = F * G

    assert np.all(np.isclose(H(fourier.fourier_space.space_grid[0, :]), H_fft, atol=1e-3))
    # plt.plot(fourier.fourier_space.space_grid[0,:], np.abs(H_fft),label='FFT_H')
    # plt.plot(fourier.fourier_space.space_grid[0,:], np.abs(H(fourier.fourier_space.space_grid[0,:])),label='analytic_H')
    # plt.legend()
    # plt.show()

    h_fft = fourier.inverse_fourier_transform(H_fft)
    assert np.all(np.isclose(h_fft, h(space.space_grid[0, :]), atol=1e-3))

    h_fft = fourier.inverse_fourier_transform(H(fourier.fourier_space.space_grid[0, :]))
    assert np.all(np.isclose(h_fft, h(space.space_grid[0, :]), atol=1e-3))
    # plt.plot(space.space_grid[0,:], h_fft,label='fft_h')
    # plt.plot(space.space_grid[0,:], h(space.space_grid[0,:]),label='analytic_h')
    # plt.legend()
    # plt.show()

    f = lambda x: np.prod(normal_dist(0., 1.)(x), axis=0)
    # f = lambda x: f(x[0, ...]) * f(x[1, ...])
    g = lambda x: np.prod(normal_dist(5., 0.5)(x), axis=0)
    h = lambda x: np.prod(normal_dist(5., 1.5)(x), axis=0)
    H = lambda x: np.prod(normal_fourier(5., 1.5)(x), axis=0)

    box = [Quantity([-10., 10.]), Quantity([-20., 20.])]
    shape = [201, 201]
    space = CartesianSpaceDomain(box, shape)
    space.build()
    fourier = FunctionFourier(space_domain=space)
    fourier.build()
    # print(fourier.space_domain, fourier.fourier_space)
    F = fourier.fourier_transform(f(space.space_grid))
    G = fourier.fourier_transform(g(space.space_grid))
    H_fft = F * G

    assert np.all(np.isclose(H(fourier.fourier_space.space_grid), H_fft, atol=1e-3))
    # plt.imshow(np.abs(H_fft),label='FFT_H')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.abs(H(fourier.fourier_space.space_grid)),label='analytic_H')
    # plt.colorbar()
    # plt.show()

    h_fft = fourier.inverse_fourier_transform(H_fft)
    # plt.imshow(np.abs(h_fft),label='fft_h')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(np.abs(h(space.space_grid)),label='analytic_h')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(np.abs(h_fft - h(space.space_grid)))
    # plt.colorbar()
    # plt.show()
    #

    assert np.all(np.isclose(h_fft, h(space.space_grid), atol=1e-3))
    assert np.all(
        np.isclose(fourier.inverse_fourier_transform(H(fourier.fourier_space.space_grid)), h(space.space_grid),
                   atol=1e-3))
    assert np.all(np.isclose(fourier.inverse_fourier_transform(np.tile(H_fft[None, ...], [2, 1, 1])),
                             np.tile(h(space.space_grid)[None, ...], [2, 1, 1]), atol=1e-3))


class GreensFunction(object):
    g: Quantity
    G: Quantity

    def __init__(self, scattering_potential: ScatteringPotential, fourier: FunctionFourier):
        self.scattering_potential = scattering_potential
        self.fourier = fourier

    def build(self):
        self.g = self.compute_g()
        self.G = self.compute_G()

    def compute_g(self):
        """
        Compute the Greens function over space.
        :return: [D, D, N1,...,ND]
        """
        raise NotImplementedError()

    def compute_G(self):
        """
        Compute the Fourier of g over Fourier space
        :return: [D, D, N1,...,ND]
        """
        raise NotImplementedError()


class TwoDimGreensFunction(GreensFunction):
    def compute_g(self):
        # [2,N1,N2]
        space_grid = self.fourier.space_domain.space_grid
        D = space_grid.shape[0]
        # [N1,N2]
        r = np.linalg.norm(space_grid, axis=0)
        ar = np.sqrt(self.scattering_potential.a_squared) * r
        r_norm = space_grid / r
        r_mn = r_norm[:, None, ...] * r_norm[None, :, ...]
        # diag terms
        A = np.eye(D)[:, :, None, None] * (
                    ar * (1. - r_norm[:, None, ...]) * (1. + r_norm[None, :, ...]) * kv(0., ar) + (1. - 2. * r_mn) * kv(
                1., ar)) / (2. * np.pi * ar)
        # off diag terms
        B = -(1. - np.eye(D)[:, :, None, None]) * r_mn * kv(2., ar) / (2. * np.pi)
        g = A + B
        return np.where(~np.isfinite(g), 1. + 0j, g)

    def compute_G(self):
        # [2,N1,N2]
        space_grid = (2. * np.pi) * self.fourier.fourier_space.space_grid
        D = space_grid.shape[0]
        # [N1, N2]
        k2 = np.sum(np.square(space_grid), axis=0)
        # [2,2,N1,N2]
        G = np.eye(D)[:, :, None, None] + (
                    (space_grid[:, None, ...] * space_grid[None, :, ...]) / self.scattering_potential.a_squared)
        G *= np.reciprocal(k2 + self.scattering_potential.a_squared)
        return G


class ThreeDimGreensFunction(GreensFunction):
    def compute_g(self):
        # [3,N1,N2,N3]
        space_grid = self.fourier.space_domain.space_grid
        D = space_grid.shape[0]
        # [N1,N2,N3]
        r = np.linalg.norm(space_grid, axis=0)
        ar = np.sqrt(self.scattering_potential.a_squared) * r

        f = np.exp(-ar) / (4. * np.pi * r)

        A = np.eye(D)[:, :, None, None, None] * (1. + ar * (1 + ar)) / np.square(ar)

        r_norm = space_grid / r
        B = -(3. + ar * (3. + ar)) * (r_norm[:, None, ...] * r_norm[None, :, ...]) / np.square(ar)

        g = f * (A + B)

        return g

    def compute_G(self):
        # [3,N1,N2,N3]
        space_grid = (2. * np.pi) * self.fourier.fourier_space.space_grid
        D = space_grid.shape[0]
        # [N1, N2, N3]
        k2 = np.sum(np.square(space_grid), axis=0)
        # [3,3,N1,N2,N3]
        G = np.eye(D)[:, :, None, None, None] + (
                    space_grid[:, None, ...] * space_grid[None, :, ...]) / self.scattering_potential.a_squared
        G *= np.reciprocal(k2 + self.scattering_potential.a_squared)

        return G


def test_green():
    A = np.random.normal(size=(2, 2))
    b = np.random.normal(size=(2))
    print(A.dot(b[:, None]), np.sum(A * b, axis=1))

    nu = 20e6 / au.s

    # print(kn(0,1 + 1j))

    box = [Quantity([-100, 100] * au.m), Quantity([-100, 100] * au.m)]
    shape = [101, 101]
    space = CartesianSpaceDomain(box, shape)
    space.build()
    # fed_model = CosExp(mean=1e12 / au.corner_indices ** 3, amp=1e11 / au.corner_indices ** 3, period=5 * au.k0, height=100 * au.k0,
    #                    thickness=10 * au.k0, space_domain=space)
    fed_model = ThinCircularHole(diameter=1 * au.m, thickness=1 * au.m, height=200 * au.m, space_domain=space)
    fed_model.build()
    B_model = ConstantParallelB(space_domain=space)
    B_model.build()
    ionosphere_model = IonosphereModel(nu, space, fed_model, B_model)
    ionosphere_model.build()
    potential = ScatteringPotential(nu, space, ionosphere_model)
    potential.build()
    fourier = FunctionFourier(space_domain=space)
    fourier.build()
    greens_function = TwoDimGreensFunction(potential, fourier)
    greens_function.build()

    G_fft = fourier.fourier_transform(greens_function.g)
    g_fft = fourier.inverse_fourier_transform(greens_function.G)

    def _extent(extents):
        return [extents[0][0].value, extents[0][1].value, extents[1][0].value, extents[1][1].value]

    fig, axs = plt.subplots(2, 2)
    img = axs[0][0].imshow(np.abs(G_fft[0, 0, :, :]), origin='lower', extent=_extent(fourier.fourier_space.extents))
    add_colorbar(img, '|G_fft|', ax=axs[0][0])
    img = axs[1][0].imshow(np.abs(greens_function.G[0, 0, :, :]), origin='lower',
                           extent=_extent(fourier.fourier_space.extents))
    add_colorbar(img, '|G|', ax=axs[1][0])

    img = axs[0][1].imshow(np.abs(g_fft[1, 0, :, :]), origin='lower', extent=_extent(fourier.space_domain.extents))
    add_colorbar(img, '|g_fft|', ax=axs[0][1])
    img = axs[1][1].imshow(np.abs(greens_function.g[1, 0, :, :]), origin='lower',
                           extent=_extent(fourier.space_domain.extents))
    add_colorbar(img, '|g|', ax=axs[1][1])

    plt.show()

    fig, axs = plt.subplots(2, 2)
    img = axs[0][0].imshow(np.angle(G_fft[0, 0, :, :]), origin='lower', extent=_extent(fourier.fourier_space.extents))
    add_colorbar(img, 'Ang G_fft', ax=axs[0][0])
    img = axs[1][0].imshow(np.angle(greens_function.G[0, 0, :, :]), origin='lower',
                           extent=_extent(fourier.fourier_space.extents))
    add_colorbar(img, 'Ang G', ax=axs[1][0])

    img = axs[0][1].imshow(np.angle(g_fft[0, 0, :, :]), origin='lower', extent=_extent(fourier.space_domain.extents))
    add_colorbar(img, 'Ang g_fft', ax=axs[0][1])
    img = axs[1][1].imshow(np.angle(greens_function.g[0, 0, :, :]), origin='lower',
                           extent=_extent(fourier.space_domain.extents))
    add_colorbar(img, 'Ang g', ax=axs[1][1])

    plt.show()


class InitialField(object):
    Ei: Quantity

    def __init__(self, s_hat: Quantity, nu: Quantity, E0: Quantity, space_domain: CartesianSpaceDomain):
        self.nu = nu
        # S, D
        self.E0 = E0
        self.space_domain = space_domain
        # S, D
        self.s_hat = s_hat
        if self.E0.shape != self.s_hat.shape:
            raise ValueError("E0 shape must match s_hat, got {} {}".format(self.E0.shape, self.s_hat.shape))

    def build(self):
        self.Ei = self.compute_Ei()

    def compute_Ei(self):
        # D, N1,...,ND
        r = self.space_domain.space_grid
        # D = r.shape[0]
        # ones = tuple(np.ones(D, dtype=np.int32))
        # # S, D, 1,...1
        # kvec = np.reshape(self.s_hat * (2. * np.pi / self.wavelength), self.s_hat.shape + ones)
        # # S, D, 1,...1
        # E0 = np.reshape(self.E0, self.E0.shape + ones)
        # # S,1, N1,...,ND
        # kr = np.sum(r * kvec, axis=1, keepdims=True)
        # S, D . D, N1,...,ND -> S,N1,...,ND
        kr = (2. * np.pi / self.wavelength) * np.tensordot(self.s_hat, r, axes=[[1], [0]])
        # S,D.S,N1,...,ND -> D,N1,...,ND
        Ei = Quantity(np.tensordot(self.E0, np.exp(1j * kr), axes=[[0], [0]]), copy=False)
        # # D, N1,...,ND
        # Ei = Quantity(np.sum(E0 * np.exp(1j * kr), axis=0), copy=False)
        return Ei

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        _validate_unit_type('nu', value, 1 / au.s)
        self._nu = value

    @property
    def E0(self):
        return self._E0

    @E0.setter
    def E0(self, value):
        _validate_unit_type('E0', value, au.Jy ** (0.5))
        self._E0 = value

    @property
    def wavelength(self):
        return constants.c / self.nu

    def plot(self):
        space_domain = self.space_domain
        D = self.Ei.shape[0]
        for i in range(D - 1):
            for (d1, d2), plot_slices in space_domain.plot_slices:
                fig = plt.figure(constrained_layout=False, figsize=(10, 5))

                gs = fig.add_gridspec(1, 2)
                # fig.subplots_adjust(top=0.87, left=0.1, right=0.95, bottom=0.1)

                ax = fig.add_subplot(gs[:, :1])

                img = np.angle(self.Ei[i, ...][plot_slices].reshape((space_domain.shape[d1], space_domain.shape[d2])))
                y_label = "Dim{} [{}]".format(d1, space_domain.extents[d1].unit)
                x_label = "Dim{} [{}]".format(d2, space_domain.extents[d2].unit)
                extent = list(space_domain.extents[d2].value) + list(space_domain.extents[d1].value)
                norm = plt.Normalize(img.min().value, img.max().value)
                sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, norm=norm, aspect='auto')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                add_colorbar(sc, r'Arg[Ei_{}] [{}]'.format(i, img.unit))

                # pos = ax.get_position()
                # cax = fig.add_axes([pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.02])
                #
                # cb1 = ColorbarBase(cax, cmap=plt.cm.binary,
                #                    norm=norm, orientation='horizontal')
                #
                # cb1.set_label(r'Arg[Es_{}] [{}]'.format(i,img.unit))
                #
                # cax.xaxis.set_ticks_position('top')
                # cax.xaxis.labelpad = -35

                ax = fig.add_subplot(gs[:, 1:2])

                img = np.abs(
                    self.Ei[i, ...][plot_slices].reshape((space_domain.shape[d1], space_domain.shape[d2])))
                y_label = "Dim{} [{}]".format(d1, space_domain.extents[d1].unit)
                x_label = "Dim{} [{}]".format(d2, space_domain.extents[d2].unit)
                extent = list(space_domain.extents[d2].value) + list(space_domain.extents[d1].value)
                norm = plt.Normalize(img.min().value, img.max().value)
                sc = ax.imshow(img, origin='lower', cmap='binary', norm=norm, extent=extent, aspect='auto')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                add_colorbar(sc, r'|Ei_{}| [{}]'.format(i, img.unit))

                # pos = ax.get_position()
                # cax = fig.add_axes([pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.02])
                #
                # cb1 = ColorbarBase(cax, cmap=plt.cm.binary,
                #                    norm=norm, orientation='horizontal')
                #
                # cb1.set_label(r'|Es_{}| [{}]'.format(i, img.unit))
                #
                # cax.xaxis.set_ticks_position('top')
                # cax.xaxis.labelpad = -35

                plt.show()


class BornSeries(object):
    electric_field: Quantity
    Es: Quantity

    def __init__(self,
                 n: int,
                 scattering_potential: ScatteringPotential,
                 greens_function: GreensFunction,
                 initial_field: InitialField,
                 function_fourier: FunctionFourier):
        self.n = n
        self.scattering_potential = scattering_potential
        self.greens_function = greens_function
        self.initial_field = initial_field
        self.function_fourier = function_fourier

    def build(self):
        self.electric_field = self.compute_electric_field()
        self.Es = self.compute_Es()

    def compute_Es(self):
        return self.electric_field - self.initial_field.Ei

    def compute_electric_field(self):
        # N1,...,ND
        potential = self.scattering_potential.potential
        # D,N1,...,ND
        Ei = self.initial_field.Ei
        S_factor = 4 * np.pi ** 2 / self.scattering_potential.wavelength ** 2 - self.scattering_potential.k2
        S = np.zeros_like(S_factor * Ei)
        # D,N1,...,ND
        E_n = Ei  # np.zeros_like(Ei)
        for n in range(self.n):
            # D,N1,...,ND
            H = potential * E_n + S
            # D, N1,...,ND
            F = self.function_fourier.fourier_transform(H)
            # D, D, N1,...,ND
            G = self.greens_function.G
            # D, N1,...,ND
            F = np.sum(G * F, axis=1)
            # D, N1,...,ND
            H2 = E_n - self.function_fourier.inverse_fourier_transform(F)
            # D, N1,...,ND
            E_np1 = E_n - self.scattering_potential.preconditioner * H2
            E_n = E_np1
        return E_n

    def plot(self):
        space_domain = self.scattering_potential.space_domain
        D = self.electric_field.shape[0]
        for i in range(D):
            for (d1, d2), plot_slices in space_domain.plot_slices:
                fig = plt.figure(constrained_layout=False, figsize=(10, 5))

                gs = fig.add_gridspec(1, 2)
                # fig.subplots_adjust(top=0.87, left=0.1, right=0.95, bottom=0.1)

                ax = fig.add_subplot(gs[:, :1])

                img = np.angle(self.Es[i, ...][plot_slices].reshape((space_domain.shape[d1], space_domain.shape[d2])))
                y_label = "Dim{} [{}]".format(d1, space_domain.extents[d1].unit)
                x_label = "Dim{} [{}]".format(d2, space_domain.extents[d2].unit)
                extent = list(space_domain.extents[d2].value) + list(space_domain.extents[d1].value)
                norm = plt.Normalize(img.min().value, img.max().value)
                sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, norm=norm, aspect='auto')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                add_colorbar(sc, r'Arg[Es_{}] [{}]'.format(i, img.unit))

                # pos = ax.get_position()
                # cax = fig.add_axes([pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.02])
                #
                # cb1 = ColorbarBase(cax, cmap=plt.cm.binary,
                #                    norm=norm, orientation='horizontal')
                #
                # cb1.set_label(r'Arg[Es_{}] [{}]'.format(i,img.unit))
                #
                # cax.xaxis.set_ticks_position('top')
                # cax.xaxis.labelpad = -35

                ax = fig.add_subplot(gs[:, 1:2])

                img = np.abs(
                    np.log(
                        self.Es.value[i, ...][plot_slices].reshape((space_domain.shape[d1], space_domain.shape[d2]))))
                y_label = "Dim{} [{}]".format(d1, space_domain.extents[d1].unit)
                x_label = "Dim{} [{}]".format(d2, space_domain.extents[d2].unit)
                extent = list(space_domain.extents[d2].value) + list(space_domain.extents[d1].value)
                norm = plt.Normalize((img).min(), (img).max())
                sc = ax.imshow(img, origin='lower', cmap='binary', norm=norm, extent=extent, aspect='auto')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                add_colorbar(sc, r'log |Es_{}| [{}]'.format(i, self.Es.unit))

                # pos = ax.get_position()
                # cax = fig.add_axes([pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.02])
                #
                # cb1 = ColorbarBase(cax, cmap=plt.cm.binary,
                #                    norm=norm, orientation='horizontal')
                #
                # cb1.set_label(r'|Es_{}| [{}]'.format(i, img.unit))
                #
                # cax.xaxis.set_ticks_position('top')
                # cax.xaxis.labelpad = -35

                plt.show()

    def plot_mayavi(self):
        space_domain = self.scattering_potential.space_domain
        D = self.electric_field.shape[0]
        mlab.quiver3d(np.log(np.abs(self.Es[0, ...]).value), np.log(np.abs(self.Es[1, ...]).value),
                      np.log(np.abs(self.Es[2, ...]).value))
        mlab.show()
        # for i in range(D):
        #     for (d1, d2), plot_slices in space_domain.plot_slices:
        #         fig = plt.figure(constrained_layout=False, figsize=(10, 5))
        #
        #         gs = fig.add_gridspec(1, 2)
        #         # fig.subplots_adjust(top=0.87, left=0.1, right=0.95, bottom=0.1)
        #
        #         ax = fig.add_subplot(gs[:, :1])
        #
        #         img = np.angle(self.Es[i, ...][plot_slices].reshape((space_domain.shape[d1], space_domain.shape[d2])))
        #         y_label = "Dim{} [{}]".format(d1, space_domain.extents[d1].unit)
        #         x_label = "Dim{} [{}]".format(d2, space_domain.extents[d2].unit)
        #         extent = list(space_domain.extents[d2].value) + list(space_domain.extents[d1].value)
        #         norm = plt.Normalize(img.min().value, img.max().value)
        #         sc = ax.imshow(img, origin='lower', cmap='binary', extent=extent, norm=norm, aspect='auto')
        #         ax.set_xlabel(x_label)
        #         ax.set_ylabel(y_label)
        #
        #         add_colorbar(sc, r'Arg[Es_{}] [{}]'.format(i, img.unit))
        #
        #         # pos = ax.get_position()
        #         # cax = fig.add_axes([pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.02])
        #         #
        #         # cb1 = ColorbarBase(cax, cmap=plt.cm.binary,
        #         #                    norm=norm, orientation='horizontal')
        #         #
        #         # cb1.set_label(r'Arg[Es_{}] [{}]'.format(i,img.unit))
        #         #
        #         # cax.xaxis.set_ticks_position('top')
        #         # cax.xaxis.labelpad = -35
        #
        #         ax = fig.add_subplot(gs[:, 1:2])
        #
        #         img = np.abs(
        #             np.log(self.Es.value[i, ...][plot_slices].reshape((space_domain.shape[d1], space_domain.shape[d2]))))
        #         y_label = "Dim{} [{}]".format(d1, space_domain.extents[d1].unit)
        #         x_label = "Dim{} [{}]".format(d2, space_domain.extents[d2].unit)
        #         extent = list(space_domain.extents[d2].value) + list(space_domain.extents[d1].value)
        #         norm = plt.Normalize((img).min(), (img).max())
        #         sc = ax.imshow(img, origin='lower', cmap='binary', norm=norm, extent=extent, aspect='auto')
        #         ax.set_xlabel(x_label)
        #         ax.set_ylabel(y_label)
        #
        #         add_colorbar(sc, r'log |Es_{}| [{}]'.format(i, self.Es.unit))
        #
        #         # pos = ax.get_position()
        #         # cax = fig.add_axes([pos.x0, pos.y0 + pos.height + 0.01, pos.width, 0.02])
        #         #
        #         # cb1 = ColorbarBase(cax, cmap=plt.cm.binary,
        #         #                    norm=norm, orientation='horizontal')
        #         #
        #         # cb1.set_label(r'|Es_{}| [{}]'.format(i, img.unit))
        #         #
        #         # cax.xaxis.set_ticks_position('top')
        #         # cax.xaxis.labelpad = -35
        #
        #         plt.show()


def test_born_series():
    nu = 20e6 / au.s

    # print(kn(0,1 + 1j))
    dx = 7.5 * 0.25

    box = [Quantity([-dx * 200, dx * 200] * au.m), Quantity([-dx * 100, dx * 100] * au.m),
           Quantity([0., dx * 4000] * au.m)]
    shape = [201, 51, 301]
    space = CartesianSpaceDomain(box, shape)
    space.build()
    # fed_model = CosExp(mean=3e10/ au.corner_indices ** 3, amp=3e12 / au.corner_indices ** 3, period=50 * au.corner_indices, height=750 * au.corner_indices,
    #                    thickness=30 * au.corner_indices, space_domain=space)
    fed_model = ThinCircularHole(diameter=21 * au.m, thickness=40 * au.m, height=750 * au.m, layer=4e12 / au.m ** 3,
                                 space_domain=space)
    fed_model.build()
    B_model = ConstantParallelB(space_domain=space)
    B_model.build()
    ionosphere_model = IonosphereModel(nu, space, fed_model, B_model)
    ionosphere_model.build()
    n_model = DiffractionHole(diameter=21 * au.m, thickness=40 * au.m, height=750 * au.m, space_domain=space)
    n_model.build()
    potential = ScatteringPotential(nu, space, ionosphere_model, boundary_depth=5.)
    potential.build()
    potential.plot()
    fourier = FunctionFourier(space_domain=space)
    fourier.build()
    greens_function = ThreeDimGreensFunction(potential, fourier)
    greens_function.build()
    s_hat = Quantity(np.array([[0., 0., 1.]]))
    E0 = Quantity(np.array([[1., 0., 0.]]) * au.Jy ** (0.5))
    initial_field = InitialField(s_hat, nu, E0, space)
    initial_field.build()
    initial_field.plot()
    born_series = BornSeries(1, potential, greens_function, initial_field, fourier)
    born_series.build()
    born_series.plot()
    born_series.plot_mayavi()
