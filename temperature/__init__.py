"""Utilities for effective temperature-related calculations with energy distributions."""
from os import path

import numpy as np
from scipy.stats import linregress, gamma as gamma_dist
from scipy.special import gamma, gammainc, expi
import csv
from scipy.interpolate import dfitpack, RectBivariateSpline
from scipy.integrate import quad
from functools import lru_cache

import pickle
from xpecgen.xpecgen import Spectrum  # Needed to unpickle the spectra

data_path = path.join(path.dirname(path.abspath(__file__)), "data")


def filter_points(xx, yy, x_min, x_max):
    """
    Select points where x_min <= x <= x_max.

    Args:
        xx (list of float): X-coordinates.
        yy (list of float): Y coordinates.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate accepted.

    Returns:
        (list of float, list of float): The lists of coordinates of the points selected.

    """
    temp = list(map(list, zip(*filter(lambda t: x_min <= t[0] <= x_max, zip(xx, yy)))))
    return temp if temp else ([], [])


def eff_temperature(xx, yy, x_range=None, method="fit"):
    """
    Find an effective temperature of data points, possibly in a given range.

    Args:
        xx: x-coordinates of the points.
        yy: y-coordinates of the points.
        x_range: A 2-tuple defining an interval in x to filter the points.
        method: The method used to calculate the temperature. Available methods include:

            - "fit": Best linear fit.

    Returns:
        float: The value of the effective temperature.

    """
    if x_range:
        xx2, yy2 = filter_points(xx, yy, x_range[0], x_range[1])
    else:
        xx2, yy2 = xx, yy
    if method == "fit":
        return _eff_temperature_fit(xx2, yy2)
    else:
        raise ValueError("Unknown method: %s" % method)


def eff_temperature_list(xx, yy, x_window, method="fit"):
    """
    Find points describing the effective temperature function using a certain window.

    Args:
        xx: x-coordinates of the points.
        yy: y-coordinates of the points.
        x_window (float or Callable): Half-size of the window used in each fit. It can be a Callable taking the window
                                      mean point and returning the half-size.
        method: The method used to calculate the temperature. Available methods include:

            - "fit": Best linear fit.

    Returns:

    """
    x_min = min(xx)
    x_max = max(xx)
    if method == "fit":
        xx2 = []
        tt = []
        if isinstance(x_window, float) or isinstance(x_window, int):
            for x in xx:
                if x - x_window >= x_min and x + x_window <= x_max:
                    xx2.append(x)
                    tt.append(_eff_temperature_fit(*filter_points(xx, yy, x - x_window, x + x_window)))
        else:  # Assumed callable
            for x in xx:
                window = x_window(x)
                if x - window >= x_min and x + window <= x_max:
                    xx2.append(x)
                    tt.append(_eff_temperature_fit(*filter_points(xx, yy, x - window, x + window)))
        return xx2, tt

    else:
        raise ValueError("Unknown method: %s" % method)


def _linear_function(x0, y0, x1, y1):
    """returns a linear function given two points"""
    a = (y0 - y1) / (x0 - x1)
    b = y0 - a * x0
    return lambda x: a * x + b


def _eff_temperature_fit(x, y):
    """Numerically find an effective temperature in the whole range given"""
    if not len(x) or len(x) != len(y):  # No points given or bad shape
        return np.nan
    # Discard points with no counts
    temp = list(map(list, zip(*filter(lambda t: t[1] > 0, zip(x, y)))))
    if not temp:
        return np.nan
    x2, y2 = temp
    slope, intercept, r_value, p_value, std_err = linregress(x2, np.log(y2))
    return -1 / slope


def _finite_temperature_sym_dif_list(x, y, i=2):
    """ Get a list of temperatures obtained by symmetric differences. i must be even"""
    # TODO: Add public interface (?)
    if i % 2:
        raise ValueError("i must be even")
    log_y = np.log(np.array(y))
    x = np.array(x)
    return np.array(x[i // 2:-i // 2]), -(x[i:] - x[:-i]) / (log_y[i:] - log_y[:-i])


def theta_mb(theta, e, alpha=1.5):
    """
    Effective temperature in a Maxwell-Boltzmann (Gamma) distribution

    Args:
        theta(float): Temperature of the MB.
        e(float): Energy where the effective temperature is measured.
        alpha: Shape parameter of the distribution (half of the degrees of freedom).

    Returns:
        float: The effective temperature.

    """
    scaled = e / theta
    return (scaled / (scaled + (alpha - 1))) * theta


def _incomplete_gamma(a, x):
    """Incomplete gamma function (not normalized, integral from x to infinity). Only implemented for real x>0"""
    if x < 0:
        raise NotImplementedError("This incomplete gamma is not implemented for x<=0")
    if a:  # not 0
        return gamma(a) * (1 - gammainc(a, x))
    else:
        # Gamma(0) diverges and indeterminate form appears, but can be reworked from definition
        return -expi(-x)  # Assuming x>0


def f_mb_findlay(theta, e, alpha=1.5, b=1.0):
    """
    MB-electron produced bremsstrahlung distribution, according to a linear model of the scaled
    bremsstrahlung scaled cross-section (Findlay's).

    Args:
        theta(float): Temperature of the MB.
        e(float): Energy where the effective temperature is measured.
        alpha (float): Shape parameter of the distribution (half of the degrees of freedom).
        b (float): Parameter of the cross-section model. Should be in [0, 1].

    Returns:
        float: The value of the probability density function in e.

    """
    scaled = e / theta
    return (theta * _incomplete_gamma(alpha, scaled) - b * e * _incomplete_gamma(alpha - 1, scaled)) / \
           (e * theta * gamma(alpha))


def theta_mb_findlay(theta, e, alpha=1.5, b=1.0):
    """
        MB-electron produced bremsstrahlung effective temperature, according to a linear model of the scaled
        bremsstrahlung scaled cross-section (Findlay's).

        Args:
            theta(float): Temperature of the MB.
            e(float): Energy where the effective temperature is measured.
            alpha (float): Shape parameter of the distribution (half of the degrees of freedom).
            b (float): Parameter of the cross-section model. Should be in [0, 1].

        Returns:
            float: The effective temperature.

    """
    # TODO: when e_g >> theta (~25 times or so) numerical convergence fails.
    # This is due to the quotient used. Perhaps previously simplified forms can be used when alpha is integer or
    # semi-integer, at least for the b=1 case.
    # This also affects the Kramer's model below.
    # Nevertheless, I doubt such a high limit will ever be in consideration
    scaled = e / theta
    return (np.exp(scaled) * scaled * (
            theta * _incomplete_gamma(alpha, scaled) - b * e * _incomplete_gamma(-1 + alpha, scaled))) / (
                   (1 - b) * scaled ** alpha + np.exp(scaled) * _incomplete_gamma(alpha, scaled))


def theta_mb_findlay_asympt(theta, e, alpha=1.5, b=1.0):
    """
        MB-electron produced bremsstrahlung effective temperature , according to a linear model of the scaled
        bremsstrahlung scaled cross-section (Findlay's) in the the asymptotic approximation (1st order).

        Args:
            theta(float): Temperature of the MB.
            e(float): Energy where the effective temperature is measured.
            alpha (float): Shape parameter of the distribution (half of the degrees of freedom).
            b (float): Parameter of the cross-section model. Should be in [0, 1].

        Returns:
            float: The effective temperature.

        Note:
            The asymptotic expansion is only useful in the asymptotic limit. That is far beyond ten times kT.

    """
    scaled = e / theta
    # The asymptotic expansion differs when b==1
    if b == 1:
        return theta * (1 + (alpha - 3) / scaled)
    else:
        return theta * (1 + (alpha - 2) / scaled)


def theta_bimb_findlay(theta1, theta2, a1, e, alpha=1.5, b=1.0):
    """
        Bi-MB-electron produced bremsstrahlung effective temperature, according to a linear model of the scaled
        bremsstrahlung scaled cross-section (Findlay's).

        Args:
            theta1(float): Temperature of the first component of the MB.
            theta2(float): Temperature of the second component of the MB.
            a1(float): Weight of the first component in the convex combination.
            e(float): Energy where the effective temperature is measured.
            alpha (float): Shape parameter of the distributions (half of the degrees of freedom).
            b (float): Parameter of the cross-section model. Should be in [0, 1].

        Returns:
            float: The effective temperature in e.

    """
    a2 = 1 - a1
    f1 = a1 * f_mb_findlay(theta1, e, alpha=alpha, b=b)
    f2 = a2 * f_mb_findlay(theta2, e, alpha=alpha, b=b)
    f = f1 + f2
    return 1 / (f1 / f / theta_mb_findlay(theta1, e, alpha=alpha, b=b) +
                f2 / f / theta_mb_findlay(theta2, e, alpha=alpha, b=b))


def f_mb_kramers(theta, e, alpha=1.5):
    """
    MB-electron produced bremsstrahlung distribution, according to to the Kramers' thick target model.

    Args:
        theta(float): Temperature of the MB.
        e(float): Energy where the effective temperature is measured.
        alpha (float): Shape parameter of the distribution (half of the degrees of freedom).

    Returns:
        float: The value of the probability density function in e.

    """
    scaled = e / theta
    return (theta * _incomplete_gamma(alpha + 1, scaled) - e * _incomplete_gamma(alpha, scaled)) / (
            e * gamma(alpha))


def theta_mb_kramers(theta, e, alpha=1.5):
    """
        MB-electron produced bremsstrahlung effective temperature, according to to the Kramers' thick target model.

        Args:
            theta(float): Temperature of the MB.
            e(float): Energy where the effective temperature is measured.
            alpha (float): Shape parameter of the distribution (half of the degrees of freedom).

        Returns:
            float: The value of the effective temperature in e.

    """
    scaled = e / theta
    return e - (e ** 2 * _incomplete_gamma(alpha, scaled)) / (theta * _incomplete_gamma(1 + alpha, scaled))


def _singlify(rect_bivariate_spline):
    """Alter a RectBivariateSpline so its call skips matrix transformation and derivative checks.
    Single evaluation is about 100% faster
    Calling in quad seems only about 5% faster"""
    tx, ty, c = rect_bivariate_spline.tck[:3]
    kx, ky = rect_bivariate_spline.degrees

    def call(self, x, y):
        """
        Altered to evaluate at a single position, no derivatives.
        """
        z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
        if not ier == 0:
            raise ValueError("Error code returned by bispev: %s" % ier)
        return z[0, 0]

    meths = {'__call__': call}
    rect_bivariate_spline.__class__ = type('PatchedRectBivariateSpline', (RectBivariateSpline,), meths)


@lru_cache()
def get_cs(Z=74):
    """
    Returns a function representing the bremsstrahlung cross_section.

    Returns:
        A function representing cross_section(e_g,e_e) in mb/keV, with e_g and e_e in keV.
    """
    # NOTE: Data is given for E_e>1keV. CS values below this level should be used with caution.
    # The default behaviour is to keep it constant
    with open(path.join(data_path, "cs", "grid.csv"), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ')
        t = next(r)
        e_e = np.array([float(a) for a in t[0].split(",")])
        log_e_e = np.log10(e_e)
        t = next(r)
        k = np.array([float(a) for a in t[0].split(",")])
    t = []
    with open(path.join(data_path, "cs", "%d.csv" % Z), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ')
        for row in r:
            t.append([float(a) for a in row[0].split(",")])
    t = np.array(t)
    scaled = RectBivariateSpline(log_e_e, k, t, kx=3, ky=1)
    _singlify(scaled)
    electron_mass = 511
    z_2 = Z * Z
    return lambda e_g, e_e: (e_e + electron_mass) ** 2 * z_2 / (e_e * e_g * (e_e + 2 * electron_mass)) * (
        scaled(np.log10(e_e), e_g / e_e))


def f_mb_nist(theta, e, alpha=1.5, Z=74, epsrel=1E-3):
    """
    MB-electron produced bremsstrahlung distribution, according to the NIST tabulated cross-section.

    Args:
        theta(float): Temperature of the MB.
        e(float): Energy where the effective temperature is measured.
        alpha (float): Shape parameter of the distribution (half of the degrees of freedom).
        Z (int): Atomic number of the medium.
        epsrel (float): Relative tolerance of the integral.

    Returns:
        float: The value of the probability density function in e.
    """
    return quad(lambda x: gamma_dist.pdf(x, alpha, scale=theta) * get_cs(Z)(e, x), e, np.inf, epsrel=epsrel)[0]


@lru_cache()
def _get_simulated_xpecgen(Z=74):
    electron_dist_x = np.arange(10, 600.1, 1)
    mesh_points = np.concatenate(([0], (electron_dist_x[1:] + electron_dist_x[:-1]) / 2, [np.inf]))

    if not path.isdir(path.join(data_path, "xpecgen", "%d" % Z)):
        raise NotImplementedError("No precomputed calculations included and automatic calculation is not implemented")
    s_list = []
    for i, e_0 in enumerate(electron_dist_x, start=1):
        with open(path.join(data_path, "xpecgen", "%d" % Z, "S%d.pkl" % i), 'rb') as f:
            s_list.append(pickle.load(f))

    return mesh_points, s_list


def fmesh_mb_xpecgen(theta, alpha=1.5, Z=74):
    """
    MB-electron produced bremsstrahlung distribution, according to the xpecgen model.

    Args:
        theta(float): Temperature of the MB.
        alpha (float): Shape parameter of the distribution (half of the degrees of freedom).
        Z (int): Atomic number of the medium.

    Returns:
        (list of float, list of float): x coordinates and y coordinates of points defining the density function.
    """
    mesh_points, s_list = _get_simulated_xpecgen(Z)
    electron_dist_y = [gamma_dist.cdf(e_1, alpha, scale=theta) - gamma_dist.cdf(e_0, alpha, scale=theta) for e_1, e_0 in
                       zip(mesh_points[1:], mesh_points[:-1])]
    s = sum([w * sp for sp, w in zip(s_list, electron_dist_y)])
    # If we wanted to remove first and last:
    # s = sum([w * sp for sp, w in zip(s_list[1:-1], electron_dist_y[1:-1])])
    s.discrete = []
    s.set_norm(1)
    return s.x, s.y
