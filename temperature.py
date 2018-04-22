"""Utilities for effective temperature-related calculations with energy distributions."""
import numpy as np
from scipy.stats import linregress
from scipy.special import gamma, gammainc, expi


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
