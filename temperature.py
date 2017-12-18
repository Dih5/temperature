import numpy as np
from scipy.stats import linregress
from scipy.special import gamma, gammainc, expi


def filter_points(xx, yy, x_min, x_max):
    """Select points where between x_min <= x <= x_max"""
    temp = list(map(list, zip(*filter(lambda t: x_min <= t[0] <= x_max, zip(xx, yy)))))
    return temp if temp else ([], [])


def eff_temperature(x, y, x_range=None, method="fit"):
    """
    Find an effective temperature of data points, possibly in a given range.

    Args:
        x: x-coordinates of the points.
        y: y-coordinates of the points.
        x_range: A 2-tuple defining an interval in x to filter the points.
        method: The method used to calculate the temperature. Available methods include:
            *"fit": Best linear fit

    Returns:

    """
    if x_range:
        x2, y2 = filter_points(x, y, x_range[0], x_range[1])
    else:
        x2, y2 = x, y
    if method == "fit":
        return eff_temperature_fit(x2, y2)
    else:
        raise ValueError("Unknown method: %s" % method)


def eff_temperature_list(xx, yy, x_window, method="fit"):
    """
    Find points describing the effective temperature function using a certain window.

    Args:
        x: x-coordinates of the points.
        y: y-coordinates of the points.
        x_window (float or Callable): Half-size of the window used in each fit. It can be a Callable taking the window
                                      mean point and returning the halfsize.
        method: The method used to calculate the temperature. Available methods include:
            *"fit": Best linear fit

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
                    tt.append(eff_temperature_fit(*filter_points(xx, yy, x - x_window, x + x_window)))
        else:  # Assumed callable
            for x in xx:
                window = x_window(x)
                if x - window >= x_min and x + window <= x_max:
                    xx2.append(x)
                    tt.append(eff_temperature_fit(*filter_points(xx, yy, x - window, x + window)))
        return xx2, tt

    else:
        raise ValueError("Unknown method: %s" % method)


def linear_function(x0, y0, x1, y1):
    """returns a linear function given two points"""
    a = (y0 - y1) / (x0 - x1)
    b = y0 - a * x0
    return lambda x: a * x + b


def eff_temperature_fit(x, y):
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


def finite_temperature_sym_dif_list(x, y, i=2):
    """ Get a list of temperatures obtained by symmetric differences. i must be even"""
    log_y = np.log(np.array(y))
    x = np.array(x)
    return np.array(x[i // 2:-i // 2]), -(x[i:] - x[:-i]) / (log_y[i:] - log_y[:-i])


def mb_temp(theta, e, alpha=1.5):
    """Effective temperature in a Maxwellian (Gamma) distribution"""
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


def mb_bremss(theta, e_g, alpha=1.5, b=1.0):
    """Maxwellian-electron produced bremsstrahlung distribution, according to a linear model of the scaled
    bremsstrahlung cross-section"""
    scaled = e_g / theta
    return (theta * _incomplete_gamma(alpha, scaled) - b * e_g * _incomplete_gamma(alpha - 1, scaled)) / \
           (e_g * theta * gamma(alpha))


def mb_bremss_temp(theta, e_g, alpha=1.5, b=1.0):
    """Maxwellian-electron produced bremsstrahlung effective temperature"""
    # TODO: when e_g >> theta (~25 times or so) numerical convergence fails.
    # This is due to the quotient used. Perhaps previously simplified forms can be used when alpha is integer or
    # semi-integer, at least for the b=1 case.
    # This also affects the Kramer's model below.
    # Nevertheless, I doubt such a high limit will ever be in consideration
    scaled = e_g / theta
    return (np.exp(scaled) * scaled * (
        theta * _incomplete_gamma(alpha, scaled) - b * e_g * _incomplete_gamma(-1 + alpha, scaled))) / (
               (1 - b) * scaled ** alpha + np.exp(scaled) * _incomplete_gamma(alpha, scaled))


def mb_bremss_asympt_temp(theta, e_g, alpha=1.5, b=1.0):
    """Maxwellian-electron produced bremsstrahlung effective temperature in the asymptotic approximation (1st order).

    NOTE: The asymptotic expansion is only useful in the asymptotic limit. That is far beyond ten times kT.
    """
    scaled = e_g / theta
    # The asymptotic expansion differs when b==1
    if b == 1:
        return theta * (1 + (alpha - 3) / scaled)
    else:
        return theta * (1 + (alpha - 2) / scaled)


def bimb_bremss_temp(theta1, theta2, a1, e_g, alpha=1.5, b=1.0):
    """Bimaxwellian-electron produced bremsstrahlung distribution according to a linear model of the scaled
    bremsstrahlung cross-section"""
    a2 = 1 - a1
    f1 = a1 * mb_bremss(theta1, e_g, alpha=alpha, b=b)
    f2 = a2 * mb_bremss(theta2, e_g, alpha=alpha, b=b)
    f = f1 + f2
    return 1 / (f1 / f / mb_bremss_temp(theta1, e_g, alpha=alpha, b=b) +
                f2 / f / mb_bremss_temp(theta2, e_g, alpha=alpha, b=b))


def mb_kramers(theta, e_g, alpha=1.5):
    """Maxwellian-electron produced bremsstrahlung distribution, according to the Kramer's thick target model"""
    scaled = e_g / theta
    return (theta * _incomplete_gamma(alpha + 1, scaled) - e_g * _incomplete_gamma(alpha, scaled)) / (
    e_g * gamma(alpha))


def mb_kramers_temp(theta, e_g, alpha=1.5):
    """Maxwellian-electron produced bremsstrahlung effective temperature, according to the Kramer's thick target
    model"""
    scaled = e_g / theta
    return e_g - (e_g ** 2 * _incomplete_gamma(alpha, scaled)) / (theta * _incomplete_gamma(1 + alpha, scaled))
