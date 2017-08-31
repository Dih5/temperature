import numpy as np
from scipy.stats import linregress


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
    Find points describing the effective temperature funtion using a certain window.

    Args:
        x: x-coordinates of the points.
        y: y-coordinates of the points.
        x_window: Half-size of the window used in each fit
        method: The method used to calculate the temperature. Available methods include:
            *"fit": Best linear fit

    Returns:

    """
    x_min = min(xx)
    x_max = max(xx)
    if method == "fit":
        xx2=[]
        tt=[]
        for x in xx:
            if x-x_window >= x_min and x+x_window<= x_max:
                xx2.append(x)
                tt.append(eff_temperature_fit(*filter_points(xx,yy,x-x_window,x+x_window)))
        return xx2, tt

    else:
        raise ValueError("Unknown method: %s" % method)

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
