import numpy as np
from matplotlib import rc
from scipy.integrate import quad
import functools
import time


def timer(func):
    """Print the runtime of the decorated function.

    ref: https://realpython.com/primer-on-python-decorators/

    Parameters
    ----------
    func : function
           the function to use.
    Returns
    -------
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        # print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        print(f"=========== FINISHED in {run_time:.4f} secs. ==============")
        return value
    return wrapper_timer


def prettify_plot():
    """
    change the plot matplotlibrc file

    To use it, please run it before plotting.

    https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files
    """
    rc('text', usetex=True)
    rc('font', family='serif', serif='Computer Modern Roman', size=8)
    # rc('legend', fontsize=10)
    # rc('mathtext', fontset='cm')
    rc('xtick', direction='in')
    rc('ytick', direction='in')


def basis(x, i, N, w):
    """
    basis.
    x: coordinate
    i: index
    N: number of basis
    w: frequency
    """
    if i == 0:
        return np.sqrt(w/(2*np.pi))
    elif i <= (N-1)/2:
        return np.sqrt(w/np.pi)*np.cos(i*w*x)
    else:
        n = i - (N-1)/2
        return np.sqrt(w/np.pi)*np.sin(n*w*x)


def quad_break(func: callable, a, b, points, **kwargs):
    """
    wrap the function `scipy.integrate.quad`
    this makes the parameter 'points' avaliable when we use the paramter
    'weight'
    """
    points = np.sort(points)
    if len(points) == 0:
        return quad(func, a, b, **kwargs)

    if a >= points[0] or b <= points[-1]:
        raise ValueError(f'points must be inside [{a:.5f}, {b:.5f}]',
                         'but the points are:', points)

    res = [0, 0]
    ps = np.concatenate([[a], points, [b]])
    for i in range(len(ps) - 1):
        resi = np.array(quad(func, ps[i], ps[i+1], **kwargs))
        res[0] += resi[0]
        res[1] += resi[1]
    return res


def root_idx(a: np.ndarray):
    """return the index where a change the sign and a is continuous."""
    idx = (a[:-1] * a[1:] < 0) * (np.abs(a[1:]) < 1e-1)
    idx = np.argwhere(idx).T[0]
    return idx
