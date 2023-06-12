import numpy as np
def runge_kutta_fourth_order(f, y0, t0, tf, h):
    """
    Solves an initial value problem of the form y' = f(t, y),
    where y(t0) = y0, using the fourth-order Runge-Kutta method.
    Parameters
    ----------
    f : function
        A function that takes two arguments, t and y, and returns
        the derivative of y at time t.
    y0 : float or array_like
        The initial value of y at time t0.
    t0 : float
        The initial time.
    tf : float
        The final time.
    h : float
        The time step.
    Returns
    -------
    ndarray
        An array of shape (n+1, m), where n = (tf - t0) / h and m
        is the size of y0. Each row represents the value of y
        at a time ti given by ti = t0 + i*h.
    Examples
    --------
    >>> f = lambda t, y: -y
    >>> y0 = 1
    >>> t0, tf = 0, 2
    >>> h = 0.1
    >>> y = runge_kutta_fourth_order(f, y0, t0, tf, h)
    >>> len(y)
    21
    >>> y[-1]
    0.1353352832366127
    """
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n+1)
    m = np.size(y0)
    y = np.zeros((n+1, m))
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2)*k1)
        k3 = f(t[i] + h/2, y[i] + (h/2)*k2)
        k4 = f(t[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

f = lambda t, y: -y
y0 = 1  # initial value of y
t0, tf = 0, 2  # initial and final time
h = 0.1  # time step
y = runge_kutta_fourth_order(f, y0, t0, tf, h)
print(y)