def SPSA(func, x0, bounds, niter=100, a=1.0, alpha=0.602, c=1.0, gamma=0.101):
    """
    Minimization of an objective function by a simultaneous perturbation
    stochastic approximation.
    This algorithm approximates the gradient of the function by finite differences
    along stochastic directions Deltak. 

    Parameters
    ----------
    func: function 
        function to be minimized:
    x0: array-like
        initial guess for parameters 
    bounds: array-like
        bounds on the variables
    niter: int
        number of iterations after which to terminate the algorithm
    a: float
       scaling parameter for step size
    alpha: float
        scaling exponent for step size
    c: float
       scaling parameter for evaluation step size
    gamma: float
        scaling exponent for evaluation step size 
    Returns
    -------
    Optimized parameters to minimize input function over bounds
    """
    A = 0.01 * niter

    bounds = np.asarray(bounds)
    project = lambda x: np.clip(x, bounds[:, 0], bounds[:, 1])

    N = len(x0)
    x = x0
    for k in range(niter):
        ak = a/(k+1.0+A)**alpha
        ck = c/(k+1.0)**gamma
        Deltak = np.random.choice([-1, 1], size=N)
        fkwargs = dict()
        # check points are in boundaries
        xplus = project(x + ck*Deltak)
        xminus = project(x - ck*Deltak)
        grad = (func(xplus, **fkwargs) - func(xminus, **fkwargs)) / (xplus-xminus)
        x = project(x - ak*grad)
    return x
