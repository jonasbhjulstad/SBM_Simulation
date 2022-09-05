import warnings

import numpy as np
from sklearn.linear_model import ridge_regression
from pysindy.optimizers.base import BaseOptimizer


class FROLS(BaseOptimizer):
    """Forward Regression Orthogonal Least-Squares (FROLS) optimizer.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteractively selecting the most correlated
    function in the library. This is a greedy algorithm.

    See the following reference for more details:

        Billings, Stephen A. Nonlinear system identification:
        NARMAX methods in the time, frequency, and spatio-temporal domains.
        John Wiley & Sons, 2013.

    Parameters
    ----------
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    kappa : float, optional (default None)
        If passed, compute the MSE errors with an extra L0 term with
        strength equal to kappa times the condition number of Theta.

    max_iter : int, optional (default 10)
        Maximum iterations of the optimization algorithm. This determines
        the number of nonzero terms chosen by the FROLS algorithm.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    ridge_kw : dict, optional (default None)
        Optional keyword arguments to pass to the ridge regression.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every
        iteration.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of FROLS.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import FROLS
    >>> lorenz = lambda z,t : [10 * (z[1] - z[0]),
    >>>                        z[0] * (28 - z[2]) - z[1],
    >>>                        z[0] * z[1] - 8 / 3 * z[2]]
    >>> t = np.arange(0, 2, .002)
    >>> x = odeint(lorenz, [-8, 8, 27], t)
    >>> opt = FROLS(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1] - t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        kappa=None,
        max_iter=10,
        alpha=0.05,
        ridge_kw=None,
        verbose=False,
    ):
        super(FROLS, self).__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
        )
        self.alpha = alpha
        self.ridge_kw = ridge_kw
        self.kappa = kappa
        if self.max_iter <= 0:
            raise ValueError("Max iteration must be > 0")
        self.verbose = verbose

    def _normed_cov(self, a, b):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                return np.vdot(a, b) / np.vdot(a, a)
            except RuntimeWarning:
                raise ValueError(
                    "Trying to orthogonalize linearly dependent columns created NaNs"
                )

    def _select_function(self, x, y, sigma, skip=[]):
        n_features = x.shape[1]
        g = np.zeros(n_features)  # Coefficients to orthogonalized functions
        ERR = np.zeros(n_features)  # Error Reduction Ratio at this step
        for m in range(n_features):
            if m not in skip:
                g[m] = self._normed_cov(x[:, m], y)
                ERR[m] = (
                    abs(g[m]) ** 2 * np.real(np.vdot(x[:, m], x[:, m])) / sigma
                )  # Error reduction

        best_idx = np.argmax(
            ERR
        )  # Select best function by maximum Error Reduction Ratio

        # Return index of best function, along with ERR and coefficient
        return best_idx, ERR[best_idx], g[best_idx]

    def _orthogonalize(self, vec, Q):
        """
        Orthogonalize vec with respect to columns of Q
        """
        Qs = vec.copy()
        s = Q.shape[1]
        for r in range(s):
            Qs -= self._normed_cov(Q[:, r], Qs) * Q[:, r]
        return Qs

    def _reduce(self, x, y):
        """
        Performs at most n_feature iterations of the
        greedy Forward Regression Orthogonal Least Squares (FROLS) algorithm
        """
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        for i in range(n_targets):\
        # For each output, choose the minimum L0-penalized loss function
            self.coef_ = 
