"""Implements multinomial and mixed logit models."""

#pylint: disable=invalid-name
import numpy as np
from scipy.stats import t
from time import time
from abc import ABC
import warnings

""" Notations
N : Number of choice situations
P : Number of observations per panel
J : Number of alternatives
K : Number of variables (Kf: fixed, Kr: random)
"""

class ChoiceModel(ABC):
    """Base class for estimation of discrete choice models."""

    def __init__(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0
        self.verbose = 1
        self.robust = False

    def _reset_attributes(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0
        self.verbose = 1
        self.robust = False

    def _as_array(self, X, y, varnames, alts, isvars, ids, weights, panels,
                  avail, scale_factor, addit):
        X = np.asarray(X)
        y = np.asarray(y)
        varnames = np.asarray(varnames) if varnames is not None else None
        alts = np.asarray(alts) if alts is not None else None
        isvars = np.asarray(isvars) if isvars is not None else None
        ids = np.asarray(ids) if ids is not None else None
        weights = np.asarray(weights) if weights is not None else None
        panels = np.asarray(panels) if panels is not None else None
        avail = np.asarray(avail) if avail is not None else None
        scale_factor = np.asarray(scale_factor) if scale_factor is not None else None
        addit = np.asarray(addit) if addit is not None else None
        return X, y, varnames, alts, isvars, ids, weights, panels, avail, scale_factor, addit

    def _pre_fit(self, alts, varnames, isvars, base_alt,
                 fit_intercept, maxiter):
        self._reset_attributes()
        self._fit_start_time = time()
        self._isvars = [] if isvars is None else list(isvars)
        self._asvars = [v for v in varnames if v not in self._isvars]
        self._varnames = list(varnames)
        self._fit_intercept = fit_intercept
        self.alternatives = np.sort(np.unique(alts))
        self.base_alt = self.alternatives[0] if base_alt is None else base_alt
        self.maxiter = maxiter

    def _post_fit(self, optim_res, coeff_names, sample_size, verbose=1, robust=False):
        self.convergence = optim_res['success']
        self.coeff_ = optim_res['x']
        self.hess_inv = optim_res['hess_inv']
        self.grad_n = optim_res.get('grad_n', np.zeros((sample_size, len(self.coeff_))))
        self.covariance = self._robust_covariance(self.hess_inv, self.grad_n) \
            if robust else self.hess_inv
            
        self.covariance = (self.covariance + self.covariance.T) / 2
        
        diag_cov = np.diag(self.covariance)
        self.stderr = np.sqrt(np.where(diag_cov < 0, np.inf, diag_cov))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self.zvalues = self.coeff_/self.stderr
            self.pvalues = 2*t.cdf(-np.abs(self.zvalues), df=max(1, sample_size - len(self.coeff_)))
            
        self.loglikelihood = -optim_res['fun']
        self.estimation_message = optim_res['message']
        self.coeff_names = coeff_names
        self.total_iter = optim_res['nit']
        self.estim_time_sec = time() - self._fit_start_time
        self.sample_size = sample_size
        self.aic = 2*len(self.coeff_) - 2*self.loglikelihood
        self.bic = np.log(sample_size)*len(self.coeff_) - 2*self.loglikelihood
        self.total_fun_eval = optim_res.get('nfev', self.total_iter)



    def _robust_covariance(self, hess_inv, grad_n):
        if grad_n is None or hess_inv is None:
            return np.eye(len(self.coeff_)) * np.nan
        if grad_n.ndim == 1:
            grad_n = grad_n[:, np.newaxis]
        n = np.shape(grad_n)[0]
        if n == 0:
             return np.eye(len(self.coeff_)) * np.nan
        meat = np.transpose(grad_n) @ grad_n
        covariance = hess_inv @ meat @ hess_inv
        correction = n / (n - 1)
        return correction * covariance

    def _setup_design_matrix(self, X):
        J = len(self.alternatives)
        N = int(len(X)/J)
        isvars = self._isvars.copy()
        asvars = self._asvars.copy()
        varnames = self._varnames.copy()

        if self._fit_intercept:
            if '_intercept' not in isvars and '_intercept' not in asvars:
                isvars.insert(0, '_intercept')
                varnames.insert(0, '_intercept')
                X = np.hstack((np.ones(J*N)[:, None], X))

        ispos = [varnames.index(i) for i in isvars]
        aspos = [varnames.index(i) for i in asvars]

        X_is_part, X_as_part = None, None

        if isvars:
            dummy = np.tile(np.eye(J), reps=(N, 1))
            base_alt_idx = np.where(self.alternatives == self.base_alt)[0]
            if base_alt_idx.size > 0:
                dummy = np.delete(dummy, base_alt_idx[0], axis=1)
            
            Xis = X[:, ispos]
            X_is_part = np.einsum('ni,nj->nij', Xis.reshape(N*J, -1), dummy).reshape(N, J, -1)

        if asvars:
            X_as_part = X[:, aspos].reshape(N, J, -1)

        if X_is_part is not None and X_as_part is not None:
            X_final = np.dstack((X_is_part, X_as_part))
        elif X_is_part is not None:
            X_final = X_is_part
        elif X_as_part is not None:
            X_final = X_as_part
        else:
            return np.zeros((N, J, 0)), np.array([])
        
        is_names = ["{}.{}".format(isvar, j) for isvar in isvars
                    for j in self.alternatives if j != self.base_alt]
        names = is_names + asvars
        return X_final, np.array(names)

    def _check_long_format_consistency(self, ids, alts):
        uq_alts, idx = np.unique(alts, return_index=True)
        uq_alts = uq_alts[np.argsort(idx)]
        expected_alts = np.tile(uq_alts, int(len(ids)/len(uq_alts)))
        if not np.array_equal(alts, expected_alts):
            raise ValueError('长格式中的 alts 值不一致')
        _, obs_by_id = np.unique(ids, return_counts=True)
        if not np.all(obs_by_id % len(uq_alts) == 0):
            raise ValueError('长格式中的 alts 和 ids 值不一致')

    def _format_choice_var(self, y, alts):
        uq_alts = np.unique(alts)
        J, N = len(uq_alts), len(y)//len(uq_alts)
        y_flat = y.flatten()
        alts_flat = alts.flatten()

        # Case 1: y is already one-hot encoded
        if y.ndim > 1 and y.shape == (N, J) and np.all(y.sum(axis=1) == 1):
            return y
        if y.ndim == 1 and np.array_equal(y.reshape(N, J).sum(axis=1), np.ones(N)):
            return y.reshape(N, J)

        # Case 2: y contains the chosen alternative's value
        y1h = (alts_flat.reshape(N, J) == y_flat.reshape(N, 1)).astype(int)
        if np.array_equal(y1h.sum(axis=1), np.ones(N)):
            return y1h

        raise ValueError("不一致的 'y' 值。确保 y 是 one-hot 编码或包含每个样本选择的方案值。")

    def _validate_inputs(self, X, y, alts, varnames, isvars, ids, weights):
        if varnames is None:
            raise ValueError('需要 varnames 参数')
        if alts is None:
            raise ValueError('需要 alts 参数')
        if X.ndim != 2:
            raise ValueError("X 必须是长格式的二维数组")
        if y is not None and y.ndim != 1:
            raise ValueError("y 必须是长格式的一维数组")
        if len(varnames) != X.shape[1]:
            raise ValueError("varnames 的长度必须与 X 的列数匹配")

    def summary(self):
        if self.coeff_ is None:
            warnings.warn("当前模型尚未被估计", UserWarning)
            return
        if not self.convergence:
            pass
            #warnings.warn("警告：未达到收敛。估计可能不可靠。", UserWarning)
        if self.convergence:
            print("优化成功终止。")

        print("    消息: {}".format(self.estimation_message ))
        print("    迭代次数: {}".format(self.total_iter))
        print("    函数评估次数: {}".format(self.total_fun_eval))
        print("    估计时间= {:.1f} 秒".format(self.estim_time_sec))
        print("-"*75)
        print("{:19} {:>13} {:>13} {:>13} {:>13}"
              .format("系数", "估计值", "标准误.", "z-值", "P>|z|"))
        print("-"*75)
        fmt = "{:19} {:13.7f} {:13.7f} {:13.7f} {:13.3g} {:3}"
        for i in range(len(self.coeff_)):
            signif = ""
            if self.pvalues[i] < 0.001:
                signif = "***"
            elif self.pvalues[i] < 0.01:
                signif = "** "
            elif self.pvalues[i] < 0.05:
                signif = "*"
            elif self.pvalues[i] < 0.1:
                signif = "."
            print(fmt.format(self.coeff_names[i][:19], self.coeff_[i],
                             self.stderr[i], self.zvalues[i], self.pvalues[i],
                             signif))
        print("-"*75)
        print("显著性:  0 '***' 0.001 '** ' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("")
        print("对数似然= {:.3f}".format(self.loglikelihood))
        print("AIC= {:.3f}".format(self.aic))
        print("BIC= {:.3f}".format(self.bic))

def diff_nonchosen_chosen(X, y, scale, addit, avail):
    N, J, K = X.shape
    y_bool_flat = y.astype(bool).flatten()
    X_flat = X.reshape(N * J, K)

    X_chosen = X_flat[y_bool_flat, :].reshape(N, 1, K)
    X_nonchosen = X_flat[~y_bool_flat, :].reshape(N, J - 1, K)

    Xd = X_nonchosen - X_chosen

    scale_d, addit_d, avail_d = None, None, None
    if scale is not None:
        scale_flat = scale.flatten()
        scale_d = scale_flat[~y_bool_flat].reshape(N, J - 1) - scale_flat[y_bool_flat].reshape(N, 1)
    if addit is not None:
        addit_flat = addit.flatten()
        addit_d = addit_flat[~y_bool_flat].reshape(N, J - 1) - addit_flat[y_bool_flat].reshape(N, 1)
    if avail is not None:
        avail_d = avail.flatten()[~y_bool_flat].reshape(N, J - 1)
        
    return Xd, scale_d, addit_d, avail_d
