"""Implements all the logic for multinomial and conditional logit models."""

#pylint: disable=line-too-long,invalid-name
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from ._choice_model import ChoiceModel, diff_nonchosen_chosen
from ._optimize import _minimize_jax, _numerical_hessian_jax
from ._device import device

""" Notations
N : Number of choice situations
J : Number of alternatives
K : Number of variables
"""

_unpack_tuple = lambda x : x if len(x) > 1 else x[0]

# This is the core function that will be differentiated.
# It calculates the negative log-likelihood for a single observation.
def _single_obs_neg_loglik(params, xd, scale_d, addit_d, weight, avail):
    if scale_d is not None:
        lambdac = params[-1]
        betas = params[:-1]
    else:
        lambdac = 1.0
        betas = params

    Vd = jnp.einsum('jk,k->j', xd, betas)
    
    if scale_d is not None:
        Vd -= lambdac * scale_d
    if addit_d is not None:
        Vd += lambdac * addit_d
        
    eVd = jnp.exp(Vd)
    
    if avail is not None:
        eVd *= avail
    
    proba = 1 / (1 + eVd.sum())
    loglik = jnp.log(jnp.maximum(proba, 1e-200))
    
    if weight is not None:
        loglik *= weight
        
    return -loglik

@partial(jax.jit, static_argnames=("return_gradient_per_sample",))
def _mnl_loglik_calculator(params, Xd, scale_d, addit_d, weights, avail, return_gradient_per_sample=False):
    """Calculates total loss and optionally per-sample gradients."""
    # Calculate total loss by vmapping the single observation function
    total_loss = jnp.sum(jax.vmap(_single_obs_neg_loglik, in_axes=(None, 0, 0, 0, 0, 0))(
        params, Xd, scale_d, addit_d, weights, avail
    ))

    aux_data = None
    if return_gradient_per_sample:
        # Calculate per-sample gradients by vmapping jax.grad
        grad_fn = jax.grad(_single_obs_neg_loglik)
        grad_n = jax.vmap(grad_fn, in_axes=(None, 0, 0, 0, 0, 0))(
            params, Xd, scale_d, addit_d, weights, avail
        )
        aux_data = grad_n
        
    return total_loss, aux_data

class MultinomialLogit(ChoiceModel):
    """Class for estimation of Multinomial and Conditional Logit Models."""
    
    def fit(self, X, y, varnames, alts, ids, isvars=None,
            weights=None, avail=None, base_alt=None, fit_intercept=False,
            init_coeff=None, maxiter=2000, random_state=None, tol_opts=None,
            verbose=1, robust=False, num_hess=True, scale_factor=None,
            addit=None, skip_std_errs=False):
        """Fit multinomial and/or conditional logit models."""
        X, y, varnames, alts, isvars, ids, weights, _, avail, scale_factor, addit \
            = self._as_array(X, y, varnames, alts, isvars, ids, weights, None, avail, scale_factor, addit)
        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        if isvars is None:
            isvars = list(varnames)
            fit_intercept = True

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)
        
        betas, X_design, y_design, weights_design, avail_design, Xnames, scale_design, addit_design = \
            self._setup_input_data(X, y, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=init_coeff,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=False, scale_factor=scale_factor, addit=addit)

        Xd, scale_d, addit_d, avail_d = diff_nonchosen_chosen(X_design, y_design, scale_design, addit_design, avail_design)
        
        betas, Xd, scale_d, addit_d, weights_design, avail_d = map(
            device.to_device, [betas, Xd, scale_d, addit_d, weights_design, avail_d]
        )
        
        loss_and_grad_fn = jax.value_and_grad(
            lambda p, *a: _mnl_loglik_calculator(p, *a, return_gradient_per_sample=True), has_aux=True
        )

        fargs = (Xd, scale_d, addit_d, weights_design, avail_d)

        gtol = tol_opts.get('gtol', 1e-6) if tol_opts else 1e-6
        optim_res = _minimize_jax(loss_and_grad_fn, betas, args=fargs,
                                  maxiter=maxiter, gtol=gtol, disp=verbose > 1)
        
        coef_names = Xnames
        if scale_factor is not None:
            coef_names = np.append(coef_names, "_scale_factor")

        optim_res['x'] = device.to_cpu(optim_res['x'])
        optim_res['grad_n'] = device.to_cpu(optim_res['grad_n'])
        
        if skip_std_errs:
            optim_res['hess_inv'] = np.eye(len(optim_res['x']))
        else:
            loss_fn_for_hess = lambda p, *a: _mnl_loglik_calculator(p, *a, return_gradient_per_sample=False)[0]
            hess_inv = _numerical_hessian_jax(device.to_device(optim_res['x']), loss_fn_for_hess, args=fargs)
            optim_res['hess_inv'] = device.to_cpu(hess_inv)

        self._post_fit(optim_res, coef_names, X_design.shape[0], verbose, robust)

    def predict(self, X, varnames, alts, ids, isvars=None, weights=None,
                avail=None, random_state=None, verbose=1,
                return_proba=False, return_freq=False, scale_factor=None, addit=None):
        X, _, varnames, alts, isvars, ids, weights, _, avail, scale_factor, addit \
            = self._as_array(X, None, varnames, alts, isvars, ids, weights, None, avail, scale_factor, addit)

        self._validate_inputs(X, None, alts, varnames, isvars, ids, weights)
        
        if isvars is None:
            isvars = list(varnames)
   
        betas, X_design, _, _, avail_design, Xnames, scale_design, addit_design = \
            self._setup_input_data(X, None, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=self.coeff_,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=True, scale_factor=scale_factor, addit=addit)
            
        coeff_names = Xnames
        coeff_names = coeff_names if scale_factor is None else np.append(coeff_names, "_scale_factor")
        if not np.array_equal(coeff_names, self.coeff_names):
            raise ValueError("提供的 'varnames' 产生的系数名称与 'self.coeff_names' 中存储的不一致")
        
        betas, X_design, scale_design, addit_design, avail_design = map(
            device.to_device, [betas, X_design, scale_design, addit_design, avail_design]
        )

        if scale_factor is not None:
            lambdac = betas[-1]
            betas_model = betas[:-1]
        else:
            lambdac = 1.0
            betas_model = betas
            
        V = jnp.einsum('njk,k->nj', X_design, betas_model)
        
        if scale_design is not None:
             V -= scale_design
        if addit_design is not None:
             V += addit_design
        
        V = lambdac * V

        eV = jnp.exp(V)
        if avail_design is not None:
            eV *= avail_design

        sum_eV = jnp.sum(eV, axis=1, keepdims=True)
        proba = eV / jnp.where(sum_eV == 0, 1, sum_eV)
        
        proba = device.to_cpu(proba)
        
        idx_max_proba = np.argmax(proba, axis=1)
        choices = self.alternatives[idx_max_proba]
        
        output = (choices, )
        if return_proba:
            output += (proba, )
        
        if return_freq:
            alt_list, counts = np.unique(choices, return_counts=True)
            freq = dict(zip(list(alt_list),
                            list(np.round(counts/np.sum(counts), 3))))
            output += (freq, )
        
        return _unpack_tuple(output)

    def _setup_input_data(self, X, y, varnames, alts, ids, isvars=None,
            weights=None, avail=None, base_alt=None, fit_intercept=False,
            init_coeff=None, random_state=None, verbose=1, predict_mode=False,
            scale_factor=None, addit=None):
        self._check_long_format_consistency(ids, alts)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        
        if not hasattr(self, '_isvars'):
             self._isvars = [] if isvars is None else list(isvars)
             self._asvars = [v for v in varnames if v not in self._isvars]

        X, Xnames = self._setup_design_matrix(X)
        N, J, K = X.shape
        
        if random_state is not None:
            np.random.seed(random_state)
               
        if weights is not None:
            weights = weights.reshape(N, J)[:, 0]

        if avail is not None:
            avail = avail.reshape(N, J)

        if init_coeff is None:
            betas = np.repeat(.0, K)
            if scale_factor is not None:
                betas = np.append(betas, 1.)
        else:
            betas = init_coeff
            n_coeff = K + (1 if scale_factor is not None else 0)
            if len(init_coeff) != n_coeff:
                raise ValueError(f"初始系数的大小必须是: {n_coeff}, 但得到的是 {len(init_coeff)}")
                
        scale = None if scale_factor is None else scale_factor.reshape(N, J)
        addit = None if addit is None else addit.reshape(N, J)
    
        return betas, X, y, weights, avail, Xnames, scale, addit

    def summary(self):
        super(MultinomialLogit, self).summary()
