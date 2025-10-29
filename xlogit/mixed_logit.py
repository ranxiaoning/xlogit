"""Implements all the logic for mixed logit models."""

#pylint: disable=invalid-name
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import scipy.stats

from ._choice_model import ChoiceModel, diff_nonchosen_chosen
from ._device import device
from .multinomial_logit import MultinomialLogit
from ._optimize import _minimize_jax, _numerical_hessian_jax

""" Notations
N : Number of choice situations
P : Number of panels
J : Number of alternatives
K : Number of variables (Kf: fixed, Kr: random)
"""
_unpack_tuple = lambda x : x if len(x) > 1 else x[0]

# --- JAX Pure Functions ---

def _apply_distribution_jax(betas_random, rvdist):
    for k, dist in enumerate(rvdist):
        if dist == 'ln':
            betas_random = betas_random.at[:, k, :].set(jnp.exp(betas_random[:, k, :]))
        elif dist == 'tn':
            betas_random = betas_random.at[:, k, :].set(jnp.where(betas_random[:, k, :] > 0, betas_random[:, k, :], 0))
    return betas_random

def _transform_rand_betas_jax(betas_mean, betas_sd_unconstrained, draws, rvidx, rvdist):
    br_mean = betas_mean[rvidx]
    br_sd = jax.nn.softplus(betas_sd_unconstrained)
    betas_random = br_mean[None, :, None] + draws * br_sd[None, :, None]
    return _apply_distribution_jax(betas_random, rvdist)

@partial(jax.jit, static_argnames=("rvidx_static", "rvdist_static", "num_panels", "num_mean_params"))
def _mxl_loglik_calculator(betas, Xd, draws, panel_ids, weights, scale_d, addit_d, avail, 
                           rvidx_static, rvdist_static, num_panels, num_mean_params):
    rvidx = np.array(rvidx_static)
    rvdist = rvdist_static
    
    if scale_d is not None:
        lambdac = betas[-1]
        betas_full = betas[:-1]
    else:
        lambdac = 1.0
        betas_full = betas

    betas_mean = betas_full[:num_mean_params]
    betas_sd_unconstrained = betas_full[num_mean_params:]

    Xdf = Xd[:, :, ~rvidx]
    Bf = betas_mean[~rvidx]
    Vdf = jnp.einsum('njk,k -> nj', Xdf, Bf)
    
    Xdr = Xd[:, :, rvidx]
    Br = _transform_rand_betas_jax(betas_mean, betas_sd_unconstrained, draws, rvidx, rvdist)
    Vdr = jnp.einsum("njk,nkr -> njr", Xdr, Br)
    
    Vd = Vdf[:, :, None] + Vdr
    if scale_d is not None:
        Vd -= lambdac * scale_d[:, :, None]
    if addit_d is not None:
        Vd += lambdac * addit_d[:, :, None]
        
    eVd = jnp.exp(Vd)
    
    if avail is not None:
        eVd *= avail[:, :, None]
    
    log_proba_n_draws = -jnp.log1p(eVd.sum(axis=1))
    
    if panel_ids is not None:
        log_lik_panel_draws = jax.ops.segment_sum(log_proba_n_draws, panel_ids, num_segments=num_panels)
        lik = jnp.exp(log_lik_panel_draws).mean(axis=1)
    else:
        lik = jnp.exp(log_proba_n_draws).mean(axis=1)

    loglik_n = jnp.log(jnp.maximum(lik, 1e-200))
    
    if weights is not None:
        loglik_n *= weights
        
    total_loss = -jnp.sum(loglik_n)
    
    return total_loss, None

class MixedLogit(ChoiceModel):
    def __init__(self):
        super(MixedLogit, self).__init__()
        self._rvidx = None
        self._rvdist = None

    def fit(self, X, y, varnames, alts, ids, randvars, isvars=None, weights=None, avail=None,  panels=None,
            base_alt=None, fit_intercept=False, init_coeff=None, maxiter=5000, random_state=None, n_draws=1000,
            halton=True, verbose=1, batch_size=None, halton_opts=None, tol_opts=None, robust=False, num_hess=False,
            scale_factor=None, optim_method="BFGS", mnl_init=True, addit=None, skip_std_errs=False):
        X, y, varnames, alts, isvars, ids, weights, panels, avail, scale_factor, addit \
            = self._as_array(X, y, varnames, alts, isvars, ids, weights, panels, avail, scale_factor, addit)

        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        if mnl_init and init_coeff is None:
            mnl = MultinomialLogit()
            init_isvars = [v for v in (isvars or []) if v not in randvars]
            mnl.fit(X, y, varnames, alts, ids, isvars=init_isvars, weights=weights, addit=addit,
                    avail=avail, base_alt=base_alt, fit_intercept=fit_intercept, skip_std_errs=True)
            
            initial_sd_unconstrained = np.log(np.exp(0.2) - 1)
            
            init_coeff = np.concatenate((mnl.coeff_, np.repeat(initial_sd_unconstrained, len(randvars))))
            if scale_factor is not None:
                init_coeff = np.append(init_coeff, 1.)

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)

        betas, X_design, y_design, panel_ids, draws_design, weights_design, avail_design, Xnames, scale_design, addit_design = \
            self._setup_input_data(X, y, varnames, alts, ids, randvars, isvars=isvars, weights=weights, avail=avail,
                                   panels=panels, init_coeff=init_coeff, random_state=random_state, n_draws=n_draws,
                                   halton=halton, verbose=verbose, predict_mode=False, halton_opts=halton_opts,
                                   scale_factor=scale_factor, addit=addit)

        Xd, scale_d, addit_d, avail_d = diff_nonchosen_chosen(X_design, y_design, scale_design, addit_design, avail_design)

        if panel_ids is not None:
            num_panels = panel_ids.max() + 1
            sample_size = num_panels
        else:
            num_panels = X_design.shape[0]
            sample_size = X_design.shape[0]

        betas_device, Xd, draws_design, panel_ids, weights_design, scale_d, addit_d, avail_d = map(
            device.to_device, [betas, Xd, draws_design, panel_ids, weights_design, scale_d, addit_d, avail_d]
        )
        
        rvidx_static = tuple(self._rvidx.tolist())
        rvdist_static = tuple(self._rvdist)
        num_mean_params = len(Xnames)

        loss_and_grad_fn = jax.value_and_grad(
            lambda p, *a: _mxl_loglik_calculator(p, *a, rvidx_static=rvidx_static, rvdist_static=rvdist_static, 
                                                 num_panels=num_panels, num_mean_params=num_mean_params), 
            has_aux=True
        )
        fargs = (Xd, draws_design, panel_ids, weights_design, scale_d, addit_d, avail_d)
        
        gtol = tol_opts.get('gtol', 1e-5) if tol_opts else 1e-5
        optim_res = _minimize_jax(loss_and_grad_fn, betas_device, args=fargs,
                                  maxiter=maxiter, gtol=gtol, disp=verbose > 0)

        unconstrained_params = device.to_cpu(optim_res['x'])
        
        final_params = unconstrained_params.copy()
        num_sd_params = len(self._rvdist)
        sd_params_unconstrained = final_params[num_mean_params : num_mean_params + num_sd_params]
        final_params[num_mean_params : num_mean_params + num_sd_params] = jax.nn.softplus(sd_params_unconstrained)
        optim_res['x'] = final_params
        
        if robust:
             grad_fn_per_choice = jax.grad(lambda p, x, d, w, s, a, av: -_mxl_loglik_calculator(p, x[None,:], d[None,:], None, w, s, a, av, rvidx_static, rvdist_static, num_panels=1, num_mean_params=num_mean_params)[0])
             grad_n_choice = jax.vmap(grad_fn_per_choice, in_axes=(None, 0, 0, 0, 0, 0, 0))(device.to_device(unconstrained_params), Xd, draws_design, weights_design, scale_d, addit_d, avail_d)
             if panel_ids is not None:
                grad_n = jax.ops.segment_sum(grad_n_choice, device.to_cpu(panel_ids), num_segments=num_panels)
             else:
                grad_n = grad_n_choice
             optim_res['grad_n'] = device.to_cpu(grad_n)
        else:
             optim_res['grad_n'] = np.zeros((sample_size, len(betas)))

        coef_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))
        if scale_factor is not None:
            coef_names = np.append(coef_names, "_scale_factor")

        if skip_std_errs:
            optim_res['hess_inv'] = np.eye(len(optim_res['x']))
        else:
            loss_fn_for_hess = lambda p, *a: _mxl_loglik_calculator(p, *a, rvidx_static=rvidx_static, rvdist_static=rvdist_static, num_panels=num_panels, num_mean_params=num_mean_params)[0]
            hess_inv_unconstrained = _numerical_hessian_jax(device.to_device(unconstrained_params), loss_fn_for_hess, args=fargs)
            
            grad_transform = jax.grad(lambda s: jax.nn.softplus(s).sum())(sd_params_unconstrained)
            
            J_transform = np.eye(len(final_params))
            J_transform[num_mean_params:, num_mean_params:] = np.diag(grad_transform)
            
            hess_inv_constrained = J_transform @ device.to_cpu(hess_inv_unconstrained) @ J_transform.T
            optim_res['hess_inv'] = hess_inv_constrained

        self._post_fit(optim_res, coef_names, sample_size, verbose, robust)

    def predict(self, X, varnames, alts, ids, isvars=None, weights=None, avail=None,  panels=None, random_state=None,
                n_draws=1000, halton=True, verbose=1, batch_size=None, return_proba=False, return_freq=False,
                halton_opts=None, scale_factor=None, addit=None):
        X, _, varnames, alts, isvars, ids, weights, panels, avail, scale_factor, addit \
            = self._as_array(X, None, varnames, alts, isvars, ids, weights, panels, avail, scale_factor, addit)
        
        self._validate_inputs(X, None, alts, varnames, isvars, ids, weights)
        
        unconstrained_coeffs = self.coeff_.copy()
        num_mean_params = len(unconstrained_coeffs) - len(self.randvars) - (1 if scale_factor else 0)
        num_sd_params = len(self.randvars)
        constrained_sds = unconstrained_coeffs[num_mean_params : num_mean_params + num_sd_params]
        unconstrained_sds = np.log(np.expm1(constrained_sds)) # More stable inverse of softplus
        unconstrained_coeffs[num_mean_params : num_mean_params + num_sd_params] = unconstrained_sds
        
        betas, X_design, _, _, draws, _, avail_design, Xnames, scale, addit_design = \
            self._setup_input_data(X, None, varnames, alts, ids, self.randvars,  isvars=isvars, weights=weights,
                                   avail=avail, panels=panels, init_coeff=unconstrained_coeffs, random_state=random_state,
                                   n_draws=n_draws, halton=halton, verbose=verbose, predict_mode=True,
                                   halton_opts=halton_opts, scale_factor=scale_factor, addit=addit)

        betas, X_design, draws, scale, addit_design, avail_design = map(device.to_device, [betas, X_design, draws, scale, addit_design, avail_design])

        rvidx = self._rvidx
        rvdist = self._rvdist
        
        if scale_factor is not None:
            lambdac = betas[-1]
            betas_full = betas[:-1]
        else:
            lambdac = 1.0
            betas_full = betas

        betas_mean = betas_full[:num_mean_params]
        betas_sd_unconstrained = betas_full[num_mean_params:]

        Xf = X_design[:, :, ~rvidx]
        Xr = X_design[:, :, rvidx]
        Bf = betas_mean[~rvidx]
        Vf = jnp.einsum('njk,k -> nj', Xf, Bf)
        
        Br = _transform_rand_betas_jax(betas_mean, betas_sd_unconstrained, draws, rvidx, rvdist)
        Vr = jnp.einsum("njk,nkr -> njr", Xr, Br)
        
        V = Vf[:, :, None] + Vr
        if scale is not None:
            V -= lambdac * scale[:, :, None]
        if addit_design is not None:
            V += lambdac * addit_design[:, :, None]

        eV = jnp.exp(V)
        if avail_design is not None:
            eV *= avail_design[:, :, None]
            
        sum_eV = eV.sum(axis=1, keepdims=True)
        proba_draws = eV / jnp.where(sum_eV == 0, 1, sum_eV)
        
        proba = proba_draws.mean(axis=-1)
        proba = device.to_cpu(proba)
        
        idx_max_proba = np.argmax(proba, axis=1)
        choices = self.alternatives[idx_max_proba]
        
        output = (choices, )
        if return_proba:
            output += (proba, )
        
        if return_freq:
            alt_list, counts = np.unique(choices, return_counts=True)
            freq = dict(zip(list(alt_list), list(np.round(counts/np.sum(counts), 3))))
            output += (freq, )
      
        return _unpack_tuple(output)

    def _setup_input_data(self, X, y, varnames, alts, ids, randvars, isvars=None, weights=None, avail=None,
                          panels=None, init_coeff=None, random_state=None, n_draws=200, halton=True, verbose=1,
                          predict_mode=False, halton_opts=None, scale_factor=None, addit=None):
        if random_state is not None:
            np.random.seed(random_state)

        self._check_long_format_consistency(ids, alts)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        
        self._isvars = [v for v in (isvars or []) if v not in randvars]
        self._asvars = [v for v in varnames if v not in self._isvars]
        
        X, Xnames = self._setup_design_matrix(X)
        self._model_specific_validations(randvars, Xnames)

        N, J, K = X.shape[0], X.shape[1], X.shape[2]
        # Kr should be the number of random variables, not the number of columns
        Kr = len(randvars)
        Ks = 1 if scale_factor is not None else 0

        panel_ids = None
        if panels is not None:
            unique_panels, panel_ids = np.unique(panels.reshape(N, J)[:, 0], return_inverse=True)
            sort_order = np.argsort(panel_ids)
            X, panel_ids = X[sort_order], panel_ids[sort_order]
            if y is not None: y = y[sort_order]
            if weights is not None: weights = weights.reshape(N, J)[:, 0][sort_order]
            if avail is not None: avail = avail.reshape(N, J)[sort_order]
            if scale_factor is not None: scale_factor = scale_factor.reshape(N,J)[sort_order]
            if addit is not None: addit = addit.reshape(N,J)[sort_order]

        if not predict_mode:
            self._setup_randvars_info(randvars, Xnames)

        self.n_draws = n_draws
        self.verbose = verbose

        if avail is not None:
            avail = avail.reshape(N, J)

        # Determine number of samples for draws (panels or individuals)
        num_draw_samples = (panel_ids.max() + 1) if panel_ids is not None else N
        draws = self._generate_draws(num_draw_samples, n_draws, halton, halton_opts=halton_opts)
        # If we have panels, we need to expand draws to match the number of choices
        if panel_ids is not None:
            draws = draws[panel_ids]
      
        if weights is not None and panels is not None:
            _, first_indices = np.unique(panel_ids, return_index=True)
            weights = weights[first_indices]
        elif weights is not None:
            weights = weights.reshape(N, J)[:, 0]

        if init_coeff is None:
            betas = np.repeat(.1, K)
            initial_sd_unconstrained = np.log(np.exp(0.2) - 1)
            betas = np.concatenate((betas, np.repeat(initial_sd_unconstrained, Kr)))
            if Ks > 0:
                betas = np.append(betas, 1.0)
        else:
            betas = np.asarray(init_coeff)
            if len(init_coeff) != (K + Kr + Ks):
                raise ValueError(f"初始系数的长度必须是: {K + Kr + Ks}, 但得到的是 {len(init_coeff)}")
        
        scale = None if scale_factor is None else scale_factor.reshape(N, J)
        addit = None if addit is None else addit.reshape(N, J)

        if device.using_gpu and verbose > 0:
            print("GPU 处理已通过 JAX 启用。")
        return betas, X, y, panel_ids, draws, weights, avail, Xnames, scale, addit

    def _setup_randvars_info(self, randvars, Xnames):
        self.randvars = randvars
        self._rvidx = np.array([ (var.split('.')[0] in self.randvars) for var in Xnames])
        self._rvdist = [self.randvars[var.split('.')[0]] for var in Xnames if (var.split('.')[0] in self.randvars)]

    def _generate_draws(self, sample_size, n_draws, halton=True, halton_opts=None):
        if halton:
            draws = self._generate_halton_draws(sample_size, n_draws, len(self._rvdist), **(halton_opts or {}))
        else:
            draws = self._generate_random_draws(sample_size, n_draws, len(self._rvdist))

        for k, dist in enumerate(self._rvdist):
            if dist in ['n', 'ln', 'tn']:
                draws[:, k, :] = scipy.stats.norm.ppf(draws[:, k, :])
            elif dist == 't':
                draws_k = draws[:, k, :]
                draws[:, k, :] = np.where(draws_k <= .5, np.sqrt(2*draws_k) - 1, 1 - np.sqrt(2*(1 - draws_k)))
            elif dist == 'u':
                draws[:, k, :] = 2*draws[:, k, :] - 1
        return draws

    def _generate_random_draws(self, sample_size, n_draws, n_vars):
        return np.random.uniform(size=(sample_size, n_vars, n_draws))

    # *** KEY CHANGE: Reverted to the original Halton implementation ***
    def _generate_halton_draws(self, sample_size, n_draws, n_vars, shuffle=False, drop=100, primes=None):
        if primes is None:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 71, 73, 79, 83, 89, 97, 101,
                      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                      199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311]
        
        def halton_seq(length, prime=3, shuffle=False, drop=100):
            req_length = length + drop
            seq = np.empty(req_length)
            seq[0] = 0
            seq_idx = 1
            t=1
            while seq_idx < req_length:
                d = 1/prime**t
                seq_size = seq_idx
                i = 1
                while i < prime and seq_idx < req_length:
                    max_seq = min(req_length - seq_idx, seq_size)
                    seq[seq_idx: seq_idx+max_seq] = seq[:max_seq] + d*i
                    seq_idx += max_seq
                    i += 1
                t += 1
            seq = seq[drop:length+drop]
            if shuffle:
                np.random.shuffle(seq)
            return seq

        draws = [halton_seq(sample_size*n_draws, prime=primes[i % len(primes)],
                            shuffle=shuffle, drop=drop).reshape(sample_size, n_draws) for i in range(n_vars)]
        draws = np.stack(draws, axis=1)
        return draws

    def _model_specific_validations(self, randvars, Xnames):
        if randvars is None:
            raise ValueError("混合 Logit 估计需要 'randvars' 参数")
        base_xnames = {name.split('.')[0] for name in Xnames}
        if not set(randvars.keys()).issubset(base_xnames):
            missing = set(randvars.keys()) - base_xnames
            raise ValueError(f"'randvars' 中的变量在变量名列表中未找到: {missing}")
        if not set(randvars.values()).issubset(["n", "ln", "t", "tn", "u"]):
            raise ValueError("'randvars' 中的混合分布错误。可接受的分布是 n, ln, t, u, tn")

    def summary(self):
        super(MixedLogit, self).summary()

    @staticmethod
    def check_if_gpu_available():
        is_available = device.using_gpu
        if is_available:
            print(f"{device.get_device_count()} 个 GPU 设备可用。xlogit 将使用 JAX 进行 GPU 处理。")
        else:
            print("** * JAX 未找到 GPU 设备。请验证 JAX 安装。将使用 CPU。")
        return is_available
