import jax
import jax.numpy as jnp
import optax
from functools import partial

def _minimize_jax(loss_and_grad_fn, x0, args, maxiter=5000, gtol=1e-6, disp=False):
    """
    使用 JAX 和 Optax (AMSGrad 优化器) 的优化例程。
    AMSGrad 通常比 Adam 更稳定。
    """
    # 使用一个更简单的固定学习率，因为复杂的模型有时对调度器很敏感
    optimizer = optax.amsgrad(learning_rate=0.01)
    opt_state = optimizer.init(x0)
    
    @jax.jit
    def step(params, opt_state, *args):
        (loss_val, _), grads = loss_and_grad_fn(params, *args)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, grads

    # 保存最佳参数
    best_params = x0
    best_loss = jnp.inf

    for i in range(maxiter):
        x0, opt_state, loss_val, grads = step(x0, opt_state, *args)
        grad_norm = jnp.linalg.norm(grads)

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = x0

        if disp and i % 100 == 0:
            print(f"迭代: {i} \t Log-Lik.= {-loss_val:.3f} \t |g|= {grad_norm:e}")

        if grad_norm < gtol:
            message = "梯度已接近于零"
            success = True
            break
    else:
        message = "达到最大迭代次数但未收敛"
        success = False
    
    # 使用找到的最佳参数
    x0 = best_params
    (final_loss, grad_n), grad_total = loss_and_grad_fn(x0, *args)

    return {
        'success': success,
        'x': x0,
        'fun': final_loss,
        'message': message,
        'grad_n': grad_n,
        'grad': grad_total,
        'nit': i + 1,
        'nfev': (i + 1),
    }


def _numerical_hessian_jax(x, fn, args):
    """
    使用 JAX 的自动微分计算 Hessian 矩阵。
    """
    loss_fn = lambda p, *a: fn(p, *a)
    hessian_mat = jax.hessian(loss_fn)(x, *args)
    
    try:
        hess_inv = jnp.linalg.inv(hessian_mat)
        return hess_inv
    except jnp.linalg.LinAlgError:
        print("警告：Hessian 矩阵是奇异的。使用伪逆。")
        return jnp.linalg.pinv(hessian_mat)
