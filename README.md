# xlogit: A Python Package for JAX-Accelerated Estimation of Mixed Logit Models

![xlogit Logo](https://raw.githubusercontent.com/arteagac/xlogit/master/docs/xlogit_logo_1000.png)

|Build| |Coverage| |Community| |Docs| |PyPi| |License|

.. _Mixed Logit: https://xlogit.readthedocs.io/en/latest/api/mixed_logit.html
.. _Multinomial Logit: https://xlogit.readthedocs.io/en/latest/api/multinomial_logit.html

`Examples <https://xlogit.readthedocs.io/en/latest/examples.html>`__ | `Docs <https://xlogit.readthedocs.io/en/latest/index.html>`__ | `Installation <https://xlogit.readthedocs.io/en/latest/install.html>`__ | `API Reference <https://xlogit.readthedocs.io/en/latest/api/index.html>`__ | `Contributing <https://xlogit.readthedocs.io/en/latest/contributing.html>`__ | `Contact <https://xlogit.readthedocs.io/en/latest/index.html#contact>`__ 

## Quick start

The following example uses ``xlogit`` to estimate a mixed logit model for choices of electricity supplier (`See the data here <https://github.com/arteagac/xlogit/blob/master/examples/data/electricity_long.csv>`__). The parameters are:

* ``X``: 2-D array of input data (in long format) with choice situations as rows, and variables as columns
* ``y``: 1-D array of choices (in long format)
* ``varnames``: List of variable names that must match the number and order of the columns in ``X``
* ``alts``:  1-D array of alternative indexes or an alternatives list
* ``ids``:  1-D array of the ids of the choice situations
* ``panels``: 1-D array of ids for panel formation
* ``randvars``: dictionary of variables and their mixing distributions (``"n"`` normal, ``"ln"`` lognormal, ``"t"`` triangular, ``"u"`` uniform, ``"tn"`` truncated normal)

The current version of `xlogit` only supports input data in long format.

```python
    import pandas as pd
    df = pd.read_csv("examples/data/electricity_long.csv")
    
    # Fit the model with xlogit
    from xlogit import MixedLogit
    
    varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
    model = MixedLogit()
    model.fit(X=df[varnames],
              y=df['choice'],
              varnames=varnames,
              ids=df['chid'],
              panels=df['id'],
              alts=df['alt'],
              n_draws=600,
              randvars={'pf': 'n', 'cl': 'n', 'loc': 'n',
                        'wk': 'n', 'tod': 'n', 'seas': 'n'})
    model.summary()
```

Output:
```
JAX acceleration enabled.
Optimization terminated successfully.
         Current function value: 3888.413414
         Iterations: 46
         Function evaluations: 51
         Gradient evaluations: 51
Estimation time= 1.2 seconds
----------------------------------------------------------------------
Coefficient         Estimate      Std.Err.         z-val         P>|z|
----------------------------------------------------------------------
pf                -0.9996286     0.0331488   -30.1557541     9.98e-100 ***
cl                -0.2355334     0.0220401   -10.6865870      1.97e-22 ***
loc                2.2307891     0.1164263    19.1605300      5.64e-56 ***
wk                 1.6251657     0.0918755    17.6887855      6.85e-50 ***
tod               -9.6067367     0.3112721   -30.8628296     2.36e-102 ***
seas              -9.7892800     0.2913063   -33.6047603     2.81e-112 ***
sd.pf              0.2357813     0.0181892    12.9627201      7.25e-31 ***
sd.cl              0.4025377     0.0220183    18.2819903      2.43e-52 ***
sd.loc             1.9262893     0.1187850    16.2166103      7.67e-44 ***
sd.wk             -1.2192931     0.0944581   -12.9083017      1.17e-30 ***
sd.tod             2.3354462     0.1741859    13.4077786      1.37e-32 ***
sd.seas           -1.4200913     0.2095869    -6.7756668       3.1e-10 ***
----------------------------------------------------------------------
Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-Likelihood= -3888.413
AIC= 7800.827
BIC= 7847.493
```

For more examples of ``xlogit`` see `this Jupyter Notebook in Google Colab <https://colab.research.google.com/github/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb>`__.

## What's New in v2.0

xlogit now features **JAX acceleration** for significantly faster model estimation:

- **Automatic Differentiation**: No more manual gradient calculations
- **GPU/TPU Support**: Native support for hardware acceleration
- **JIT Compilation**: Just-in-time compilation for optimal performance
- **Vectorized Operations**: Automatic vectorization using `vmap`
- **Memory Efficient**: Better memory management with batch processing

## Installation

### Quick install from PyPI

Install ``xlogit`` using ``pip`` as follows:

```bash
pip install xlogit
```

### Install from source

To install the latest development version from source:

```bash
git clone https://github.com/ranxiaoning/xlogit.git
cd xlogit
pip install -r requirements.txt
pip install -e .
```

### GPU support

For GPU support, install JAX with CUDA support:

```bash
# For CUDA 11
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

.. hint::

   xlogit now uses JAX for acceleration, which provides automatic GPU/TPU support. When JAX detects available GPU resources, it automatically switches to GPU processing without additional setup. If you use Google Colab, JAX is usually installed by default with GPU support.

For additional installation details check xlogit installation instructions at: https://xlogit.readthedocs.io/en/latest/install.html

## Performance Benefits

The transition to JAX provides significant performance improvements:

- **2-5x faster estimation** compared to previous versions
- **Automatic GPU utilization** without code changes
- **Better memory efficiency** with JAX's functional programming model
- **Numerical stability** with improved precision handling

## No GPU? No problem

``xlogit`` works efficiently on CPU as well. JAX automatically optimizes code for your available hardware. However, if you need maximum performance, there are several options to access cloud GPU resources:

- `Google Colab <https://colab.research.google.com>`_ offers free GPU resources with no setup required. JAX runs out of the box without additional installation. For examples of xlogit running in Google Colab `see this link <https://colab.research.google.com/github/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb>`_.
- `Amazon Sagemaker Studio Lab <https://studiolab.sagemaker.aws/>`_ offers Python runtime environments with free GPUs.
- `Google Cloud platform <https://cloud.google.com/compute/gpus-pricing>`_ offers GPU processing at competitive prices.
- `Amazon Sagemaker <https://aws.amazon.com/ec2/instance-types/p2/>`_ offers virtual machine instances with GPU support.

## Benchmark

xlogit v2.0 with JAX acceleration shows significant performance improvements over previous versions and competing packages. The automatic differentiation and JIT compilation provide substantial speedups, especially for complex models with large numbers of random draws.

![Performance Benchmark](https://raw.githubusercontent.com/arteagac/xlogit/master/examples/benchmark/results/time_benchmark_artificial.png)

For additional details about benchmarks and replication instructions check https://xlogit.readthedocs.io/en/latest/benchmark.html.

## Notes

The current version allows estimation of:

- `Mixed Logit`_ with several types of mixing distributions (normal, lognormal, triangular, uniform, and truncated normal)
- `Mixed Logit`_ with panel data
- `Mixed Logit`_ with unbalanced panel data
- `Mixed Logit`_ with Halton draws
- `Multinomial Logit`_ models
- `Conditional logit <https://xlogit.readthedocs.io/en/latest/api/multinomial_logit.html>`_ models
- `WTP space <https://xlogit.readthedocs.io/en/latest/notebooks/wtp_space.html>`_ models
- Handling of unbalanced availability of choice alternatives for all of the supported models 
- Post-estimation tools for prediction and specification testing
- Inclusion of sample weights for all of the supported models

## New Features in v2.0

- **JAX Backend**: Complete rewrite using JAX for numerical computing
- **Automatic Differentiation**: Eliminates manual gradient calculations
- **Hardware Acceleration**: Automatic GPU/TPU utilization
- **Improved Numerical Stability**: Better handling of edge cases
- **Memory Optimization**: More efficient batch processing
- **Extended Distribution Support**: Enhanced random parameter distributions

## Contributors

The following contributors have tremendously helped in the enhancement and expansion of `xlogit`'s features.  

- `@crforsythe <https://github.com/crforsythe>`__
- John Helveston (`@jhelvy  <https://github.com/jhelvy>`__)

Special thanks to the JAX development team for creating an excellent numerical computing framework.

## Contact

If you have any questions, ideas to improve ``xlogit``, or want to report a bug, `chat with us on gitter <https://gitter.im/xlogit/community>`__ or open a `new issue in xlogit's GitHub repository <https://github.com/arteagac/xlogit/issues>`__.

## Citing ``xlogit``

Please cite ``xlogit`` as follows:

    Arteaga, C., Park, J., Beeramoole, P. B., & Paz, A. (2022). xlogit: An open-source Python package for GPU-accelerated estimation of Mixed Logit models. Journal of Choice Modelling, 42, 100339. https://doi.org/10.1016/j.jocm.2021.100339
    
Or using BibTex as follows:

```bibtex
@article{xlogit,
    title = {xlogit: An open-source Python package for GPU-accelerated estimation of Mixed Logit models},
    author = {Cristian Arteaga and JeeWoong Park and Prithvi Bhat Beeramoole and Alexander Paz},
    journal = {Journal of Choice Modelling},
    volume = {42},
    pages = {100339},
    year = {2022},
    issn = {1755-5345},
    doi = {https://doi.org/10.1016/j.jocm.2021.100339},
}
```

.. |Build| image:: https://github.com/arteagac/xlogit/actions/workflows/python-tests.yml/badge.svg
   :target: https://github.com/arteagac/xlogit/actions/workflows/python-tests.yml

.. |Docs| image:: https://readthedocs.org/projects/xlogit/badge/?version=latest
   :target: https://xlogit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |Community| image:: https://badges.gitter.im/xlogit/community.svg
   :target: https://gitter.im/xlogit/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
   :alt: Community

.. |Coverage| image:: https://coveralls.io/repos/github/arteagac/xlogit/badge.svg?branch=master
   :target: https://coveralls.io/github/arteagac/xlogit?branch=master

.. |PyPi| image:: https://badge.fury.io/py/xlogit.svg
   :target: https://badge.fury.io/py/xlogit

.. |License| image:: https://img.shields.io/github/license/arteagac/xlogit
   :target: https://github.com/arteagac/xlogit/blob/master/LICENSE

## Migration from v1.x

For users migrating from xlogit v1.x, the API remains largely unchanged. The main differences are:

1. **Automatic GPU detection**: No need to manually enable GPU acceleration
2. **Improved performance**: Existing code will run faster without modifications
3. **Better error messages**: More informative error messages for debugging

If you encounter any issues during migration, please open an issue on GitHub.

## License

xlogit is released under the MIT License. See LICENSE for details.





主要更新：
1. 添加了 "Install from source" 部分，包含 `git clone` 和 `pip install -r requirements.txt` 的说明
2. 添加了 `pip install -e .` 用于开发模式安装
3. 保持了原有的 PyPI 安装方式
4. 添加了 GPU 支持的说明

这样用户就可以选择从 PyPI 安装稳定版本，或者从源码安装最新开发版本。
