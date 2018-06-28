.. _bayesian_optimization:

Introduction to Bayesian Optimization (BO)
==========================================

Bayesian optimization is a model-based, black-box optimization algorithm that is tailored for very expensive objective functions (a.k.a. cost functions) :cite:`a-brochu2010tutorial,a-Mockus2013`. As a black-box optimization algorithm, Bayesian optimization searches for the maximum of an unknown objective function from which samples can be obtained (e.g., by measuring the performance of a robot). Like all model-based optimization algorithms (e.g. surrogate-based algorithms, kriging, or DACE), Bayesian optimization creates a model of the objective function with a regression method, uses this model to select the next point to acquire, then updates the model, etc. It is called *Bayesian* because, in its general formulation :cite:`a-Mockus2013`, this algorithm chooses the next point by computing a posterior distribution of the objective function using the likelihood of the data already acquired and a prior on the type of function.


.. figure:: ../pics/bo_concept.png
   :alt: concept of Bayesian optimization

   **Bayesian Optimization of a toy problem.** (A) The goal of this toy problem is to find the maximum of the unknown objective function. (B) The Gaussian process is initialized, as it is customary, with a constant mean and a constant variance. (C) The next potential solution is selected and evaluated. The model is then updated according to the acquired data. (D) Based on the new model, another potential solution is selected and evaluated. (E-G) This process repeats until the maximum is reached.

.. _gaussian-process:

Gaussian Process
----------------


Limbo uses Gaussian process regression to find a model :cite:`a-Rasmussen2006`, which is a common choice for Bayesian optimization :cite:`a-brochu2010tutorial`. Gaussian processes are particularly interesting for regression because they not only model the cost function, but also the uncertainty associated with each prediction. For a cost function :math:`f`, usually unknown, a Gaussian process defines the probability distribution of the possible values :math:`f(\mathbf{x})` for each point :math:`\mathbf{x}`. These probability distributions are Gaussian, and are therefore defined by a mean (:math:`\mu`) and a standard deviation (:math:`\sigma`). However, :math:`\mu` and :math:`\sigma` can be different for each :math:`\mathbf{x}`; we therefore define a probability distribution *over functions*:

.. math::
  P(f(\mathbf{x})|\mathbf{x}) = \mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))

where :math:`\mathcal{N}` denotes the standard normal distribution.


To estimate :math:`\mu(\mathbf{x})` and :math:`\sigma(\mathbf{x})`, we need to fit the Gaussian process to the data. To do so, we assume that each observation :math:`f(\mathbf{\chi})` is a sample from a normal distribution. If we have a data set made of several observations, that is, :math:`f(\mathbf{\chi}_1), f(\mathbf{\chi}_2), ..., f(\mathbf{\chi}_t)`, then the vector :math:`\left[f(\mathbf{\chi}_1), f(\mathbf{\chi}_2), ..., f(\mathbf{\chi}_t)\right]` is a sample from a *multivariate* normal distribution, which is defined by a mean vector and a covariance matrix. A Gaussian process is therefore a generalization of a :math:`n`-variate normal distribution, where :math:`n` is the number of observations. The covariance matrix is what relates one observation to another: two observations that correspond to nearby values of :math:`\chi_1` and :math:`\chi_2` are likely to be correlated (this is a prior assumption based on the fact that functions tend to be smooth, and is injected into the algorithm via a prior on the likelihood of functions), two observations that correspond to distant values of :math:`\chi_1` and :math:`\chi_2` should not influence each other (i.e. their distributions are not correlated). Put differently, the covariance matrix represents that distant samples are almost uncorrelated and nearby samples are strongly correlated. This covariance matrix is defined via a *kernel function*, called :math:`k(\chi_1, \chi_2)`, which is usually based on the Euclidean distance between :math:`\chi_1` and :math:`\chi_2` (see the "kernel function" sub-section below).

Given a set of observations :math:`\mathbf{P}_{1:t}=f(\mathbf{\chi}_{1:t})` and a sampling noise :math:`\sigma^2_{noise}` (which is a user-specified parameter), the Gaussian process is computed as follows :cite:`a-brochu2010tutorial,a-Rasmussen2006`:

.. math::
  \begin{gathered}
   P(f(\mathbf{x})|\mathbf{P}_{1:t},\mathbf{x}) = \mathcal{N}(\mu_{t}(\mathbf{x}), \sigma_{t}^2(\mathbf{x}))\\
  \begin{array}{l}
   \mathrm{where:}\\
   \mu_{t}(\mathbf{x})= \mathbf{k}^\intercal\mathbf{K}^{-1}\mathbf{P}_{1:t}\\
   \sigma_{t}^2(\mathbf{x})=k(\mathbf{x},\mathbf{x}) - \mathbf{k}^\intercal\mathbf{K}^{-1}\mathbf{k}\\
   \mathbf{K}=\left[ \begin{array}{ c c c}
      k(\mathbf{\chi}_1,\mathbf{\chi}_1) &\cdots & k(\mathbf{\chi}_1,\mathbf{\chi}_{t}) \\
      \vdots   &  \ddots &  \vdots  \\
      k(\mathbf{\chi}_{t},\mathbf{\chi}_1) &  \cdots &  k(\mathbf{\chi}_{t},\mathbf{\chi}_{t})\end{array} \right]
  + \sigma_{noise}^2I\\
   \mathbf{k}=\left[ \begin{array}{ c c c c }k(\mathbf{x},\mathbf{\chi}_1) & k(\mathbf{x},\mathbf{\chi}_2) & \cdots & k(\mathbf{x},\mathbf{\chi}_{t}) \end{array} \right]
   \end{array}
  \end{gathered}

Our implementation of Bayesian optimization uses this Gaussian process model to search for the maximum  of the unknown objective function :math:`f(\mathbf{x})`. It selects the next :math:`\chi` to test by selecting the maximum of the *acquisition function*, which balances exploration -- improving the model in the less explored parts of the search space -- and exploitation -- favoring parts that the models predicts as promising. Once an observation is made, the algorithm updates the Gaussian process to take the new data into account. In classic Bayesian optimization, the Gaussian process is initialized with a constant mean because it is assumed that all the points of the search space are equally likely to be good. The model is progressively refined after each observation.


.. _likelihood:

Optimizing the hyper-parameters of a Gaussian process
------------------------------------------------------

A GP is fully specified by its mean function :math:`\mu(\mathbf{x})` and covariance function :math:`k(\chi_1, \chi_2)` (a.k.a. *kernell function*). Nevertheless, the kernel function often includes some parameters, called *hyperparameters*, that need to be tuned. For instance, one of the most common kernel is the Squared Exponential covariance function:

.. math::

    k_{SE}(\chi_1, \chi_2) = \sigma_f^2 \cdot \exp\left( -\frac{\left|\left|\chi_1 - \chi_2\right|\right|^2}{2 l^2}  \right)

For some datasets, it makes sense to hand-tune these parameters (e.g., when there are very few samples). Ideally, our objective should be to learn :math:`l^2` (characteristic length scale) and :math:`\sigma_f^2` (overall variance).

A classic way to do so is to maximize the probability of the data given the hyper-parameters :math:`\mathbf{\vartheta}` (there are other ways, e.g. cross-validation). We use a log because it makes the optimization simpler and does not change the result.

The marginal likelihood can be computed as follows:

.. math::


  \log p(\mathbf{P}_{1:t}\mid\boldsymbol{\chi}_{1:t},\boldsymbol{\theta})= -\frac{1}{2}(\mathbf{P}_{1:t}-\mu_0)^\intercal\mathbf{K}^{-1}(\mathbf{P}_{1:t}-\mu_0) - \frac{1}{2}\log\mid\mathbf{K}\mid - \frac{n}{2}\log2\pi


where :math:`\mu_0` is the mean function (prior).

Limbo provides many algorithms to optimize the likelihood. Some algorithms are gradient-free (e.g. CMA-ES), some others use the gradient of the log-likelihood (e.g. rprop), see :ref:`opt-tutorial` and the :ref:`the Limbo implementation guide <opt-api>`.

For more details, see :cite:`a-Rasmussen2006` (chapter 5).



.. _kernel-functions:

Kernel function
----------------

The kernel function is the covariance function of the Gaussian
process. It defines the influence of a solution's performance on the performance and confidence estimations of
not-yet-tested solutions that are nearby.

The Squared Exponential covariance function and the Matern kernel are the most common kernels for Gaussian processes :cite:`a-brochu2010tutorial,a-Rasmussen2006`. Both kernels are variants of the "bell curve". The Matern kernel is more general (it includes the Squared Exponential function as a special case) and  allows us to control not only the distance at which effects become nearly zero (as a function of parameter :math:`\rho`), but also the rate at which distance effects decrease (as a function of parameter :math:`\nu`).

The Matern kernel function is computed as follows :cite:`a-matern1960spatial,a-stein1999interpolation` (with :math:`\nu=5/2`):

.. math ::
  \begin{array}{l}
  k(\mathbf{x}_1,\mathbf{x}_2)=\left(1+ \frac{\sqrt{5}d(\mathbf{x}_1,\mathbf{x}_2)}{\rho}+\frac{5d(\mathbf{x}_1,\mathbf{x}_2)^2}{3\rho^2}\right)\exp\left(-\frac{\sqrt{5}d(\mathbf{x}_1,\mathbf{x}_2)}{\rho}\right)\\
  \textrm{where }d(\mathbf{x}_1,\mathbf{x}_2) \textrm{ is the Euclidean distance.}
  \end{array}

.. _acqui-functions:


There are other kernel functions in Limbo, and it is easy to define more. See :ref:`the Limbo implementation guide <kernel-api>` for the available kernel functions.

Acquisition function
--------------------

In order to find the next point to evaluate, we optimize the acquisition function over the model. This step is another optimization problem, but does not require evaluating the objective function. In general, for this optimization problem we can derive the exact equation and find a solution with gradient-based optimization, or use any other optimizer (e.g. CMA-ES).

Several different acquisition functions exist, such as the probability
of improvement, the expected improvement, or the Upper Confidence
Bound (UCB) :cite:`a-brochu2010tutorial`. For instance, the
equation for the UCB is:

.. math::

  \mathbf{x}_{t+1}= \operatorname*{arg\,max}_\mathbf{x} (\mu_{t}(\mathbf{x})+ \kappa\sigma_t(\mathbf{x}))
  \label{ucb}

where :math:`\kappa` is a user-defined parameter that tunes the tradeoff between exploration and exploitation.

Here, the emphasis on exploitation vs. exploration is explicit and easy to adjust. The UCB function can be seen as the maximum value (argmax) across all solutions of the weighted sum of the expected performance (mean of the Gaussian, :math:`\mu_{t}(\mathbf{x})`) and of the uncertainty (standard deviation of the Gaussian, :math:`\sigma_t(\mathbf{x})`) of each solution. This sum is weighted by the :math:`\kappa` factor. With a low :math:`\kappa`, the algorithm will choose solutions that are expected to be high-performing. Conversely, with a high :math:`\kappa`, the algorithm will focus its search on unexplored areas of the search space that may have high-performing solutions. The
:math:`\kappa` factor enables fine adjustments to the
exploitation/exploration trade-off of the algorithm.

There are other acquisition functions in Limbo, and it is easy to define more. See :ref:`the Limbo implementation guide <acqui-api>` for the available acquisition functions.

-----

.. bibliography:: refs.bib
  :style: plain
  :cited:
  :keyprefix: a-
