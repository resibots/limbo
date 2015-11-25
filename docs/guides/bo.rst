Introduction to Bayesian Optimization (BO) and to Limbo-specific concepts
==========================================================================

General concepts of Bayesian optimization
-----------------------------------------

Bayesian optimization is a model-based, black-box optimization algorithm that is tailored for very expensive objective functions (a.k.a. cost functions) :cite:`brochu2010tutorial,Mockus2013`. As a black-box optimization algorithm, Bayesian optimization searches for the maximum of an unknown objective function from which samples can be obtained (e.g., by measuring the performance of a robot). Like all model-based optimization algorithms (e.g. surrogate-based algorithms, kriging, or DACE), Bayesian optimization creates a model of the objective function with a regression method, uses this model to select the next point to acquire, then updates the model, etc. It is called *Bayesian* because, in its general formulation :cite:`Mockus2013`, this algorithm chooses the next point by computing a posterior distribution of the objective function using the likelihood of the data already acquired and a prior on the type of function.


.. figure:: ../pics/bo_concept.png
   :alt: concept of Bayesian optimization

   **Bayesian Optimization of a toy problem.** (A) The goal of this toy prob- lem is to find the maximum of the unknown objective function. (B) The Gaussian process is initialized, as it is customary, with a constant mean and a constant variance. (C) The next potential solution is selected and evaluated. The model is then updated according to the acquired data. (D) Based on the new model, another potential solution is selected and evaluated. (E-G) This process repeats until the maximum is reached.

.. _gaussian-process:

Gaussian Process
^^^^^^^^^^^^^^^^^

Here we use Gaussian process regression to find a model :cite:`Rasmussen2006`, which is a common choice for Bayesian optimization :cite:`brochu2010tutorial`. Gaussian processes are particularly interesting for regression because they not only model the cost function, but also the uncertainty associated with each prediction. For a cost function :math:`f`, usually unknown, a Gaussian process defines the probability distribution of the possible values :math:`f(\mathbf{x})` for each point :math:`\mathbf{x}`. These probability distributions are Gaussian, and are therefore defined by a mean (:math:`\mu`) and a standard deviation (:math:`\sigma`). However, :math:`\mu` and :math:`\sigma` can be different for each :math:`\mathbf{x}`; we therefore define a probability distribution *over functions*:

.. math::
  P(f(\mathbf{x})|\mathbf{x}) = \mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))

where :math:`\mathcal{N}` denotes the standard normal distribution.


To estimate :math:`\mu(\mathbf{x})` and :math:`\sigma(\mathbf{x})`, we need to fit the Gaussian process to the data. To do so, we assume that each observation :math:`f(\mathbf{\chi})` is a sample from a normal distribution. If we have a data set made of several observations, that is, :math:`f(\mathbf{\chi}_1), f(\mathbf{\chi}_2), ..., f(\mathbf{\chi}_t)`, then the vector :math:`\left[f(\mathbf{\chi}_1), f(\mathbf{\chi}_2), ..., f(\mathbf{\chi}_t)\right]` is a sample from a *multivariate* normal distribution, which is defined by a mean vector and a covariance matrix. A Gaussian process is therefore a generalization of a :math:`n`-variate normal distribution, where :math:`n` is the number of observations. The covariance matrix is what relates one observation to another: two observations that correspond to nearby values of :math:`\chi_1` and :math:`\chi_2` are likely to be correlated (this is a prior assumption based on the fact that functions tend to be smooth, and is injected into the algorithm via a prior on the likelihood of functions), two observations that correspond to distant values of :math:`\chi_1` and :math:`\chi_2` should not influence each other (i.e. their distributions are not correlated). Put differently, the covariance matrix represents that distant samples are almost uncorrelated and nearby samples are strongly correlated. This covariance matrix is defined via a *kernel function*, called :math:`k(\chi_1, \chi_2)`, which is usually based on the Euclidean distance between :math:`\chi_1` and :math:`\chi_2` (see the "kernel function" sub-section below).

Given a set of observations :math:`\mathbf{P}_{1:t}=f(\mathbf{\chi}_{1:t})` and a sampling noise :math:`\sigma^2_{noise}` (which is a user-specified parameter), the Gaussian process is computed as follows :cite:`brochu2010tutorial,Rasmussen2006`:

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

Our implementation of Bayesian optimization uses this Gaussian process model to search for the maximum of the objective function :math:`f(\mathbf{x})`, :math:`f(\mathbf{x})` being unknown. It selects the next :math:`\chi` to test by selecting the maximum of the *acquisition function*, which balances exploration -- improving the model in the less explored parts of the search space -- and exploitation -- favoring parts that the models predicts as promising. Here, we use the "Upper Confidence Bound" acquisition function (see the "information acquisition function" section below). Once the observation is made, the algorithm updates the Gaussian process to take the new data into account. In classic Bayesian optimization, the Gaussian process is initialized with a constant mean because it is assumed that all the points of the search space are equally likely to be good. The model is then progressively refined after each observation.

Optimizing the hyper-parameters of a Gaussian process
......................................................

.. _kernel-functions:

Kernel function
^^^^^^^^^^^^^^^^^

The kernel function is the covariance function of the Gaussian
process. It defines the influence of a solution's performance on the performance and confidence estimations of
not-yet-tested solutions that are nearby.

The Squared Exponential covariance function and the Matern kernel are the most common kernels for Gaussian processes :cite:`brochu2010tutorial,Rasmussen2006`. Both kernels are variants of the "bell curve". The Matern kernel is more general (it includes the Squared Exponential function as a special case) and  allows us to control not only the distance at which effects become nearly zero (as a function of parameter :math:`\rho`), but also the rate at which distance effects decrease (as a function of parameter :math:`\nu`).

The Matern kernel function is computed as follows :cite:`matern1960spatial,stein1999interpolation` (with :math:`\nu=5/2`):

.. math ::
  \begin{array}{l}
  k(\mathbf{x}_1,\mathbf{x}_2)=\left(1+ \frac{\sqrt{5}d(\mathbf{x}_1,\mathbf{x}_2)}{\rho}+\frac{5d(\mathbf{x}_1,\mathbf{x}_2)^2}{3\rho^2}\right)\exp\left(-\frac{\sqrt{5}d(\mathbf{x}_1,\mathbf{x}_2)}{\rho}\right)\\
  \textrm{where }d(\mathbf{x}_1,\mathbf{x}_2) \textrm{ is the Euclidean distance.}
  \end{array}

.. _acqui-functions:


There are other kernel functions in Limbo, and it is easy to define more. See :ref:`the Limbo implementation guide <kernel-guide>` for the available kernel functions.

Acquisition function
^^^^^^^^^^^^^^^^^^^^^

The selection of the next solution to evaluate is made by
finding the solution that maximizes the acquisition function. This
step is another optimization problem, but does not require testing the controller in simulation or reality. In
general, for this optimization problem we can derive the exact
equation and find a solution with gradient-based optimization, or use any other optimizer (e.g. CMA-ES)

Several different acquisition functions exist, such as the probability
of improvement, the expected improvement, or the Upper Confidence
Bound (UCB) :cite:`brochu2010tutorial`. For instance, the
equation for the UCB is:

.. math::

  \mathbf{x}_{t+1}= \operatorname*{arg\,max}_\mathbf{x} (\mu_{t}(\mathbf{x})+ \kappa\sigma_t(\mathbf{x}))
  \label{ucb}

where :math:`\kappa` is a user-defined parameter that tunes the tradeoff between exploration and exploitation.

The acquisition function handles the exploitation/exploration trade-off. In the UCB function, the emphasis on exploitation vs. exploration is explicit and easy to adjust. The UCB function can be seen as the maximum value (argmax) across all solutions of the weighted sum of the expected performance (mean of the Gaussian, :math:`\mu_{t}(\mathbf{x})`) and of the uncertainty (standard deviation of the Gaussian, :math:`\sigma_t(\mathbf{x})`) of each solution. This sum is weighted by the :math:`\kappa` factor. With a low :math:`\kappa`, the algorithm will choose solutions that are expected to be high-performing. Conversely, with a high :math:`\kappa`, the algorithm will focus its search on unexplored areas of the search space that may have high-performing solutions. The
:math:`\kappa` factor enables fine adjustments to the
exploitation/exploration trade-off of the algorithm.

There are other acquisition functions in Limbo, and it is easy to define more. See :ref:`the Limbo implementation guide <acquisition-guide>` for the available acquisition functions.

Limbo-specific concepts
-----------------------

.. _mean-functions:

Mean function
^^^^^^^^^^^^^
In classic Bayesian optimization, the Gaussian process is initialized with a constant mean because it is assumed that all the points of the search space are equally likely to be good. The model is then progressively refined after each observation. This constant mean is is one of the main prior used to build the Gaussian process: setting its value is often critical for the performance of the algorithm (see :cite:`lizotte2007automatic`).

Nevertheless, it can be useful to use more complex priors. This is, for instance, the case when we can use a low-fidelity simulator as a prior for physical experiments with a robot :cite:`cully_robots_2015`. To incorporate this idea into the Bayesian optimization, *limbo* models the *difference* between the prediction of the behavior-performance map and the actual performance on the real robot, instead of directly modeling the objective function. This idea is incorporated into the Gaussian process by modifying the update equation for the mean function (:math:`\mu_t(\mathbf{x})`):

.. math::

  \mu_{t}(\mathbf{x})= \mathcal{P}(\mathbf{x}) + \mathbf{k}^\intercal\mathbf{K}^{-1}(\mathbf{P}_{1:t}-\mathcal{P}(\mathbf{\chi}_{1:t}))


where :math:`\mathcal{P}(\mathbf{x})` is the performance of :math:`\mathbf{x}` according to the simulation and :math:`\mathcal{P}(\mathbf{\chi}_{1:t})` is the performance of all the previous observations, also according to the simulation. Replacing :math:`\mathbf{P}_{1:t}` by :math:`\mathbf{P}_{1:t}-\mathcal{P}(\mathbf{\chi}_{1:t})` means that the Gaussian process models the difference between the actual performance :math:`\mathbf{P}_{1:t}` and the performance from the behavior-performance map :math:`\mathcal{P}(\mathbf{\chi}_{1:t})`. The term :math:`\mathcal{P}(\mathbf{x})` is the prediction given by the mean function (the behavior-performance map in :cite:`cully_robots_2015`).

See :ref:`the Limbo implementation guide <mean-guide>` for the available mean functions.


Black lists
^^^^^^^^^^^^

When performing physical experiments, it is possible that some solutions cannot be properly evaluated. For example, this situation happens often with a physical robot, typically because (1) The robot may be outside the sensor’s range, for example when the robot is not visible from the camera’s point of view, making it impossible to assess its performance and (2) The sensor may return intractable values (infinity, NaN,...).

Different solutions exist to deal with missing data. The simplest way consists in redoing the evaluation. This may work, but only if the problem is not deterministic, otherwise the algorithm will be continuously redoing the same, not working, evaluation. A second solution consists in assigning a very low value to the behavior’s performance, like a punishment. This approach will work with evolutionary algorithms because the corresponding individual will very likely be removed from the population in the next generation. By contrast, this approach will have a dramatic effect on algorithms using models of the reward function, like Bayesian Optimization, as the models will be completely distorted.

These different methods to deal with missing data do not fit well with the Bayesian Optimization framework. Limbo uses a different approach, compatible with Bayesian Optimization, which preserves the model’s stability. The overall idea is to encourage the algorithm to avoid regions around behaviors that could not be evaluated, which may contain other behaviors that are not evaluable too, but without providing any performance value, which is likely to increase the model’s instability.

In order to provide the information that some behaviors have already been tried, we define a blacklist of samples. Each time a behavior cannot be properly evaluated, this behavior is added into the blacklist (and not in the pool of tested behaviors). Because the performance value is not available, only the behavior’s location in the search space is added to the blacklist. In other words, the blacklists are a list of samples with missing performance data.
Thanks to this distinction between valid samples and blacklisted ones, the algorithm can consider only the valid samples when computing the mean of the Gaussian Process and both valid and blacklisted samples when computing the variance. By ignoring blacklisted samples, the mean will remain unchanged and free to move according to future observations  By contrast, the variance will consider both valid and blacklisted samples and will “mark” them as already explored .


State-based optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

In many applications, the tasks can be expressed according to the robot’s state. For example, reaching a target with a robotics arm means to place the robot’s end effector at a particular location and walking forward can be expressed as moving the center of mass of the robot. For robotics manipulation, the state of the robot can be extended with the state of the manipulated object. In the same way, all the observations can be expressed as a part of the robot’s state (the observable part).

Instead of modeling the performance function, it is sometimes more effective to use n Gaussian processes to model the state, and then combine these values into a single one for the acquisition function, using an **aggregator**.

Limbo implements this concept.

.. bibliography:: refs.bib
  :style: plain
