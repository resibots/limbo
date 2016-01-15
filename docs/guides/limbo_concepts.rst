Limbo-specific concepts
=======================

Limbo extends the traditionnal Bayesian optimization algorithm with a few ideas developped in our group.

.. _mean-functions:

Mean function
-------------

In classic Bayesian optimization, the Gaussian process is initialized with a constant mean because it is assumed that all the points of the search space are equally likely to be good. The model is then progressively refined after each observation. This constant mean is is one of the main prior used to build the Gaussian process: setting its value is often critical for the performance of the algorithm (see :cite:`lizotte2007automatic`).

Nevertheless, it can be useful to use more complex priors. This is, for instance, the case when we can use a low-fidelity simulator as a prior for physical experiments with a robot :cite:`cully_robots_2015`. To incorporate this idea into the Bayesian optimization, *limbo* models the *difference* between the prediction of the behavior-performance map and the actual performance on the real robot, instead of directly modeling the objective function. This idea is incorporated into the Gaussian process by modifying the update equation for the mean function (:math:`\mu_t(\mathbf{x})`):

.. math::

  \mu_{t}(\mathbf{x})= \mathcal{P}(\mathbf{x}) + \mathbf{k}^\intercal\mathbf{K}^{-1}(\mathbf{P}_{1:t}-\mathcal{P}(\mathbf{\chi}_{1:t}))


where :math:`\mathcal{P}(\mathbf{x})` is the performance of :math:`\mathbf{x}` according to the mean function (*the prior*) and :math:`\mathcal{P}(\mathbf{\chi}_{1:t})` is the performance of all the previous observations, also according to the mean function (prior).

Replacing :math:`\mathbf{P}_{1:t}` by :math:`\mathbf{P}_{1:t}-\mathcal{P}(\mathbf{\chi}_{1:t})` means that the Gaussian process models the difference between the actual performance :math:`\mathbf{P}_{1:t}` and the performance from the behavior-performance map :math:`\mathcal{P}(\mathbf{\chi}_{1:t})`. The term :math:`\mathcal{P}(\mathbf{x})` is the prediction given by the mean function (the behavior-performance map in :cite:`cully_robots_2015`).

See :ref:`the Limbo implementation guide <mean-guide>` for the available mean functions.

Black lists
-----------

When performing physical experiments, it is possible that some solutions cannot be properly evaluated. For example, this situation happens often with a physical robot, typically because (1) The robot may be outside the sensor’s range, for example when the robot is not visible from the camera’s point of view, making it impossible to assess its performance and (2) The sensor may return intractable values (infinity, NaN,...).

Different solutions exist to deal with missing data. The simplest way consists in redoing the evaluation. This may work, but only if the problem is not deterministic, otherwise the algorithm will be continuously redoing the same, not working, evaluation. A second solution consists in assigning a very low value to the behavior’s performance, like a punishment. This approach will work with evolutionary algorithms because the corresponding individual will very likely be removed from the population in the next generation. By contrast, this approach will have a dramatic effect on algorithms using models of the reward function, like Bayesian Optimization, as the models will be completely distorted.

These different methods to deal with missing data do not fit well with the Bayesian Optimization framework. Limbo uses a different approach, compatible with Bayesian Optimization, which preserves the model’s stability. The overall idea is to encourage the algorithm to avoid regions around behaviors that could not be evaluated, which may contain other behaviors that are not evaluable too, but without providing any performance value, which is likely to increase the model’s instability.

In order to provide the information that some behaviors have already been tried, we define a blacklist of samples. Each time a behavior cannot be properly evaluated, this behavior is added into the blacklist (and not in the pool of tested behaviors). Because the performance value is not available, only the behavior’s location in the search space is added to the blacklist. In other words, the blacklists are a list of samples with missing performance data.
Thanks to this distinction between valid samples and blacklisted ones, the algorithm can consider only the valid samples when computing the mean of the Gaussian Process and both valid and blacklisted samples when computing the variance. By ignoring blacklisted samples, the mean will remain unchanged and free to move according to future observations  By contrast, the variance will consider both valid and blacklisted samples and will “mark” them as already explored .


State-based optimization
------------------------

In many applications, the tasks can be expressed according to the robot’s state. For example, reaching a target with a robotics arm means to place the robot’s end effector at a particular location and walking forward can be expressed as moving the center of mass of the robot. For robotics manipulation, the state of the robot can be extended with the state of the manipulated object. In the same way, all the observations can be expressed as a part of the robot’s state (the observable part).

Instead of modeling the performance function, it is sometimes more effective to use n Gaussian processes to model the state, and then combine these values into a single one for the acquisition function, using an **aggregator**.

Limbo implements this concept.

.. bibliography:: refs.bib
  :style: plain
