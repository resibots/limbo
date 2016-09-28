Limbo-specific concepts
=======================

Limbo extends the traditionnal Bayesian optimization algorithm with a few ideas developped in our group.

.. _mean-functions:

Mean function
-------------

In classic Bayesian optimization, the Gaussian process is initialized with a constant mean because it is assumed that all the points of the search space are equally likely to be good. The model is then progressively refined after each observation. This constant mean is one of the main prior used to build the Gaussian process: setting its value is often critical for the performance of the algorithm (see :cite:`b-lizotte2007automatic`).

Nevertheless, it can be useful to use more complex priors. This is, for instance, the case when we can use a low-fidelity simulator as a prior for physical experiments with a robot :cite:`b-cully_robots_2015`. To incorporate this idea into the Bayesian optimization, *limbo* models the *difference* between the prediction of the behavior-performance map and the actual performance on the real robot, instead of directly modeling the objective function. This idea is incorporated into the Gaussian process by modifying the update equation for the mean function (:math:`\mu_t(\mathbf{x})`):

.. math::

  \mu_{t}(\mathbf{x})= \mathcal{P}(\mathbf{x}) + \mathbf{k}^\intercal\mathbf{K}^{-1}(\mathbf{P}_{1:t}-\mathcal{P}(\mathbf{\chi}_{1:t}))


where :math:`\mathcal{P}(\mathbf{x})` is the performance of :math:`\mathbf{x}` according to the mean function (*the prior*) and :math:`\mathcal{P}(\mathbf{\chi}_{1:t})` is the performance of all the previous observations, also according to the mean function (prior).

Replacing :math:`\mathbf{P}_{1:t}` by :math:`\mathbf{P}_{1:t}-\mathcal{P}(\mathbf{\chi}_{1:t})` means that the Gaussian process models the difference between the actual performance :math:`\mathbf{P}_{1:t}` and the performance from the behavior-performance map :math:`\mathcal{P}(\mathbf{\chi}_{1:t})`. The term :math:`\mathcal{P}(\mathbf{x})` is the prediction given by the mean function (the behavior-performance map in :cite:`b-cully_robots_2015`).

See :ref:`the Limbo implementation guide <mean-api>` for the available mean functions.

.. _state-based-bo:

State-based optimization
------------------------

In many applications, the tasks can be expressed according to the robot’s state. For example, reaching a target with a robotics arm means to place the robot’s end effector at a particular location and walking forward can be expressed as moving the center of mass of the robot. For robotics manipulation, the state of the robot can be extended with the state of the manipulated object. In the same way, all the observations can be expressed as a part of the robot’s state (the observable part).

Instead of modeling the performance function, it is sometimes more effective to use n Gaussian processes to model the state, and then combine these values into a single one for the acquisition function, using an **aggregator**.

Limbo implements this concept.

-----

.. bibliography:: refs.bib
  :style: plain
  :cited:
  :keyprefix: b-
