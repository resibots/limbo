Frequently Asked Questions
==========================

.. _faq-waf:

Why waf (and not Cmake, or <insert your favorite build system>)?
--------------------------------------------------------------------------


Short answer: because we used it in `Sferes2 <http://www.github.com/sferes2/sferes>`_ and we liked it!

Here is the longer answer. When we started Sferes2, around 2007, the free software community was starting to be tired of Automake/Autoconf (the *de facto* standard at that time) and it was looking for more modern alternatives. Two/Three software had some tractions:

- `Cmake <http://www.cmake.org>`_, which generates makefiles (like Autoconf / Automake), uses a custom language, and emphasizes Windows/Unix portability;

- `Scons <http://www.scons.org>`_, which can be seen as a set of high level python functions / objects to build software;

- `Waf <http://www.waf.org>`_, which was a bit like "a simpler scons"

We tried all of them.

Cmake uses a custom language that is not especially well designed in our opinion (it looks like a language from the 70s and it ignores most of the research in computer languages of the last 30 years). Generating makefiles is nice, but (1) it is slow (because generating files is slow), (2) it is a 40 year old syntax (which does the job, OK), and (3) it is not designed for parallel build (whereas all our computers are now multi-core). Compiling on both Unix and MS Windows is easy, but Windows is not our target platform. On the contrary, cross-compiling for embedded platforms (e.g. an ARM-based Raspberry Pi or a robot) was very hard with the first versions of Cmake (whereas it was easy with Autoconf / Automake!).

Scons is nice because it is based on Python, a nice, general purpose language that almost everybody know. It makes it easy to build complex build frameworks, but simple things can be not-that-simple to do (at least, in 2007, it probably improved since then). In addition, Scons does not generate makefiles, which makes it faster and more adapted to parallel builds.

Waf has all the advantages of Scons but the learning curve was less steep. It is based on Python, it has a nice modern feel, it is fast, and it parallelizes the builds by default. More importantly, using python also allows us to easily add services that is not typically part of a build system, for instance, to submit jobs to a cluster (we have to parse a JSON file, etc., which is trivial in python, but not in Cmake!). See :doc:`guides/framework`.

Overall, it seems that cmake won the "war" in the free software world, mainly because a few high-profile projects chose it instead of Scons or Waf -- the most prominent (and "contagious") project is most probably QT. However, the battle was tough. For instance, I remember that no build system was perfect for QT and someone even made a waf-based version of QT. Also, keep in mind that QT needs a build system that works very well on MS Windows, while we do not care (we have no Windows-based clusters and no Windows-based robots...). They have different goals. In robotics, ROS heavily relies on Cmake, but one could wonder if Catkin would have been faster/nicer/better if it had been based on waf or scons. Last, a few high profile projects chose waf. For instance, `Pebble <http://www.pebble.com>`_, the smart watches, or `Samba <http://www.samba.org>`_, the Windows-compatible file sharing system for Unix.

Where is the configuration file?
--------------------------------------------------

Short answer: There is no configuration file because we target developpers/researchers who want to write the smallest amount of code when they add a new functionnality/concept, and not "end-users" who want an external optimizer that they can easily call on their problem.

Long answer is in the :ref:`Parameters guide <params-guide>`.

Why am I getting "'NoLFOpt' should never be called!" assertion failure?
------------------------------------------------------------------------

Most probably, you are using the `BOptimizer` class and you have set an `hp_period` (rate at which the hyperparams are optimized) bigger than 0, but you are using a Gaussian Process model with no hyperparameters optimization. This should never happen. So, if you do not want to optimize any hyperparameters, set `hp_period` parameter to -1. On the other hand, if you want use a Gaussian Process model that does optimize the hyperparameters, check :ref:`here <gp-hpopt>` for available hyperparameters optimization options.

Why am I getting "'XXXLFOpt' was never called!" errors?
-------------------------------------------------------------------------------------------------------

Most probably, you are using the `BOptimizer` class and you have set an `hp_period` (rate at which the hyperparams are optimized) less than 1, but you are using a Gaussian Process model that optimizes the hyperparameters. This should never happen. If you want use a Gaussian Process model that does optimize the hyperparameters, set the `hp_period` parameter to a value bigger than 0. On the other hand, if you do not want to optimize any hyperparameters, set `hp_period` parameter to -1 and use a Gaussian Process model that does not optimize the hyperparameters. Check :ref:`here <gp-hpopt>` for available hyperparameters optimization options.

Why do I get "[NLOptNoGrad]: nlopt invalid argument"
----------------------------------------------------
We need optimizers to optimize the hyper-parameters; by default, we use unbounded optimization... but many optimizers in NLOpt do not support unbounded optimization (in particular, DIRECT). In that case, they just return and throw an exception "[NLOptNoGrad]: nlopt invalid argument". No optimization is performed.

The easiest fix is to use another optimizer (we suggest RProp). You can also add bounds (by creating a new hyper-parameter optimization procedure).

Why C++11? (and not <insert your favorite language>)?
-----------------------------------------------------
We have specific needs that mainly revolve around high-performance, minimzing boilerplate code, and easy interface with hardware and existing libraries:

- Easy interface with high-performance libraries (Intel MKL, multi-core parallelization, MPI, etc.), with hardware (robots, ROS, etc.), and with our existing code (e.g. Sferes2): we want to focus on the 'real code', and avoid writing interface code as much as possible;

- High-efficiency: template-based C++ provides a way to write algorithms in a very abstract way with zero or almost zero overhead (abstraction without the cost!);

- Static typing: we need as much help as possible from the compiler to avoid bugs in scientific code;

- Easy to install on remote clusters;

- Long-term use: our libraries will be used for at least 10 years in our group, therefore we want to use a language that will still exist in 10 years and that is not moving too fast (we do not want to rewrite our code every other month).

Modern C++11 appears to be a good choice to fulfill all these criteria: it is reasonably easy to use, very easy to interface with everything, and very high-performance... but we keep a close eye on `Julia <http://julialang.org>`_, `Scala <http://www.scala-lang.org>`_, and `Rust <http://www.rust-lang.org>`_!
