Statistics / Writing results
=================================================

.. highlight:: c++

Statistics are functors that are called at the end of each iteration of the Bayesian optimizer. Their job is to:

- write the results to files;
- write the current state of the optimization;
- write the data that are useful for your own analyses.

All the statistics are written in a directory called ``hostname_date_pid-number``. For instance: ``wallepro-perso.loria.fr_2016-05-13_16_16_09_72226``

Limbo provides a few classes for common uses (see :ref:`statistics-stats` for details):

- ``ConsoleSummary``: writes a summary to ``std::cout`` at each iteration of the algorithm
- ``AggregatedObservations``: records the value of each evaluation of the function (after aggregation) [filename ``aggregated_observations.dat``]
- ``BestAggregatedObservations``: records the best value observed so far after each iteration [filename ``aggregated_observations.dat``]
- ``Observations``: records the value of each evaluation of the function (before aggregation) [filename: ``observations.dat``]
- ``Samples``: records the position in the search space of the evaluated points [filename: ``samples.dat``]
- ``BestObservations``: records the best observation after each iteration? [filename ``best_observations.dat``]
- ``BestSamples``: records the position in the search space of the best observation after each iteration [filename: ``best_samples.dat``]

These statistics are for "advanced users":

- ``GPAcquisitions``
- ``GPKernelHParams``
- ``GPLikelihood``
- ``GPMeanHParams``

The default statistics list is::

  boost::fusion::vector<stat::Samples<Params>, stat::AggregatedObservations<Params>,
    stat::ConsoleSummary<Params>>

Writing your own statistics class
----------------------------------
Limbo only provides generic statistics classes. However, it is often useful to add user-defined statistics classes that are specific to a particular experiment.

All the statistics functors follow the same template:

.. code:: c++

  template <typename Params>
  struct Samples : public limbo::stat::StatBase<Params> {
      template <typename BO, typename AggregatorFunction>
      void operator()(const BO& bo, const AggregatorFunction&)
      {
        // code
      }
  };

In a few words, they take a `BO` object (instance of the Bayesian optimizer) and do what they want.


For instance, we could add a statistics class that writes the worst observation at each iteration. Here is how to write this functor:

.. literalinclude:: ../../src/tutorials/statistics.cpp
   :language: c++
   :linenos:
   :lines: 112-141

In order to configure the Bayesian optimizer to use our new statistics class, we first need to define a new statistics list which includes our new `WorstObservation`:


.. literalinclude:: ../../src/tutorials/statistics.cpp
   :language: c++
   :linenos:
   :lines: 147-151

Then, we use it to define the optimizer:

.. literalinclude:: ../../src/tutorials/statistics.cpp
   :language: c++
   :linenos:
   :lines: 154

The full source code is available in `src/tutorials/statistics.cpp` and reproduced here:

.. literalinclude:: ../../src/tutorials/statistics.cpp
   :language: c++
   :linenos:
   :lines: 48-
