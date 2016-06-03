Statistics / Writing results
=================================================

.. highlight:: c++

Statistics are functors that are called at the end of each iteration of the Bayesian optimizer. Their job is to:

- write the results to files;
- write the current state of the optimization;
- write the data that are useful for your own analyses.

All the statistics are written in a directory called ``hostname_date_pid-number``. For instance: ``wallepro-perso.loria.fr_2016-05-13_16_16_09_72226``

Limbo provides a few classes for common uses (see :ref:`statistics-stats` for details):

- ``ConsoleSummary``: write a summary to ``std::cout`` at each iteration of the algorithm
- ``AggregatedObservations``: what values of the evaluation function have been evaluated (after aggregation) [filename ``aggregated_observations.dat``]
- ``BestAggregatedObservations``: what is the best value at each iteration? [filename ``aggregated_observations.dat``]
- ``Observations``: what values of the evaluation function have been measured? [filename: filename: ``observations.dat``]
- ``Samples``: what points have been  tested? [filename: `samples.dat`]
- ``BestObservations``: what is the best observation for each iteration? [filename ``best_observations.dat``]
- ``BestSamples``: what is the best solution for each iteration? [filename: ``best_samples.dat``]
- ``BlSamples`: what candidate solutions have been blacklisted? [filename: ``bl_samples.dat``]

These statistics are for "advanced users":

- ``GPAcquisitions``
- ``GPKernelHParams``
- ``GPLikelihood``
- ``GPMeanHParams``

The default statistics list is: `boost::fusion::vector<stat::Samples<Params>, stat::AggregatedObservations<Params>, stat::ConsoleSummary<Params>>`

Writing your own statistics class
----------------------------------
Limbo only provides generic statistics class. Because they are generic, it is often useful to add user-defined statistics class that are specific to a particular experiment.

All the statistics functors follow the same template:
Template:

.. code:: c++

  template <typename Params>
  struct Samples : public limbo::stat::StatBase<Params> {
      template <typename BO, typename AggregatorFunction>
      void operator()(const BO& bo, const AggregatorFunction&, bool blacklisted)
      {
        // code
      }
  };

In a few words, they take a `BO` object  (in instance of the Bayesian optimizer) and do what they want.


For instance, we could add a statistics class that writes the worst observation at each iteration. Here is how to write this functor:


.. code:: c++

  template <typename Params>
  struct WorstObservation : public stat::StatBase<Params> {
      template <typename BO, typename AggregatorFunction>
      void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
      {
        // [optional] if statistics have been disabled or if there are no observations, we do not do anything
        if (!bo.stats_enabled() || bo.observations().empty())
            return;

        // [optional] we create a file to write / you can use your own file but remember that this method is called at each iteration (you need to create it in the constructor)
        this->_create_log_file(bo, "worst_observations.dat");

        // [optional] we add a header to the file to make it easier to read later
        if (bo.total_iterations() == 0)
            (*this->_log_file) << "#iteration worst_observation sample" << std::endl;

        // ----- search for the worst observation ----
        // 1. get the aggregated observations
        auto rewards = std::vector<double>(bo.observations().size());
        std::transform(bo.observations().begin(), bo.observations().end(), rewards.begin(), afun);
        // 2. search for the worst element
        auto min_e = std::min_element(rewards.begin(), rewards.end());
        auto min_obs = bo.observations()[std::distance(rewards.begin(), min_e)];
        auto min_sample = bo.samples()[std::distance(rewards.begin(), min_e)];

        // ----- write what we have found ------
        // the file is (*this->_log_file)
        (*this->_log_file) << bo.total_iterations() << " " << min_obs.transpose() << " " << min_sample.transpose() << std::endl;
      }
  };

Then we need to install it into the Bayesian optimiser. The first thing to do is to is to define a new statistics list which include our new `WorstObservation`:

.. code:: c++

  using stat_t =
    boost::fusion::vector<stat::ConsoleSummary<Params>,
                          stat::Samples<Params>,
                          stat::Observations<Params>,
                          WorstObservation<Params> >;

Then, we need to use when defining the optimizer:

.. code:: c++

    bayes_opt::BOptimizer<Params, statsfun<stat_t>> boptimizer;

The full source code is available in `src/tutorials/statistics.cpp`
