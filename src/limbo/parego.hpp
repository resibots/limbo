#ifndef PAREGO_HPP_
#define PAREGO_HPP_

#include <algorithm>
#include "bo_base.hpp"
#include "pareto.hpp"

namespace limbo {
  namespace defaults {
    struct parego {
      BO_PARAM(double, rho, 0.05);
    };
  }

  template <
    class Params
    , class A2 = boost::parameter::void_
    , class A3 = boost::parameter::void_
    , class A4 = boost::parameter::void_
    , class A5 = boost::parameter::void_
    , class A6 = boost::parameter::void_
    , class A7 = boost::parameter::void_
    >
  class Parego : public BoBase<Params, obs_type<Eigen::VectorXd>,
    A2, A3, A4, A5, A6, A7> {
   public:
    typedef BoBase<Params, obs_type<Eigen::VectorXd>, A2, A3, A4, A5, A6, A7> base_t;
    typedef typename base_t::obs_t obs_t;
    typedef typename base_t::model_t model_t;
    typedef typename base_t::inner_optimization_t inner_optimization_t;
    typedef typename base_t::acquisition_function_t acquisition_function_t;
    typedef std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> pareto_point_t;
    typedef std::vector<pareto_point_t> pareto_t;

    template<typename EvalFunction>
    void optimize(const EvalFunction& feval, bool reset = true) {
      this->_init(feval, reset);

      std::vector<double> scalarized = _scalarize_obs();
      model_t model(EvalFunction::dim);
      model.compute(this->_samples, scalarized, Params::boptimizer::noise());

      inner_optimization_t inner_optimization;

      while (this->_samples.size() == 0 || this->_pursue()) {
        acquisition_function_t acqui(model, this->_iteration);

        Eigen::VectorXd new_sample = inner_optimization(acqui, acqui.dim());
        this->add_new_sample(new_sample, feval(new_sample));
        std::cout << this->_iteration
                  << " | new sample:" << new_sample.transpose()
                  << " => " << feval(new_sample).transpose() << std::endl;
        scalarized = _scalarize_obs();
        model.compute(this->_samples, scalarized, Params::boptimizer::noise());
        this->_update_stats();

        this->_iteration++;
      }
    }

    pareto_t data_pareto_front() const {
      std::vector<double> v(this->_samples.size());
      std::fill(v.begin(), v.end(), 0.0f);
      return pareto::pareto_set(_pack_data(this->_samples, this->_observations, v));
    }

    pareto_t model_pareto_front(double min, double max, double inc) const {
      assert(this->_observations.size());
      size_t nb_objs = this->_observations[0].size();
      size_t dim = this->_samples[0].size();

      // compute a model for each dimension
      std::vector<std::vector<double> > uni_obs(nb_objs);
      for (size_t i = 0; i < this->_observations.size(); ++i)
        for (size_t j = 0; j < this->_observations[i].size(); ++j)
          uni_obs[j].push_back(this->_observations[i][j]);
      std::vector<model_t> models(nb_objs, model_t(nb_objs));
      for (size_t i = 0; i < uni_obs.size(); ++i) {
        for (size_t j = 0; j < uni_obs[i].size(); ++j)
          std::cout << uni_obs[i][j] << " ";
        std::cout << std::endl;
        models[i].compute(this->_samples, uni_obs[i], 0.0);
      }
      std::cout << "enumerating..." << std::endl;
      std::vector<Eigen::VectorXd> points;
      _enumerate(dim - 1, min, max, inc, Eigen::VectorXd::Zero(dim), points);

      std::cout << "evaluating models..." << std::endl;
      std::vector<Eigen::VectorXd> objs;
      std::vector<double> sigma;
      tie(objs, sigma) = _eval_models(points, models);

      std::ofstream ofs("model.dat");
      for (size_t i = 0; i < objs.size(); ++i)
        ofs << objs[i].transpose() << std::endl;

      std::cout << "building Pareto set" << std::endl;

      return pareto::pareto_set(_pack_data(points, objs, sigma));
    }
   protected:
    pareto_t _pack_data(const std::vector<Eigen::VectorXd>& points,
                        const std::vector<Eigen::VectorXd>& objs,
                        const std::vector<double>& sigma) const {
      assert(points.size() == objs.size());
      assert(sigma.size() == objs.size());
      pareto_t p(points.size());
      par::loop (0, p.size(), [&](size_t k) {
        p[k] = std::make_tuple(points[k], objs[k], sigma[k]);
      });
      return p;
    }

    std::tuple<std::vector<Eigen::VectorXd>, std::vector<double> >
    _eval_models(const std::vector<Eigen::VectorXd>& points, const std::vector<model_t>& models) const {
      std::vector<Eigen::VectorXd> objs(points.size());
      std::vector<double> sigma(points.size());
      std::fill(sigma.begin(), sigma.end(), 0.0);
      int nb_objs = models.size();
      par::loop(size_t(0), points.size(), [&](size_t k) {
        if (k % 10000 == 0) {
          std::cout << k << " [" << points.size() << "] ";
          std::cout.flush();
        }
        Eigen::VectorXd p(nb_objs);
        for (size_t i = 0; i < p.size(); ++i) {
          p[i] = models[i].mu(points[k]);
          sigma[k] = std::max(sigma[k], models[i].sigma(points[k]));
        }
        objs[k] = p;
      });
      return std::make_tuple(objs, sigma);
    }

    void _enumerate(int dim, double min, double max, double inc,
                    Eigen::VectorXd p,
                    std::vector<Eigen::VectorXd>& res) const {
      for (double x = min; x < max; x += inc) {
        p[dim] = x;
        if (dim == 0)
          res.push_back(p);
        else
          _enumerate(dim - 1, min, max, inc, p, res);
      }
    }

    std::vector<double> _scalarize_obs() {
      assert(this->_observations.size() != 0);

      Eigen::VectorXd lambda = Eigen::VectorXd::Random(this->_observations[0].size());
      lambda = (lambda.array() + 1.0) / 2.0;
      double sum = lambda.sum();
      lambda = lambda / sum;
      // scalarize (Tchebycheff)
      std::vector<double> scalarized;
      for (auto x : this->_observations) {
        double y = (lambda.array() * x.array()).maxCoeff();
        double s = (lambda.array() * x.array()).sum();
        scalarized.push_back(y + Params::parego::rho() * s);
      }
      return scalarized;
    }

  };







}

#endif
