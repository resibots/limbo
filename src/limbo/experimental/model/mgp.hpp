//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef LIMBO_MODEL_MGP_HPP
#define LIMBO_MODEL_MGP_HPP

#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <limbo/model/gp/no_lf_opt.hpp>
#include <limbo/tools.hpp>

#include <boost/fusion/include/at.hpp>

namespace limbo {
    namespace experimental {

        struct Init_GPs {
            Init_GPs(int dim_in, int dim_out)
                : _dim_in(dim_in), _dim_out(dim_out) {}

            template <typename Model>
            void operator()(Model& m) const { m = Model(_dim_in, _dim_out); }

            int _dim_in, _dim_out;
        };

        struct Compute {
            Compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations,
                const Eigen::VectorXd& noises, bool compute_kernel)
                : _samples(samples), _observations(observations), _noises(noises), _compute_kernel(compute_kernel) {}

            template <typename Model>
            void operator()(Model& m) const { m.compute(_samples, _observations, _noises, _compute_kernel); }

            const std::vector<Eigen::VectorXd>& _samples;
            const std::vector<Eigen::VectorXd>& _observations;
            const Eigen::VectorXd& _noises;
            bool _compute_kernel;
        };

        struct Recompute {
            Recompute(bool update_obs_mean) : _update_obs_mean(update_obs_mean) {}
            template <typename Model>

            void operator()(Model& m) const
            {
                m.recompute(_update_obs_mean);
            }

            const bool& _update_obs_mean;
        };

        struct Optimize_hyperparams {
            template <typename Model>
            void operator()(Model& m) const { m.optimize_hyperparams(); }
        };

        struct Add_sample {
            Add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation, double noise)
                : _sample(sample), _observation(observation), _noise(noise) {}

            template <typename Model>
            void operator()(Model& m) const { m.add_sample(_sample, _observation, _noise); }

            const Eigen::VectorXd& _sample;
            const Eigen::VectorXd& _observation;
            const double& _noise;
        };

        namespace model {
            template <typename Params, typename Models>
            class MGP {
            public:
                typedef typename boost::mpl::if_<boost::fusion::traits::is_sequence<Models>, Models, boost::fusion::vector<Models>>::type models_t;

            protected:
                int _dim_in;
                int _dim_out;

                models_t _models;

            public:
                /// useful because the model might be created before knowing anything about the process
                MGP() : _dim_in(-1), _dim_out(-1) {}

                /// useful because the model might be created  before having samples
                MGP(int dim_in, int dim_out)
                    : _dim_in(dim_in), _dim_out(dim_out)
                {
                    boost::fusion::for_each(_models, Init_GPs(_dim_in, _dim_out));
                }

                /// Compute the GP from samples, observation, noise. This call needs to be explicit!
                void compute(const std::vector<Eigen::VectorXd>& samples,
                    const std::vector<Eigen::VectorXd>& observations,
                    const Eigen::VectorXd& noises, bool compute_kernel = true)
                {
                    boost::fusion::for_each(_models, Compute(samples, observations, noises, compute_kernel));
                }

                /// Do not forget to call this if you use hyper-prameters optimization!!
                void optimize_hyperparams()
                {
                    boost::fusion::for_each(_models, Optimize_hyperparams());
                }

                /// add sample and update the GP. This code uses an incremental implementation of the Cholesky
                /// decomposition. It is therefore much faster than a call to compute()
                void add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation, double noise)
                {
                    boost::fusion::for_each(_models, Add_sample(sample, observation, noise));
                }

                template <size_t I = 0>
                std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
                {
                    return boost::fusion::at_c<I>(_models).query(v);
                }

                template <size_t I = 0>
                Eigen::VectorXd mu(const Eigen::VectorXd& v) const
                {
                    return boost::fusion::at_c<I>(_models).mu(v);
                }

                template <size_t I = 0>
                double sigma(const Eigen::VectorXd& v) const
                {
                    return boost::fusion::at_c<I>(_models).sigma(v);
                }

                /// return the number of dimensions of the input
                template <size_t I = 0>
                int dim_in() const
                {
                    return boost::fusion::at_c<I>(_models).dim_in();
                }

                /// return the number of dimensions of the output
                template <size_t I = 0>
                int dim_out() const
                {
                    return boost::fusion::at_c<I>(_models).dim_out();
                }

                size_t nb_gps() const { return boost::fusion::size(_models); }

                template <size_t I = 0>
                auto kernel_function() const -> decltype(boost::fusion::at_c<I>(_models).kernel_function())
                {
                    return boost::fusion::at_c<I>(_models).kernel_function();
                }

                template <size_t I = 0>
                auto mean_function() const -> decltype(boost::fusion::at_c<I>(_models).mean_function())
                {
                    return boost::fusion::at_c<I>(_models).mean_function();
                }

                /// return the maximum observation (only call this if the output of the GP is of dimension 1)
                template <size_t I = 0>
                Eigen::VectorXd max_observation() const
                {
                    return boost::fusion::at_c<I>(_models).max_observation();
                }

                /// return the mean observation (only call this if the output of the GP is of dimension 1)
                template <size_t I = 0>
                Eigen::VectorXd mean_observation() const
                {
                    return boost::fusion::at_c<I>(_models).mean_observation();
                }

                template <size_t I = 0>
                const Eigen::MatrixXd& mean_vector() const { return boost::fusion::at_c<I>(_models).mean_vector(); }

                template <size_t I = 0>
                const Eigen::MatrixXd& obs_mean() const { return boost::fusion::at_c<I>(_models).obs_mean(); }

                /// return the number of samples used to compute the GP
                template <size_t I = 0>
                int nb_samples() const { return boost::fusion::at_c<I>(_models).nb_samples(); }

                ///  recomputes the GP
                void recompute(bool update_obs_mean = true)
                {
                    boost::fusion::for_each(_models, Recompute(update_obs_mean));
                }

                /// return the likelihood (do not compute it!)
                template <size_t I = 0>
                double get_lik() const { return boost::fusion::at_c<I>(_models).get_lik(); }

                /// set the likelihood (you need to compute it from outside!)
                template <size_t I = 0>
                void set_lik(const double& lik) { boost::fusion::at_c<I>(_models).set_lik(lik); }

                /// LLT matrix (from Cholesky decomposition)
                // const Eigen::LLT<Eigen::MatrixXd>& llt() const { return _llt; }
                template <size_t I = 0>
                const Eigen::MatrixXd& matrixL() const { return boost::fusion::at_c<I>(_models).matrixL(); }

                template <size_t I = 0>
                const Eigen::MatrixXd& alpha() const { return boost::fusion::at_c<I>(_models).alpha(); }

                /// return the list of samples that have been tested so far
                template <size_t I = 0>
                const std::vector<Eigen::VectorXd>& samples() const { return boost::fusion::at_c<I>(_models).samples(); }
            };
        }
    }
}

#endif
