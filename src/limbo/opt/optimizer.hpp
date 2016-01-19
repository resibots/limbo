#ifndef LIMBO_OPT_OPTIMIZER_HPP
#define LIMBO_OPT_OPTIMIZER_HPP

#include <tuple>

#include <Eigen/Core>

#include <boost/optional.hpp>

namespace limbo {
    namespace opt {
        // return type of the function to optimize
        typedef std::pair<double, boost::optional<Eigen::VectorXd>> eval_t;

        // return with opt::no_grand(your_val) if no gradient is available
        eval_t no_grad(double x) { return eval_t{x, boost::optional<Eigen::VectorXd>{}}; }

        // get the gradient from a function evaluation
        const Eigen::VectorXd& grad(const eval_t& fg)
        {
            assert(std::get<1>(fg).is_initialized());
            return std::get<1>(fg).get();
        }

        // get the value from a function evaluation
        double fun(const eval_t& fg)
        {
            return std::get<0>(fg);
        }

        // eval f without gradient
        template <typename F>
        double eval(const F& f, const Eigen::VectorXd& x)
        {
            return std::get<0>(f(x, false));
        }

        // eval f with gradient
        template <typename F>
        eval_t eval_grad(const F& f, const Eigen::VectorXd& x)
        {
            return f(x, true);
        }
    }
}

#endif
