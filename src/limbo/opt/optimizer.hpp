#ifndef LIMBO_OPT_OPTIMIZER_HPP
#define LIMBO_OPT_OPTIMIZER_HPP

#include <tuple>

#include <Eigen/Core>

#include <boost/optional.hpp>

namespace limbo {
    ///\defgroup opt_tools
    namespace opt {

        ///@ingroup opt_tools
        /// return type of the function to optimize
        typedef std::pair<double, boost::optional<Eigen::VectorXd>> eval_t;

        ///@ingroup opt_tools
        ///return with opt::no_grad(your_val) if no gradient is available (to be used in functions to be optimized)
        eval_t no_grad(double x) { return eval_t{x, boost::optional<Eigen::VectorXd>{}}; }

        ///@ingroup opt_tools
        /// get the gradient from a function evaluation (eval_t)
        const Eigen::VectorXd& grad(const eval_t& fg)
        {
            assert(std::get<1>(fg).is_initialized());
            return std::get<1>(fg).get();
        }

        ///@ingroup opt_tools
        /// get the value from a function evaluation (eval_t)
        double fun(const eval_t& fg)
        {
            return std::get<0>(fg);
        }

        ///@ingroup opt_tools
        /// Evaluate f without gradient (to be called from the optimization algorithms that do not use the gradient)
        template <typename F>
        double eval(const F& f, const Eigen::VectorXd& x)
        {
            return std::get<0>(f(x, false));
        }

        ///@ingroup opt_tools
        /// Evaluate f with gradient (to be called from the optimization algorithms that use the gradient)
        template <typename F>
        eval_t eval_grad(const F& f, const Eigen::VectorXd& x)
        {
            return f(x, true);
        }
    }
}

#endif
