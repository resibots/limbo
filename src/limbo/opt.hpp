#ifndef LIMBO_OPT_HPP
#define LIMBO_OPT_HPP

#include <limbo/opt/optimizer.hpp>
#ifdef USE_LIBCMAES
#include <limbo/opt/cmaes.hpp>
#endif
#include <limbo/opt/grid_search.hpp>
#ifdef USE_NLOPT
#include <limbo/opt/nlopt_grad.hpp>
#include <limbo/opt/nlopt_no_grad.hpp>
#endif
#include <limbo/opt/parallel_repeater.hpp>
#include <limbo/opt/random_point.hpp>
#include <limbo/opt/rprop.hpp>

#endif
