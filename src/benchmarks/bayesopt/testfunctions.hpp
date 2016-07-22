/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for
   Bayesian optimization.

   Copyright (C) 2011-2013 Ruben Martinez-Cantin <rmcantin@unizar.es>

   BayesOpt is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <bayesopt/bayesopt.hpp>

using namespace bayesopt;

// support functions
inline double sign(double x)
{
    if (x < 0)
        return -1;
    if (x > 0)
        return 1;
    return 0;
}

inline double sqr(double x)
{
    return x * x;
};

inline double hat(double x)
{
    if (x != 0)
        return log(fabs(x));
    return 0;
}

inline double c1(double x)
{
    if (x > 0)
        return 10;
    return 5.5;
}

inline double c2(double x)
{
    if (x > 0)
        return 7.9;
    return 3.1;
}

inline vectord t_osz(const vectord& x)
{
    vectord r = x;
    for (int i = 0; i < x.size(); i++)
        r(i) = sign(x(i)) * exp(hat(x(i)) + 0.049 * sin(c1(x(i)) * hat(x(i))) + sin(c2(x(i)) * hat(x(i))));
    return r;
}

struct Sphere {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;

    double operator()(const vectord& x) const
    {
        vectord opt(2);
        opt <<= 0.5, 0.5;

        return sqr(norm_2(x - opt));
    }

    matrixd solutions() const
    {
        matrixd sols(1, 2);
        sols <<= 0.5, 0.5;
        return sols;
    }
};

struct Ellipsoid {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;

    double operator()(const vectord& x) const
    {
        vectord opt(2);
        opt <<= 0.5, 0.5;
        vectord z = t_osz(x - opt);
        double r = 0;
        for (size_t i = 0; i < dim_in; ++i)
            r += std::pow(10, ((double)i) / (dim_in - 1.0)) * z(i) * z(i) + 1;
        return r;
    }

    matrixd solutions() const
    {
        matrixd sols(1, 2);
        sols <<= 0.5, 0.5;
        return sols;
    }
};

struct Rastrigin {
    static constexpr size_t dim_in = 4;
    static constexpr size_t dim_out = 1;

    double operator()(const vectord& x) const
    {
        double f = 10 * dim_in;
        for (size_t i = 0; i < dim_in; ++i)
            f += x(i) * x(i) - 10 * cos(2 * M_PI * x(i));
        return f;
    }

    matrixd solutions() const
    {
        matrixd sols(1, dim_in);
        for (size_t i = 0; i < dim_in; ++i)
            sols(0, i) = 0;
        return sols;
    }
};

// see : http://www.sfu.ca/~ssurjano/hart3.html
struct Hartmann3 {
    static constexpr size_t dim_in = 3;
    static constexpr size_t dim_out = 1;

    double operator()(const vectord& x) const
    {
        matrixd a(4, 3);
        matrixd p(4, 3);
        a <<= 3.0, 10, 30, 0.1, 10, 35, 3.0, 10, 30, 0.1, 10, 36;
        p <<= 0.3689, 0.1170, 0.2673, 0.4699, 0.4387, 0.7470, 0.1091, 0.8732, 0.5547,
            0.0382, 0.5743, 0.8828;
        vectord alpha(4);
        alpha <<= 1.0, 1.2, 3.0, 3.2;

        double res = 0;
        for (int i = 0; i < 4; i++) {
            double s = 0.0f;
            for (size_t j = 0; j < 3; j++) {
                s += a(i, j) * (x(j) - p(i, j)) * (x(j) - p(i, j));
            }
            res += alpha(i) * exp(-s);
        }
        return -res;
    }

    matrixd solutions() const
    {
        matrixd sols(1, 3);
        sols <<= 0.114614, 0.555649, 0.852547;
        return sols;
    }
};

// see : http://www.sfu.ca/~ssurjano/hart6.html
struct Hartmann6 {
    static constexpr size_t dim_in = 6;
    static constexpr size_t dim_out = 1;

    double operator()(const vectord& x) const
    {
        matrixd a(4, 6);
        matrixd p(4, 6);
        a <<= 10, 3, 17, 3.5, 1.7, 8, 0.05, 10, 17, 0.1, 8, 14, 3, 3.5, 1.7, 10, 17,
            8, 17, 8, 0.05, 10, 0.1, 14;
        p <<= 0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886, 0.2329, 0.4135, 0.8307,
            0.3736, 0.1004, 0.9991, 0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665,
            0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381;

        vectord alpha(4);
        alpha <<= 1.0, 1.2, 3.0, 3.2;

        double res = 0;
        for (int i = 0; i < 4; i++) {
            double s = 0.0f;
            for (size_t j = 0; j < 6; j++) {
                s += a(i, j) * sqr(x(j) - p(i, j));
            }
            res += alpha(i) * exp(-s);
        }
        return -res;
    }

    matrixd solutions() const
    {
        matrixd sols(1, 6);
        sols <<= 0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573;
        return sols;
    }
};

// see : http://www.sfu.ca/~ssurjano/goldpr.html
// (with ln, as suggested in Jones et al.)
struct GoldsteinPrice {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;

    double operator()(const vectord& xx) const
    {
        vectord x = (4.0 * xx);
        x(0) -= 2.0;
        x(1) -= 2.0;
        double r = (1 + (x(0) + x(1) + 1) * (x(0) + x(1) + 1) * (19 - 14 * x(0) + 3 * x(0) * x(0) - 14 * x(1) + 6 * x(0) * x(1) + 3 * x(1) * x(1))) * (30 + (2 * x(0) - 3 * x(1)) * (2 * x(0) - 3 * x(1)) * (18 - 32 * x(0) + 12 * x(0) * x(0) + 48 * x(1) - 36 * x(0) * x(1) + 27 * x(1) * x(1)));

        return log(r) - 5;
    }

    matrixd solutions() const
    {
        matrixd sols(1, 2);
        sols <<= 0.5, 0.25;
        return sols;
    }
};

struct BraninNormalized {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;

    double operator()(const vectord& x) const
    {
        double a = x(0) * 15 - 5;
        double b = x(1) * 15;
        return sqr(b - (5.1 / (4 * sqr(M_PI))) * sqr(a) + 5 * a / M_PI - 6) + 10 * (1 - 1 / (8 * M_PI)) * cos(a) + 10;
    }

    matrixd solutions() const
    {
        matrixd sols(3, 2);
        sols <<= 0.1238938, 0.818333,
            0.5427728, 0.151667,
            0.961652, 0.1650;
        return sols;
    }
};

struct SixHumpCamel {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;
    double operator()(const vectord& x) const
    {
        double x1 = -3 + 6 * x(0);
        double x2 = -2 + 4 * x(1);
        double x1_2 = x1 * x1;
        double x2_2 = x2 * x2;

        double tmp1 = (4 - 2.1 * x1_2 + (x1_2 * x1_2) / 3) * x1_2;
        double tmp2 = x1 * x2;
        double tmp3 = (-4 + 4 * x2_2) * x2_2;
        return tmp1 + tmp2 + tmp3;
    }

    matrixd solutions() const
    {
        matrixd sols(2, 2);
        sols <<= 0.0898, -0.7126,
            -0.0898, 0.7126;
        sols(0, 0) = (sols(0, 0) + 3.0) / 6.0;
        sols(1, 0) = (sols(1, 0) + 3.0) / 6.0;
        sols(0, 1) = (sols(0, 1) + 2.0) / 4.0;
        sols(1, 1) = (sols(1, 1) + 2.0) / 4.0;
        return sols;
    }
};

template <typename Function>
class Benchmark : public bayesopt::ContinuousModel {
public:
    Benchmark(bopt_params par) : ContinuousModel(Function::dim_in, par) {}

    double evaluateSample(const vectord& xin)
    {
        return f(xin);
    }

    bool checkReachability(const vectord& query)
    {
        return true;
    };

    double accuracy(double x)
    {
        matrixd sols = f.solutions();
        double diff = std::abs(x - f(row(sols, 0)));
        double min_diff = diff;

        for (size_t i = 1; i < sols.size1(); i++) {
            diff = std::abs(x - f(row(sols, i)));
            if (diff < min_diff)
                min_diff = diff;
        }

        return min_diff;
    }

    Function f;
};
