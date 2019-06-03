//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
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
#ifndef BENCHMARKS_REGRESSION_TEST_FUNCTIONS_HPP
#define BENCHMARKS_REGRESSION_TEST_FUNCTIONS_HPP

// Rastrigin Function: https://www.sfu.ca/~ssurjano/rastr.html
struct Rastrigin {
    double operator()(const Eigen::VectorXd& x) const
    {
        double f = 10 * x.size();
        for (int i = 0; i < x.size(); ++i)
            f += x(i) * x(i) - 10 * std::cos(2 * M_PI * x(i));
        return f;
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -5.12, 5.12;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return -1;
    }
};

// Ackley Function: https://www.sfu.ca/~ssurjano/ackley.html
struct Ackley {
    double operator()(const Eigen::VectorXd& x) const
    {
        double a = 20, b = 0.2, c = 2 * M_PI;
        double A = 0.0, B = 0.0;
        for (int i = 0; i < x.size(); i++) {
            A += x(i) * x(i);
            B += std::cos(c * x(i));
        }

        A = -b * std::sqrt(A / double(x.size()));
        B = B / double(x.size());

        return -a * std::exp(A) - std::exp(B) + a + std::exp(1);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -32.768, 32.768;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return -1;
    }
};

// Bukin N.6 Function: https://www.sfu.ca/~ssurjano/bukin6.html
struct Bukin {
    double operator()(const Eigen::VectorXd& x) const
    {
        return 100.0 * std::sqrt(std::abs(x(1) - 0.01 * x(0) * x(0))) + 0.01 * std::abs(x(0) + 10);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -15, 5;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        tmp << -3, 3;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 2;
    }
};

// Cross-In-Tray Function: https://www.sfu.ca/~ssurjano/crossit.html
struct CrossInTray {
    double operator()(const Eigen::VectorXd& x) const
    {
        double A = std::sin(x(0)) * std::sin(x(1));
        double B = std::abs(100 - x.norm() / M_PI);

        return -0.0001 * std::pow((std::abs(A * std::exp(B)) + 1), 0.1);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -10, 10;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 2;
    }
};

// Drop-Wave Function: https://www.sfu.ca/~ssurjano/drop.html
struct DropWave {
    double operator()(const Eigen::VectorXd& x) const
    {
        return -(1.0 + std::cos(12 * x.norm())) / (0.5 * x.squaredNorm() + 2.0);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -5.12, 5.12;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 2;
    }
};

// Gramacy & Lee 2012 Function: https://www.sfu.ca/~ssurjano/grlee12.html
struct GramacyLee {
    double operator()(const Eigen::VectorXd& x) const
    {
        return std::sin(10 * M_PI * x(0)) / (2.0 * x(0)) + std::pow((x(0) - 1.0), 4.0);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << 0.5, 2.5;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 1;
    }
};

// A simple step function
struct Step {
    double operator()(const Eigen::VectorXd& x) const
    {
        if (x(0) <= 0.0)
            return 0.0;
        else
            return 1.0;
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -2.0, 2.0;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 1;
    }
};

// Holder-Table Function: https://www.sfu.ca/~ssurjano/holder.html
struct HolderTable {
    double operator()(const Eigen::VectorXd& x) const
    {
        double A = std::sin(x(0)) * std::cos(x(1));
        double B = std::abs(1.0 - x.norm() / M_PI);

        return -std::abs(A * std::exp(B));
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -10, 10;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 2;
    }
};

// Levy Function: https://www.sfu.ca/~ssurjano/levy.html
struct Levy {
    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd w = 1.0 + (x.array() - 1.0) / 4.0;

        double A = std::sin(M_PI * w(0)) * std::sin(M_PI * w(0));
        double B = 0.0;
        for (int i = 0; i < x.size() - 1; i++) {
            double tmp = 1.0 + 10.0 * std::sin(M_PI * w(i) + 1) * std::sin(M_PI * w(i) + 1);
            B += (w(i) - 1.0) * (w(i) - 1.0) * tmp;
        }

        double C = (w(x.size() - 1) - 1.0) * (w(x.size() - 1) - 1.0) * (1.0 + std::sin(2 * M_PI * w(x.size() - 1)) * std::sin(2 * M_PI * w(x.size() - 1)));

        return A + B + C;
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -10, 10;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return -1;
    }
};

// Schwefel Function: https://www.sfu.ca/~ssurjano/schwef.html
struct Schwefel {
    double operator()(const Eigen::VectorXd& x) const
    {
        double A = 418.9829 * x.size();
        double B = 0.0;
        for (int i = 0; i < x.size(); i++) {
            B += x(i) * std::sin(std::sqrt(std::abs(x(i))));
        }

        return A - B;
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -500, 500;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return -1;
    }
};

// Six-Hump-Camel Function: https://www.sfu.ca/~ssurjano/camel6.html
struct SixHumpCamel {
    double operator()(const Eigen::VectorXd& x) const
    {
        double x1 = x(0);
        double x2 = x(1);
        double x1_2 = x1 * x1;
        double x2_2 = x2 * x2;

        double tmp1 = (4 - 2.1 * x1_2 + (x1_2 * x1_2) / 3.) * x1_2;
        double tmp2 = x1 * x2;
        double tmp3 = (-4 + 4 * x2_2) * x2_2;
        return tmp1 + tmp2 + tmp3;
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << -3, 3;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        tmp << -2, 2;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 2;
    }
};

// Hartmann6 Function: https://www.sfu.ca/~ssurjano/hart6.html
struct Hartmann6 {
    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::MatrixXd a(4, 6);
        Eigen::MatrixXd p(4, 6);
        a << 10, 3, 17, 3.5, 1.7, 8, 0.05, 10, 17, 0.1, 8, 14, 3, 3.5, 1.7, 10, 17,
            8, 17, 8, 0.05, 10, 0.1, 14;
        p << 0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886, 0.2329, 0.4135, 0.8307,
            0.3736, 0.1004, 0.9991, 0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665,
            0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381;

        Eigen::VectorXd alpha(4);
        alpha << 1.0, 1.2, 3.0, 3.2;

        double res = 0;
        for (int i = 0; i < 4; i++) {
            double s = 0.0f;
            for (size_t j = 0; j < 6; j++) {
                s += a(i, j) * (x(j) - p(i, j)) * (x(j) - p(i, j));
            }
            res += alpha(i) * exp(-s);
        }

        return res;
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << 0, 1;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 6;
    }
};

// Robot Arm Function: https://www.sfu.ca/~ssurjano/robot.html
struct RobotArm {
    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd q = x.head(4);
        Eigen::VectorXd L = x.tail(4);

        double u = 0.0, v = 0.0;

        for (int i = 0; i < 4; i++) {
            u += L(i) * std::cos(q.head(i + 1).sum());
            v += L(i) * std::sin(q.head(i + 1).sum());
        }

        return std::sqrt(v * v + u * u);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        Eigen::VectorXd tmp(2);
        tmp << 0., 2. * M_PI;
        std::vector<Eigen::VectorXd> b;
        b.push_back(tmp);
        b.push_back(tmp);
        b.push_back(tmp);
        b.push_back(tmp);
        tmp << 0., 1.;
        b.push_back(tmp);
        b.push_back(tmp);
        b.push_back(tmp);
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 8;
    }
};

// OTL Circuit Function: https://www.sfu.ca/~ssurjano/otlcircuit.html
struct OTLCircuit {
    double operator()(const Eigen::VectorXd& x) const
    {
        double Rb1 = x(0);
        double Rb2 = x(1);
        double Rf = x(2);
        double Rc1 = x(3);
        double Rc2 = x(4);
        double beta = x(5);

        double Vb1 = 12. * Rb2 / (Rb1 + Rb2);

        double term1 = (Vb1 + 0.74) * beta * (Rc2 + 9.) / (beta * (Rc2 + 9.) + Rf);
        double term2 = 11.35 * Rf / (beta * (Rc2 + 9.) + Rf);
        double term3 = 0.74 * Rf * beta * (Rc2 + 9.) / ((beta * (Rc2 + 9.) + Rf) * Rc1);

        return term1 + term2 + term3;
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        std::vector<Eigen::VectorXd> b;
        Eigen::VectorXd tmp(2);
        tmp << 50., 150.;
        b.push_back(tmp);
        tmp << 25., 70.;
        b.push_back(tmp);
        tmp << 0.5, 3.;
        b.push_back(tmp);
        tmp << 1.2, 2.5;
        b.push_back(tmp);
        tmp << 0.25, 1.2;
        b.push_back(tmp);
        tmp << 50., 300.;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 6;
    }
};

// Piston Simulation Function: https://www.sfu.ca/~ssurjano/piston.html
struct PistonSimulation {
    double operator()(const Eigen::VectorXd& x) const
    {
        double M = x(0);
        double S = x(1);
        double V_0 = x(2);
        double k = x(3);
        double P_0 = x(4);
        double T_a = x(5);
        double T_0 = x(6);

        double A = P_0 * S + 19.62 * M - k * V_0 / S;

        double V = S * (std::sqrt(A * A + 4. * k * P_0 * V_0 * T_a / T_0) - A) / (2. * k);

        return 2. * M_PI * std::sqrt(M / (k + S * S * P_0 * V_0 * T_a / (T_0 * V * V)));
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        std::vector<Eigen::VectorXd> b;
        Eigen::VectorXd tmp(2);
        tmp << 30., 60.;
        b.push_back(tmp);
        tmp << 0.005, 0.020;
        b.push_back(tmp);
        tmp << 0.002, 0.010;
        b.push_back(tmp);
        tmp << 1000., 5000.;
        b.push_back(tmp);
        tmp << 90000., 110000.;
        b.push_back(tmp);
        tmp << 290., 296.;
        b.push_back(tmp);
        tmp << 340., 360.;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 7;
    }
};

// Inverse Dynamics of a 2dof planar arm I (the first torque)
struct PlanarInverseDynamicsI {
    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd ddq = x.head(2);
        Eigen::VectorXd dq = x.segment(2, 2);
        Eigen::VectorXd q = x.segment(4, 2);

        // double m1 = x(6), l1 = x(7);
        double m1 = 0.5, l1 = 0.5;
        double r1 = l1 / 2.;
        // double m2 = x(8), l2 = x(9);
        double m2 = 0.5, l2 = 0.5;
        double r2 = l2 / 2.;
        double I1 = m1 * l1 * l1 / 12.;
        double I2 = m2 * l2 * l2 / 12.;

        double a = I1 + I2 + m1 * r1 * r1 + m2 * (l1 * l1 + r2 * r2);
        double b = m2 * l1 * r2;
        double delta = I2 + m2 * r2 * r2;

        Eigen::MatrixXd M(2, 2);
        M << a + 2. * b * std::cos(q(1)), delta + b * std::cos(q(1)),
            delta + b * std::cos(q(1)), delta;
        // std::cout << M << std::endl
        //   << std::endl;

        Eigen::MatrixXd C(2, 2);
        C << -b * std::sin(q(1)) * dq(1), -b * std::sin(q(1)) * (dq(0) + dq(1)),
            b * std::sin(q(1)) * dq(0), 0.;
        // std::cout << C << std::endl;

        Eigen::VectorXd tau = M * ddq + C * dq;

        return tau(0);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        std::vector<Eigen::VectorXd> b;
        for (int i = 0; i < 4; i++) {
            Eigen::VectorXd tmp(2);
            tmp << -2. * M_PI, 2. * M_PI;
            b.push_back(tmp);
        }
        for (int i = 0; i < 2; i++) {
            Eigen::VectorXd tmp(2);
            tmp << -M_PI, M_PI;
            b.push_back(tmp);
        }
        // for (int i = 0; i < 4; i++) {
        //     Eigen::VectorXd tmp(2);
        //     tmp << 0.1, 1.;
        //     b.push_back(tmp);
        // }
        return b;
    }

    int dims() const
    {
        return 6;
    }
};

// Inverse Dynamics of a 2dof planar arm II (the first torque)
struct PlanarInverseDynamicsII {
    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd ddq = x.head(2);
        Eigen::VectorXd dq = x.segment(2, 2);
        Eigen::VectorXd q = x.segment(4, 2);

        // double m1 = x(6), l1 = x(7);
        double m1 = 0.5, l1 = 0.5;
        double r1 = l1 / 2.;
        // double m2 = x(8), l2 = x(9);
        double m2 = 0.5, l2 = 0.5;
        double r2 = l2 / 2.;
        double I1 = m1 * l1 * l1 / 12.;
        double I2 = m2 * l2 * l2 / 12.;

        double a = I1 + I2 + m1 * r1 * r1 + m2 * (l1 * l1 + r2 * r2);
        double b = m2 * l1 * r2;
        double delta = I2 + m2 * r2 * r2;

        Eigen::MatrixXd M(2, 2);
        M << a + 2. * b * std::cos(q(1)), delta + b * std::cos(q(1)),
            delta + b * std::cos(q(1)), delta;
        // std::cout << M << std::endl
        //   << std::endl;

        Eigen::MatrixXd C(2, 2);
        C << -b * std::sin(q(1)) * dq(1), -b * std::sin(q(1)) * (dq(0) + dq(1)),
            b * std::sin(q(1)) * dq(0), 0.;
        // std::cout << C << std::endl;

        Eigen::VectorXd tau = M * ddq + C * dq;

        return tau(1);
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        std::vector<Eigen::VectorXd> b;
        for (int i = 0; i < 4; i++) {
            Eigen::VectorXd tmp(2);
            tmp << -2. * M_PI, 2. * M_PI;
            b.push_back(tmp);
        }
        for (int i = 0; i < 2; i++) {
            Eigen::VectorXd tmp(2);
            tmp << -M_PI, M_PI;
            b.push_back(tmp);
        }
        // for (int i = 0; i < 4; i++) {
        //     Eigen::VectorXd tmp(2);
        //     tmp << 0.1, 1.;
        //     b.push_back(tmp);
        // }
        return b;
    }

    int dims() const
    {
        return 6;
    }
};

#endif
