#ifndef BENCHMARKS_REGRESSION_TEST_FUNCTIONS_HPP
#define BENCHMARKS_REGRESSION_TEST_FUNCTIONS_HPP

struct Rastrigin {
    double operator()(const Eigen::VectorXd& x) const
    {
        double f = 10 * x.size();
        for (int i = 0; i < x.size(); ++i)
            f += x(i) * x(i) - 10 * cos(2 * M_PI * x(i));
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

struct SixHumpCamel {
    double operator()(const Eigen::VectorXd& x) const
    {
        double x1 = x(0);
        double x2 = x(1);
        double x1_2 = x1 * x1;
        double x2_2 = x2 * x2;

        double tmp1 = (4 - 2.1 * x1_2 + (x1_2 * x1_2) / 3) * x1_2;
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

#endif
