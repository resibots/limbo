#ifndef BO_DEFAULT_PARAMS_HPP
#define BO_DEFAULT_PARAMS_HPP

namespace limbo {
  namespace tests {
    namespace def {
      struct cmaes {
        BO_PARAM(double, stopMaxFunEvals, 1e5);
        BO_PARAM(double, stopMaxIter, 1e5);
        BO_PARAM(double, stopTolFun, 1e-5);
        BO_PARAM(double, stopTolFunHist, 1e-6);
        BO_PARAM(int, nb_pts, 20);
        BO_PARAM(bool, discrete, false);
      };
    }
  }
}
#endif
