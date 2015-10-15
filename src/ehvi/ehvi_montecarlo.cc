#include "ehvi_hvol.h"
#include "ehvi_montecarlo.h"
#include <random>
#include <deque>
#include <algorithm>
#include <math.h>
#include "ehvi_consts.h"
#include <fstream>
using namespace std;

const double TWOPI = 6.28318530718;

// Box-Muller transform using a Mersenne Twister.
class RNG {
public:
    mt19937 twister;
    RNG()
    {
        saved = false;
        twister.seed(SEED);
    }

    double getnumber(double mu, double s)
    {
        if (saved) {
            saved = false;
            return (s * savednumber) + mu;
        }
        double U1 = (double)twister() / twister.max();
        double U2 = (double)twister() / twister.max();
        double R = sqrt(-2 * log(U1));
        double theta = TWOPI * U2;
        savednumber = R * cos(theta);
        saved = true;
        return (s * R * sin(theta)) + mu;
    }

private:
    bool saved;
    double savednumber;
} rng;

double ehvi3d_montecarlo(deque<individual*> P, double r[], double mu[],
    double s[])
{
    // Monte Carlo simulation. Gives an approximation of the EHVI by repeatedly
    // generating pseudorandom normally-distributed points and calculating their
    // hypervolume improvement.
    double answer = 0;
    individual candidate;
    sort(P.begin(), P.end(), zcomparator);
    for (int i = 1; i <= MONTECARLOS; i++) {
        // Generate new candidate individual. If it fails to dominate r, we discard
        // it.
        candidate.f[0] = rng.getnumber(mu[0], s[0]);
        if (candidate.f[0] <= r[0])
            continue;
        candidate.f[1] = rng.getnumber(mu[1], s[1]);
        if (candidate.f[1] <= r[1])
            continue;
        candidate.f[2] = rng.getnumber(mu[2], s[2]);
        if (candidate.f[2] <= r[2])
            continue;
        // Calculate its hypervolume improvement by subtracting already-covered
        // hypervolume
        // from the box covered by the candidate.
        double hvol = (candidate.f[0] - r[0]) * (candidate.f[1] - r[1]) * (candidate.f[2] - r[2]);
        double Sminus = hvol3d(P, r, candidate.f);
        hvol -= Sminus;
        if (hvol > 0)
            answer += hvol;
    }
    return answer / MONTECARLOS;
}
