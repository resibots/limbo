// Include this if you want to calculate the EHVI of multiple individuals at the
// same
// time. This is more efficient than repeatedly calling an EHVI function on the
// same
// population.
#include "helper.h"
#include "ehvi_consts.h"
#include <vector>
using namespace std;

struct mus { // Holds mean & variance for a Gaussian distribution
    double mu[DIMENSIONS];
    double s[DIMENSIONS];
};

vector<double> ehvi3d_5term(deque<individual*> P, double r[],
    vector<mus*>& pdf);
vector<double> ehvi3d_sliceupdate(deque<individual*> P, double r[],
    vector<mus*>& pdf);
