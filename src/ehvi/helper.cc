#include "helper.h"
#include "ehvi_consts.h"
#include <math.h>

// Comparison functions for sorting the specialind struct in y and z dimensions.
bool specialycomparator(specialind* A, specialind* B)
{
    return A->point->f[1] < B->point->f[1];
}

bool specialzcomparator(specialind* A, specialind* B)
{
    return A->point->f[2] < B->point->f[2];
}

// Probability density function for the normal distribution.
double gausspdf(double x) { return SQRT_TWOPI_NEG * exp(-(x * x) / 2); }

// Cumulative distribution function for the normal distribution
double gausscdf(double x) { return 0.5 * (1 + erf(x / SQRT_TWO)); }

// Partial expected improvement function 'psi'.
double exipsi(double fmax, double cellcorner, double mu, double s)
{
    return (s * gausspdf((cellcorner - mu) / s)) + ((fmax - mu) * gausscdf((cellcorner - mu) / s));
}

// Comparator function for sorting the inviduals in ascending order of x
// coordinate.
bool xcomparator(individual* A, individual* B) { return A->f[0] < B->f[0]; }

// Comparator function for sorting the inviduals in ascending order of y
// coordinate.
bool ycomparator(individual* A, individual* B) { return A->f[1] < B->f[1]; }

// Comparator function for sorting the inviduals in ascending order of z
// coordinate.
bool zcomparator(individual* A, individual* B) { return A->f[2] < B->f[2]; }
