#ifndef EHVI_HELPER_H
#define EHVI_HELPER_H
#include "ehvi_consts.h"
// Header of the helper functions used for expected hypervolume calculations.
// Also contains the individual struct.

// Individual struct. Holds one individual.
struct individual {
    double f[DIMENSIONS];
};

// This struct holds the heightmaps and an array of dominated hypervolumes
// used in the cell calculations for the current z value.
// To update chunk, all values of slice are multiplied by height
// and added to it. Updating slice uses some geometry, see paper.
struct thingy {
    double slice; // area covered by z-slice
    double chunk; // S^-
    int highestdominator; // highest z coordinate of point dominating the point.
    double xlim,
        ylim; // 1-dimensional limits (zlim is calculated from highestdominator)
};

// for convenient calculation of height maps.
struct specialind {
    individual* point; // the individual's basic stats.
    int xorder, yorder,
        zorder; // Position of the individual in its respective sorting lists
};

// compares specialind in sort function.
bool specialycomparator(specialind* A, specialind* B);
bool specialzcomparator(specialind* A, specialind* B);

// Probability density function for the normal distribution.
double gausspdf(double x);

// Cumulative distribution function for the normal distribution
double gausscdf(double x);

// Partial expected improvement function 'psi'.
double exipsi(double fmax, double cellcorner, double mu, double s);

// Comparator function for sorting the inviduals in ascending order of x
// coordinate.
bool xcomparator(individual* A, individual* B);

// Comparator function for sorting the inviduals in ascending order of y
// coordinate.
bool ycomparator(individual* A, individual* B);

// Comparator function for sorting the inviduals in ascending order of z
// coordinate.
bool zcomparator(individual* A, individual* B);
#endif
