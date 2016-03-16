// Header to be included in order to use the hypervolume calculations
// implemented
// in ehvi_hvol.cc
#include "helper.h"
#include <deque>
using namespace std;

// Call this on a deque P of points SORTED IN ORDER OF ASCENDING Z COORDINATE
// to get the hypervolume of P contained in a box-shaped volume.
// fmax is the upper corner, cl is the lower corner of the box for which
// the hypervolume should be calculated. Points outside this box are not
// ignored,
// but only the hypervolume that fits in the box is considered.
// cl must match up with existing z coordinate for the function to return the
// correct answer.
double hvol3d(deque<individual*> P, double cl[DIMENSIONS],
    double fmax[DIMENSIONS]);

// Calculates the 2D slice between r and fmax, flattened in dimension
// 'dimension'.
double calculateslice(deque<individual*> P, double r[DIMENSIONS],
    double fmax[DIMENSIONS], int dimension);

// Returns the 2d hypervolume for the population P with reference
// point r.
double calculateS(deque<individual*> P, double r[]);
