#ifndef EHVI_CONSTS_H
#define EHVI_CONSTS_H

const int DIMENSIONS = 3; // Amount of function vectors.
const long long MONTECARLOS = 100000LL; // Amount of Monte Carlo simulations to
// do. Currently 100 thousand.
const int SEED = 12345678; // Used to seed the random number engine.

// Some pre-calculated constants to speed up calculations
const double SQRT_TWO = 1.41421356237; // sqrt(2)
const double SQRT_TWOPI_NEG = 0.3989422804; // 1/sqrt(2*PI)
const double HALFPI = 1.57079632679; // pi/2
#endif
