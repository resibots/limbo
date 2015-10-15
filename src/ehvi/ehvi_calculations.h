// The exact EHVI calculation schemes except slice update.
#include "helper.h"
#include <deque>
using namespace std;

// When doing 2d hypervolume calculations, uncomment the following line
// to NOT use O(1) S-minus updates:
//#define NAIVE_DOMPOINTS

double ehvi2d(deque<individual*> P, double r[], double mu[], double s[]);

double ehvi3d_2term(deque<individual*> P, double r[], double mu[], double s[]);

double ehvi3d_5term(deque<individual*> P, double r[], double mu[], double s[]);

double ehvi3d_8term(deque<individual*> P, double r[], double mu[], double s[]);
