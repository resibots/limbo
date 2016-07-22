#ifndef HYPERVOL_H__
#define HYPERVOL_H__
extern "C" {
int FilterNondominatedSet(double* front[], int noPoints, int noObjectives);
double CalculateHypervolume(double* front[], int noPoints, int noObjectives);
}
#endif
