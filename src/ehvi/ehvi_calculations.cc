// The exact EHVI calculation schemes.
#include "helper.h"
#include "ehvi_hvol.h"
#include <deque>
#include <algorithm> //sorting
#include <math.h> //INFINITY macro
using namespace std;

// When doing 2d hypervolume calculations, uncomment the following line
// to NOT use O(1) S-minus updates:
//#define NAIVE_DOMPOINTS

// Returns the expected 2d hypervolume improvement of population P with
// reference
// point r, mean vector mu and standard deviation vector s.
double ehvi2d(deque<individual*> P, double r[], double mu[], double s[])
{
    sort(P.begin(), P.end(), xcomparator);
    double answer = 0; // The eventual answer
    int k = P.size(); // Holds amount of points.
#ifdef NAIVE_DOMPOINTS
    deque<individual*> dompoints; // For the old-fashioned way.
#endif
    double Sminus; // Correction term for the integral.
    int Sstart = k - 1, Shorizontal = 0;
    // See thesis for explanation of how the O(1) iteration complexity
    // is reached. NOTE: i = y = f[1], j = x = f[0]
    for (int i = 0; i <= k; i++) {
        Sminus = 0;
        Shorizontal = Sstart;
        for (int j = k - i; j <= k; j++) {
            double fmax[2]; // staircase width and height
            double cl1, cl2, cu1, cu2; // Boundaries of grid cells
            if (j == k) {
                fmax[1] = r[1];
                cu1 = INFINITY;
            }
            else {
                fmax[1] = P[j]->f[1];
                cu1 = P[j]->f[0];
            }
            if (i == k) {
                fmax[0] = r[0];
                cu2 = INFINITY;
            }
            else {
                fmax[0] = P[k - i - 1]->f[0];
                cu2 = P[k - i - 1]->f[1];
            }
            cl1 = (j == 0 ? r[0] : P[j - 1]->f[0]);
            cl2 = (i == 0 ? r[1] : P[k - i]->f[1]);
// Cell boundaries have been decided. Determine Sminus.
#ifdef NAIVE_DOMPOINTS
            dompoints.clear();
            for (int m = 0; m < k; m++) {
                if (cl1 >= P[m]->f[0] && cl2 >= P[m]->f[1]) {
                    dompoints.push_back(P[m]);
                }
            }
            Sminus = calculateS(dompoints, fmax);
#else
            if (Shorizontal > Sstart) {
                Sminus += (P[Shorizontal]->f[0] - fmax[0]) * (P[Shorizontal]->f[1] - fmax[1]);
            }
            Shorizontal++;
#endif
            // And then we integrate.
            double psi1 = exipsi(fmax[0], cl1, mu[0], s[0]) - exipsi(fmax[0], cu1, mu[0], s[0]);
            double psi2 = exipsi(fmax[1], cl2, mu[1], s[1]) - exipsi(fmax[1], cu2, mu[1], s[1]);
            double gausscdf1 = gausscdf((cu1 - mu[0]) / s[0]) - gausscdf((cl1 - mu[0]) / s[0]);
            double gausscdf2 = gausscdf((cu2 - mu[1]) / s[1]) - gausscdf((cl2 - mu[1]) / s[1]);
            double sum = (psi1 * psi2) - (Sminus * gausscdf1 * gausscdf2);
            if (sum > 0)
                answer += sum;
        }
        Sstart--;
    }
    return answer;
}

// Subtracts expected dominated dominated hypervolume from expected dominated
// hypervolume.
double ehvi3d_2term(deque<individual*> P, double r[], double mu[],
    double s[])
{
    double answer = 0; // The eventual answer.
    int n = P.size(); // Holds amount of points.
    double Sminus; // Correction term for the integral.
    deque<individual*> Py, Pz; // P sorted by y/z coordinate
    sort(P.begin(), P.end(), ycomparator);
    for (size_t i = 0; i < P.size(); i++) {
        Py.push_back(P[i]);
    }
    sort(P.begin(), P.end(), zcomparator);
    for (unsigned int i = 0; i < P.size(); i++) {
        Pz.push_back(P[i]);
    }
    sort(P.begin(), P.end(), xcomparator);
    for (int z = 0; z <= n; z++) {
        for (int y = 0; y <= n; y++) {
            for (int x = 0; x <= n; x++) {
                double fmax[3]; // upper corner of hypervolume improvement box
                double cl[3], cu[3]; // Boundaries of grid cells
                cl[0] = (x == 0 ? r[0] : P[x - 1]->f[0]);
                cl[1] = (y == 0 ? r[1] : Py[y - 1]->f[1]);
                cl[2] = (z == 0 ? r[2] : Pz[z - 1]->f[2]);
                cu[0] = (x == n ? INFINITY : P[x]->f[0]);
                cu[1] = (y == n ? INFINITY : Py[y]->f[1]);
                cu[2] = (z == n ? INFINITY : Pz[z]->f[2]);
                // Calculate expected one-dimensional improvements w.r.t. r
                double psi1 = exipsi(r[0], cl[0], mu[0], s[0]) - exipsi(r[0], cu[0], mu[0], s[0]);
                double psi2 = exipsi(r[1], cl[1], mu[1], s[1]) - exipsi(r[1], cu[1], mu[1], s[1]);
                double psi3 = exipsi(r[2], cl[2], mu[2], s[2]) - exipsi(r[2], cu[2], mu[2], s[2]);
                // Calculate the probability of being within the cell.
                double gausscdf1 = gausscdf((cu[0] - mu[0]) / s[0]) - gausscdf((cl[0] - mu[0]) / s[0]);
                double gausscdf2 = gausscdf((cu[1] - mu[1]) / s[1]) - gausscdf((cl[1] - mu[1]) / s[1]);
                double gausscdf3 = gausscdf((cu[2] - mu[2]) / s[2]) - gausscdf((cl[2] - mu[2]) / s[2]);
                // Calculate the 'expected position of p' and the correction term Sminus
                if (gausscdf1 == 0 || gausscdf2 == 0 || gausscdf3 == 0)
                    continue; // avoid division by 0, cell contribution is 0 in these
                // cases anyway.
                fmax[0] = (psi1 / gausscdf1) + r[0];
                fmax[1] = (psi2 / gausscdf2) + r[1];
                fmax[2] = (psi3 / gausscdf3) + r[2];
                Sminus = hvol3d(Pz, r, fmax);
                // the expected hypervolume improvement is the expected rectangular
                // volume
                // w.r.t. r minus the correction term Sminus
                double sum = (psi1 * psi2 * psi3) - (Sminus * gausscdf1 * gausscdf2 * gausscdf3);
                if (sum > 0) // Safety check; "Not-A-Number > 0" returns false
                    answer += sum;
            }
        }
    }
    return answer;
}

double ehvi3d_5term(deque<individual*> P, double r[], double mu[],
    double s[])
{
    // Extra-complicated 3-dimensional ehvi function. Subtracts 4 quantities off a
    // rectangular volume.
    double answer = 0; // The eventual answer
    int n = P.size(); // Holds amount of points.
    double Sminus; // Correction term for the integral.
    deque<individual*> Py, Pz; // P sorted by y/z coordinate
    sort(P.begin(), P.end(), ycomparator);
    for (size_t i = 0; i < P.size(); i++) {
        Py.push_back(P[i]);
    }
    sort(P.begin(), P.end(), zcomparator);
    for (unsigned int i = 0; i < P.size(); i++) {
        Pz.push_back(P[i]);
    }
    sort(P.begin(), P.end(), xcomparator);
    for (int z = 0; z <= n; z++) {
        for (int y = 0; y <= n; y++) {
            for (int x = 0; x <= n; x++) {
                double v[3]; // upper corner of hypervolume improvement box
                double cl[3], cu[3]; // Boundaries of grid cells
                cl[0] = (x == 0 ? r[0] : P[x - 1]->f[0]);
                cl[1] = (y == 0 ? r[1] : Py[y - 1]->f[1]);
                cl[2] = (z == 0 ? r[2] : Pz[z - 1]->f[2]);
                cu[0] = (x == n ? INFINITY : P[x]->f[0]);
                cu[1] = (y == n ? INFINITY : Py[y]->f[1]);
                cu[2] = (z == n ? INFINITY : Pz[z]->f[2]);
                // We have to find v. This naive implementation is O(n) per iteration.
                v[0] = r[0];
                v[1] = r[1];
                v[2] = r[2];
                bool dominated = false;
                for (unsigned int i = 0; i < P.size(); i++) {
                    if (P[i]->f[0] >= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] >= cu[2]) {
                        dominated = true;
                        break;
                    }
                    else if (P[i]->f[0] <= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] >= cu[2]) {
                        if (P[i]->f[0] > v[0])
                            v[0] = P[i]->f[0];
                    }
                    else if (P[i]->f[0] >= cu[0] && P[i]->f[1] <= cu[1] && P[i]->f[2] >= cu[2]) {
                        if (P[i]->f[1] > v[1])
                            v[1] = P[i]->f[1];
                    }
                    else if (P[i]->f[0] >= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] <= cu[2]) {
                        if (P[i]->f[2] > v[2])
                            v[2] = P[i]->f[2];
                    }
                }
                if (dominated)
                    continue; // Cell's contribution is 0.
                Sminus = hvol3d(Pz, v, cl);
                // And then we integrate.
                double psi1 = exipsi(v[0], cl[0], mu[0], s[0]) - exipsi(v[0], cu[0], mu[0], s[0]);
                double psi2 = exipsi(v[1], cl[1], mu[1], s[1]) - exipsi(v[1], cu[1], mu[1], s[1]);
                double psi3 = exipsi(v[2], cl[2], mu[2], s[2]) - exipsi(v[2], cu[2], mu[2], s[2]);

                double gausscdf1 = gausscdf((cu[0] - mu[0]) / s[0]) - gausscdf((cl[0] - mu[0]) / s[0]);
                double gausscdf2 = gausscdf((cu[1] - mu[1]) / s[1]) - gausscdf((cl[1] - mu[1]) / s[1]);
                double gausscdf3 = gausscdf((cu[2] - mu[2]) / s[2]) - gausscdf((cl[2] - mu[2]) / s[2]);
                double sum = (psi1 * psi2 * psi3) - (Sminus * gausscdf1 * gausscdf2 * gausscdf3);
                // gausscdf represents chance of a point falling within the range
                // [cl,cu)
                // psi = partial expected improvement
                // so psi - (gausscdf * (cl - v)) = p's expected distance from cl
                double xslice = calculateslice(P, v, cl, 0);
                double yslice = calculateslice(Py, v, cl, 1);
                double zslice = calculateslice(Pz, v, cl, 2);
                //    cout << "Current partial contribution: " << sum << endl;
                sum -= (xslice * gausscdf2 * gausscdf3 * (psi1 - (gausscdf1 * (cl[0] - v[0]))));
                sum -= (yslice * gausscdf1 * gausscdf3 * (psi2 - (gausscdf2 * (cl[1] - v[1]))));
                sum -= (zslice * gausscdf1 * gausscdf2 * (psi3 - (gausscdf3 * (cl[2] - v[2]))));
                // cout << "Calculated partial contribution: " << sum << endl;
                if (sum > 0)
                    answer += sum;
            }
        }
    }
    return answer;
}

double ehvi3d_8term(deque<individual*> P, double r[], double mu[],
    double s[])
{
    // Implementation of the fully decomposed 3d EHVI calculation. Adds 8
    // quantities. NOT DONE YET.
    double answer = 0; // The eventual answer
    double sum; // Partial answer-holding variable.
    int n = P.size(); // Holds amount of points.
    double tempcorr, temprect,
        tempimp; // Correction term, rectangular volume, temp. improvement
    deque<individual*> Py, Pz; // P sorted by y/z coordinate
    sort(P.begin(), P.end(), ycomparator);
    for (size_t i = 0; i < P.size(); i++) {
        Py.push_back(P[i]);
    }
    sort(P.begin(), P.end(), zcomparator);
    for (unsigned int i = 0; i < P.size(); i++) {
        Pz.push_back(P[i]);
    }
    sort(P.begin(), P.end(), xcomparator);
    for (int z = 0; z <= n; z++) {
        for (int y = 0; y <= n; y++) {
            for (int x = 0; x <= n; x++) {
                double v[3]; // upper corner of hypervolume improvement box
                double cl[3], cu[3]; // Boundaries of grid cell
                cl[0] = (x == 0 ? r[0] : P[x - 1]->f[0]);
                cl[1] = (y == 0 ? r[1] : Py[y - 1]->f[1]);
                cl[2] = (z == 0 ? r[2] : Pz[z - 1]->f[2]);
                cu[0] = (x == n ? INFINITY : P[x]->f[0]);
                cu[1] = (y == n ? INFINITY : Py[y]->f[1]);
                cu[2] = (z == n ? INFINITY : Pz[z]->f[2]);
                // We have to find v. This naive implementation is O(n) per iteration.
                v[0] = r[0];
                v[1] = r[1];
                v[2] = r[2];
                bool dominated = false;
                for (unsigned int i = 0; i < P.size(); i++) {
                    if (P[i]->f[0] >= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] >= cu[2]) {
                        dominated = true;
                        break;
                    }
                    else if (P[i]->f[0] <= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] >= cu[2]) {
                        if (P[i]->f[0] > v[0])
                            v[0] = P[i]->f[0];
                    }
                    else if (P[i]->f[0] >= cu[0] && P[i]->f[1] <= cu[1] && P[i]->f[2] >= cu[2]) {
                        if (P[i]->f[1] > v[1])
                            v[1] = P[i]->f[1];
                    }
                    else if (P[i]->f[0] >= cu[0] && P[i]->f[1] >= cu[1] && P[i]->f[2] <= cu[2]) {
                        if (P[i]->f[2] > v[2])
                            v[2] = P[i]->f[2];
                    }
                }
                if (dominated)
                    continue; // Cell's contribution is 0.

                // We will first calculate the chance of being in the cell:
                double gausscdf1 = gausscdf((cu[0] - mu[0]) / s[0]) - gausscdf((cl[0] - mu[0]) / s[0]);
                double gausscdf2 = gausscdf((cu[1] - mu[1]) / s[1]) - gausscdf((cl[1] - mu[1]) / s[1]);
                double gausscdf3 = gausscdf((cu[2] - mu[2]) / s[2]) - gausscdf((cl[2] - mu[2]) / s[2]);

                // And the expected improvement with regards to cl:
                double psi1 = exipsi(cl[0], cl[0], mu[0], s[0]) - exipsi(cl[0], cu[0], mu[0], s[0]);
                double psi2 = exipsi(cl[1], cl[1], mu[1], s[1]) - exipsi(cl[1], cu[1], mu[1], s[1]);
                double psi3 = exipsi(cl[2], cl[2], mu[2], s[2]) - exipsi(cl[2], cu[2], mu[2], s[2]);

                // FIRST QUANTITY: Q_emptyset.
                tempcorr = hvol3d(Pz, r, cl);
                temprect = (cl[0] - r[0]) * (cl[1] - r[1]) * (cl[2] - r[2]);
                sum = gausscdf1 * gausscdf2 * gausscdf3 * (temprect - tempcorr);
                // SECOND QUANTITY: Q_x
                tempcorr = calculateslice(P, r, cl, 0);
                temprect = (cl[1] - r[1]) * (cl[2] - r[2]);
                tempimp = (temprect - tempcorr) * psi1;
                sum += tempimp * gausscdf2 * gausscdf3;

                // THIRD QUANTITY: Q_y
                tempcorr = calculateslice(Py, r, cl, 1);
                temprect = (cl[0] - r[0]) * (cl[2] - r[2]);
                tempimp = (temprect - tempcorr) * psi2;
                sum += tempimp * gausscdf1 * gausscdf3;

                // FOURTH QUANTITY: Q_z
                tempcorr = calculateslice(Pz, r, cl, 2);
                temprect = (cl[0] - r[0]) * (cl[1] - r[1]);
                tempimp = (temprect - tempcorr) * psi3;
                sum += tempimp * gausscdf1 * gausscdf2;

                // FIFTH QUANTITY: Q_xy
                tempimp = (cl[2] - v[2]) * gausscdf3;
                sum += psi1 * psi2 * tempimp;

                // SIXTH QUANTITY: Q_xz
                tempimp = (cl[1] - v[1]) * gausscdf2;
                sum += psi1 * psi3 * tempimp;

                // SEVENTH QUANTITY: Q_yz
                tempimp = (cl[0] - v[0]) * gausscdf1;
                sum += psi2 * psi3 * tempimp;

                // EIGHTH QUANTITY: Q_xyz
                sum += psi1 * psi2 * psi3;

                if (sum > 0)
                    answer += sum;
            }
        }
    }
    return answer;
}
