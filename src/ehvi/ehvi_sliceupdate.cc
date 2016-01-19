#include "ehvi_consts.h"
#include <deque>
#include <algorithm>
#include <iostream> //For error on exception only.
#include <cmath> //INFINITY macro
#include "ehvi_hvol.h"

using namespace std;

#ifndef EHVI_SLICEUPDATE
#define EHVI_SLICEUPDATE

double ehvi3d_sliceupdate(deque<individual*> P, double r[], double mu[],
    double s[])
{
    // EHVI calculation algorithm with time complexity O(n^3).
    double answer = 0; // The eventual answer.
    specialind* newind;
    int n = P.size(); // Holds amount of points.
    thingy* Pstruct; // 2D array with information about the shape of the dominated
    // hypervolume
    deque<specialind*> Px, Py,
        Pz; // P sorted by x/y/z coordinate with extra information.
    double cellength[3] = {0};
    try {
        // Create sorted arrays which also contain extra information allowing the
        // location in
        // the other sorting orders to be ascertained in O(1).
        sort(P.begin(), P.end(), xcomparator);
        for (int i = 0; i < n; i++) {
            newind = new specialind;
            newind->point = P[i];
            newind->xorder = i;
            Px.push_back(newind);
            Py.push_back(newind);
            Pz.push_back(newind);
        }
        sort(Py.begin(), Py.end(), specialycomparator);
        for (int i = 0; i < n; i++) {
            Py[i]->yorder = i;
        }
        sort(Pz.begin(), Pz.end(), specialzcomparator);
        for (int i = 0; i < n; i++) {
            Pz[i]->zorder = i;
        }
        // Then also reserve memory for the structure array.
        Pstruct = new thingy[n * n];
        for (int k = 0; k < n * n; k++) {
            Pstruct[k].slice = 0;
            Pstruct[k].chunk = 0;
            Pstruct[k].highestdominator = -1;
            Pstruct[k].xlim = 0;
            Pstruct[k].ylim = 0;
        }
    }
    catch (...) {
        cout << "An exception was thrown. There probably isn't enough memory "
                "available." << endl;
        cout << "-1 will be returned." << endl;
        return -1;
    }
    // Now we establish dominance in the 2-dimensional slices. Note: it is assumed
    // that
    // P is mutually nondominated. This implementation of that step is O(n^3).
    for (int i = 0; i < n; i++) {
        for (int j = Pz[i]->yorder; j >= 0; j--)
            for (int k = Pz[i]->xorder; k >= 0; k--) {
                Pstruct[k + j * n].highestdominator = i;
            }
        for (int j = Px[i]->zorder; j >= 0; j--)
            for (int k = Px[i]->yorder; k >= 0; k--) {
                Pstruct[k + j * n].xlim = Px[i]->point->f[0] - r[0];
            }
        for (int j = Py[i]->zorder; j >= 0; j--)
            for (int k = Py[i]->xorder; k >= 0; k--) {
                Pstruct[k + j * n].ylim = Py[i]->point->f[1] - r[1];
            }
    }
    // And now for the actual EHVI calculations.
    for (int z = 0; z <= n; z++) {
        // Recalculate Pstruct for the next 2D slice.
        if (z > 0)
            for (int i = 0; i < n * n; i++) {
                Pstruct[i].chunk += Pstruct[i].slice * cellength[2];
            }
        // This step is O(n^2).
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                if (Pstruct[x + y * n].highestdominator < z) { // cell is not dominated

                    if (x > 0 && y > 0) {
                        Pstruct[x + y * n].slice = (Pstruct[x + (y - 1) * n].slice - Pstruct[(x - 1) + (y - 1) * n].slice) + Pstruct[(x - 1) + y * n].slice;
                    }
                    else if (y > 0) {
                        Pstruct[x + y * n].slice = Pstruct[x + (y - 1) * n].slice;
                    }
                    else if (x > 0) {
                        Pstruct[x + y * n].slice = Pstruct[(x - 1) + y * n].slice;
                    }
                    else
                        Pstruct[x + y * n].slice = 0;
                }
                else {
                    Pstruct[x + y * n].slice = (Px[x]->point->f[0] - r[0]) * (Py[y]->point->f[1] - r[1]);
                }
            }
        }
        // Okay, now we are going to calculate the EHVI, for real.
        for (int y = 0; y <= n; y++) {
            for (int x = 0; x <= n; x++) {
                double cl[3], cu[3]; // Boundaries of grid cells
                cl[0] = (x == 0 ? r[0] : Px[x - 1]->point->f[0]);
                cl[1] = (y == 0 ? r[1] : Py[y - 1]->point->f[1]);
                cl[2] = (z == 0 ? r[2] : Pz[z - 1]->point->f[2]);
                cu[0] = (x == n ? INFINITY : Px[x]->point->f[0]);
                cu[1] = (y == n ? INFINITY : Py[y]->point->f[1]);
                cu[2] = (z == n ? INFINITY : Pz[z]->point->f[2]);
                cellength[0] = cu[0] - cl[0];
                cellength[1] = cu[1] - cl[1];
                cellength[2] = cu[2] - cl[2];
                if (cellength[0] == 0 || cellength[1] == 0 || cellength[2] == 0 || (x < n && y < n && Pstruct[x + y * n].highestdominator >= z))
                    continue; // Cell is dominated or of size 0.
                // We have easy access to Sminus and zslice because they are part of
                // Pstruct.
                // xslice and yslice can be calculated from Pstruct->chunk.
                double slice[3], Sminus, v[3];
                if (x > 0 && y > 0) {
                    Sminus = Pstruct[(x - 1) + (y - 1) * n].chunk;
                    slice[0] = (x == n ? 0 : (Pstruct[x + (y - 1) * n].chunk - Sminus) / cellength[0]);
                    slice[1] = (y == n ? 0 : (Pstruct[(x - 1) + y * n].chunk - Sminus) / cellength[1]);
                    slice[2] = Pstruct[(x - 1) + (y - 1) * n].slice;
                }
                else {
                    Sminus = 0;
                    slice[0] = ((y == 0 || x == n)
                            ? 0
                            : (Pstruct[x + (y - 1) * n].chunk - Sminus) / cellength[0]);
                    slice[1] = ((x == 0 || y == n)
                            ? 0
                            : (Pstruct[(x - 1) + y * n].chunk - Sminus) / cellength[1]);
                    slice[2] = 0;
                }
                if (y == n || z == n)
                    v[0] = 0;
                else
                    v[0] = Pstruct[y + z * n].xlim;
                if (x == n || z == n)
                    v[1] = 0;
                else
                    v[1] = Pstruct[x + z * n].ylim;
                if (x == n || y == n)
                    v[2] = 0;
                else
                    v[2] = (Pstruct[x + y * n].highestdominator == -1
                            ? 0
                            : (Pz[Pstruct[x + y * n].highestdominator]->point->f[2] - r[2]));
                // All correction terms have been established. Calculate the cell's
                // contribution to the integral.
                double psi1 = exipsi(r[0], cl[0], mu[0], s[0]) - exipsi(r[0], cu[0], mu[0], s[0]);
                double psi2 = exipsi(r[1], cl[1], mu[1], s[1]) - exipsi(r[1], cu[1], mu[1], s[1]);
                double psi3 = exipsi(r[2], cl[2], mu[2], s[2]) - exipsi(r[2], cu[2], mu[2], s[2]);

                double gausscdf1 = gausscdf((cu[0] - mu[0]) / s[0]) - gausscdf((cl[0] - mu[0]) / s[0]);
                double gausscdf2 = gausscdf((cu[1] - mu[1]) / s[1]) - gausscdf((cl[1] - mu[1]) / s[1]);
                double gausscdf3 = gausscdf((cu[2] - mu[2]) / s[2]) - gausscdf((cl[2] - mu[2]) / s[2]);

                double ex1 = psi1 - (gausscdf1 * (cl[0] - r[0]));
                double ex2 = psi2 - (gausscdf2 * (cl[1] - r[1]));
                double ex3 = psi3 - (gausscdf3 * (cl[2] - r[2]));

                double sum = (psi1 * psi2 * psi3) - (Sminus * gausscdf1 * gausscdf2 * gausscdf3);
                // Subtract correction terms:
                sum -= (slice[0] * gausscdf2 * gausscdf3 * ex1);
                sum -= (slice[1] * gausscdf1 * gausscdf3 * ex2);
                sum -= (slice[2] * gausscdf1 * gausscdf2 * ex3);
                sum -= v[0] * ex2 * ex3 * gausscdf1;
                sum -= v[1] * ex1 * ex3 * gausscdf2;
                sum -= v[2] * ex1 * ex2 * gausscdf3;
                if (sum > 0)
                    answer += sum;
            }
        }
    }
    return answer;
}
#endif
