// Command-line application for calculating the 3-Dimensional
// Expected Hypervolume Improvement.
// By Iris Hupkens, 2013
#include <deque>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "ehvi_calculations.h"
#include "ehvi_sliceupdate.h"
#include "ehvi_montecarlo.h"
#include "string.h"
#include "ehvi_multi.h"
#include <vector>
using namespace std;

// Performs the EHVI calculations using the scheme requested by the user.
void doscheme(char* schemename, deque<individual*>& testcase, double r[],
    vector<mus*>& pdf)
{
    double answer;
    vector<double> answervector;
    if (pdf.size() == 1)
        if (strcmp(schemename, "sliceupdate") == 0) {
            cerr << "Calculating with slice-update scheme..." << endl;
            answer = ehvi3d_sliceupdate(testcase, r, pdf[0]->mu, pdf[0]->s);
            cout << answer << endl;
        }
        else if (strcmp(schemename, "2term") == 0) {
            cerr << "Calculating with 2-term scheme..." << endl;
            answer = ehvi3d_2term(testcase, r, pdf[0]->mu, pdf[0]->s);
            cout << answer << endl;
        }
        else if (strcmp(schemename, "5term") == 0) {
            cerr << "Calculating with 5-term scheme..." << endl;
            answer = ehvi3d_5term(testcase, r, pdf[0]->mu, pdf[0]->s);
            cout << answer << endl;
        }
        else if (strcmp(schemename, "8term") == 0) {
            cerr << "Calculating with 8-term scheme..." << endl;
            answer = ehvi3d_8term(testcase, r, pdf[0]->mu, pdf[0]->s);
            cout << answer << endl;
        }
        else if (strcmp(schemename, "montecarlo") == 0) {
            cerr << "Calculating with Monte Carlo scheme (" << MONTECARLOS
                 << " iterations)..." << endl;
            answer = ehvi3d_montecarlo(testcase, r, pdf[0]->mu, pdf[0]->s);
            cout << answer << endl;
        }
        else {
            cerr << "Scheme " << schemename
                 << " does not exist. Proper options are:" << endl
                 << "2term" << endl
                 << "5term" << endl
                 << "8term" << endl
                 << "sliceupdate" << endl
                 << "montecarlo" << endl;
        }
    else {
        if (strcmp(schemename, "sliceupdate") == 0) {
            cerr << "Calculating with slice-update scheme (multi-version)..." << endl;
            answervector = ehvi3d_sliceupdate(testcase, r, pdf);
            for (int i = 0; i < answervector.size(); i++)
                cout << answervector[i] << endl;
        }
        else if (strcmp(schemename, "5term") == 0) {
            cerr << "Calculating with 5-term scheme (multi-version)..." << endl;
            answervector = ehvi3d_5term(testcase, r, pdf);
            for (int i = 0; i < answervector.size(); i++)
                cout << answervector[i] << endl;
        }
        else {
            cerr << "Scheme " << schemename << " does not exist." << endl
                 << "Multi-versions have only been implemented for the 5-term and "
                    "slice-update schemes!" << endl;
        }
    }
}

// Checks if p dominates P. Removes points dominated by p from P and return the
// number of points removed.
int checkdominance(deque<individual*>& P, individual* p)
{
    int nr = 0;
    for (int i = P.size() - 1; i >= 0; i--) {
        if (p->f[0] >= P[i]->f[0] && p->f[1] >= P[i]->f[1] && p->f[2] >= P[i]->f[2]) {
            cerr << "Individual " << (i + 1)
                 << " is dominated or the same as another point; removing." << endl;
            P.erase(P.begin() + i);
            nr++;
        }
    }
    return nr;
}

// Loads a testcase from the file with the name filename.
void loadtestcase(char* filename, deque<individual*>& testcase, double r[],
    vector<mus*>& pdf)
{
    ifstream file;
    int n, inds = 0;
    file.open(filename, ios::in);
    file >> n;
    for (int i = 0; i < n; i++) {
        individual* tempvidual = new individual;
        file >> tempvidual->f[0] >> tempvidual->f[1] >> tempvidual->f[2];
        tempvidual->f[0] = tempvidual->f[0];
        tempvidual->f[1] = tempvidual->f[1];
        tempvidual->f[2] = tempvidual->f[2];
        checkdominance(testcase, tempvidual);
        testcase.push_back(tempvidual);
    }
    file >> r[0] >> r[1] >> r[2];
    while (!file.eof()) {
        if (inds > 0)
            pdf.push_back(new mus);
        file >> pdf[inds]->mu[0] >> pdf[inds]->mu[1] >> pdf[inds]->mu[2];
        file >> pdf[inds]->s[0] >> pdf[inds]->s[1] >> pdf[inds]->s[2];
        if (file.fail()) {
            // We discover this while trying to read an individual and will end it
            // here.
            pdf.pop_back();
            file.close();
            return;
        }
        inds++;
    }
    file.close();
}

int main(int argc, char* argv[])
{
    int n;
    deque<individual*> testcase;
    double r[DIMENSIONS];
    double mu[DIMENSIONS];
    double s[DIMENSIONS];
    vector<mus*> pdf;
    mus* tempmus = new mus;
    pdf.push_back(tempmus);
    cout << setprecision(10);
    if (argc > 1) {
        cerr << "Loading testcase from file..." << endl;
        loadtestcase(argv[1], testcase, r, pdf);
        if (argc == 2) {
            cerr << "No scheme specified, defaulting to slice-update..." << endl;
            if (pdf.size() == 1) {
                cout << "===>" << ehvi2d(testcase, r, pdf[0]->mu, pdf[0]->s)
                     << std::endl;

                cout << ehvi3d_sliceupdate(testcase, r, pdf[0]->mu, pdf[0]->s) << endl;
            }
            else {
                vector<double> answer = ehvi3d_sliceupdate(testcase, r, pdf);
                for (int i = 0; i < answer.size(); i++)
                    cout << answer[i] << endl;
            }
        }
        else
            for (int i = 2; i < argc; i++) {
                doscheme(argv[i], testcase, r, pdf);
            }
        return 0;
    }
    else {
        cerr << "Welcome to the EHVI calculator. Please create a testcase to try "
                "out " << endl
             << " the available calculation schemes." << endl;
        cerr << "(Alternative usage: \"" << argv[0] << " FILENAME [schemes] \""
             << endl;
        cerr << "How many individuals?" << endl;
        cin >> n;
        cerr << "Enter their x, y and z coordinates. They will be tested for "
                "mutual non-dominance." << endl;
        for (int i = 1; i <= n; i++) {
            individual* tempvidual = new individual;
            cerr << "Individual " << i << " of " << n << ": ";
            cin >> tempvidual->f[0] >> tempvidual->f[1] >> tempvidual->f[2];
            i = i - checkdominance(testcase, tempvidual);
            testcase.push_back(tempvidual);
        }
        cerr << "Enter the x, y and z coordinate of the reference point. It should "
                "be dominated" << endl;
        cerr << "by all individuals in the population." << endl;
        cin >> r[0] >> r[1] >> r[2];
        cerr << "Enter the mean vector." << endl;
        cin >> mu[0] >> mu[1] >> mu[2];
        cerr << "Enter the standard deviation vector." << endl;
        cin >> s[0] >> s[1] >> s[2];
    }
    cerr << "Running your test, please wait..." << endl;
    double answer;
    cerr << "Performing Monte Carlo simulation (" << MONTECARLOS
         << " iterations)..." << endl;
    answer = ehvi3d_montecarlo(testcase, r, mu, s);
    cout << answer << endl;
    cerr << "Calculating with 8-term scheme.." << endl;
    answer = ehvi3d_8term(testcase, r, mu, s);
    cout << answer << endl;
    cerr << "Calculating with 5-term scheme.." << endl;
    answer = ehvi3d_5term(testcase, r, mu, s);
    cout << answer << endl;
    cerr << "Calculating with 2-term scheme.." << endl;
    answer = ehvi3d_2term(testcase, r, mu, s);
    cout << answer << endl;
    cerr << "Calculating with slice update scheme..." << endl;
    answer = ehvi3d_sliceupdate(testcase, r, mu, s);
    cout << answer << endl;
    return 0;
}
