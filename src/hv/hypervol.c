/*---------------------------------------------------------------------------*/
/* Hypervolume Metric Calculation                                            */
/*---------------------------------------------------------------------------*/
/* This program calculates for a given set of objective vectors the volume   */
/* of the dominated space, enclosed by the nondominated points and the       */
/* origin. Here, a maximization problem is assumed, for minimization         */
/* or mixed optimization problem the objective vectors have to be trans-     */
/* formed accordingly. The hypervolume metric has been proposed in:          */
/*                                                                           */
/* 1. E. Zitzler and L. Thiele. Multiobjective Optimization Using            */
/*    Evolutionary Algorithms - A Comparative Case Study. Parallel Problem   */
/*    Solving from Nature - PPSN-V, September 1998, pages 292-301.           */
/*                                                                           */
/* A more detailed description and extensions can be found in:               */
/*                                                                           */
/* 2. E. Zitzler. Evolutionary Algorithms for Multiobjective Optimization:   */
/*    Methods and Applications. Swiss Federal Institute of Technology (ETH)  */
/*    Zurich. Shaker Verlag, Germany, ISBN 3-8265-6831-1, December 1999.     */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/* Usage (command line parameters):                                          */
/*                                                                           */
/* 1) number of objectives                                                   */
/* 2) name of the file containing the first set of objective vectors         */
/* 3) name of the file containing the second set of objective vectors (this  */
/*    parameter is optional)                                                 */
/*                                                                           */
/* The file format is as follows: Each line in the file describes one point  */
/* of the trade-off front and contains a sequence of numbers, separated by   */
/* blanks. Per line, the first number corresponds to the first objective,    */
/* the second number to the second objective, and so forth.                  */
/*                                                                           */
/* Output:                                                                   */
/*                                                                           */
/* 1) volume of the space dominated by the first set of objective vectors    */
/* 2) volume of the space dominated by the second set of objective vectors   */
/*    (if the third command line parameter has been specified)               */
/* 3) volume of the space which is dominated by the first set of objective   */
/*    but not by the second set of objective vectors (if the third command   */
/*    line parameter has been specified)                                     */
/* 4) volume of the space which is dominated by the second set of objective  */
/*    but not by the first set of objective vectors (if the third command    */
/*    line parameter has been specified)                                     */
/*                                                                           */
/* Outputs 1+2) refer to the S metric and outputs 3+4) to the D metric as    */
/* described in reference 2 (see above).                                     */
/*---------------------------------------------------------------------------*/
/* Eckart Zitzler                                                            */
/* Computer Engineering and Networks Laboratory (TIK)                        */
/* Swiss Federal Institute of Technology (ETH) Zurich, Switzerland           */
/* (c)2001                                                                   */
/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#define ERROR(x) fprintf(stderr, x), fprintf(stderr, "\n"), exit(1)

int Dominates(double point1[], double point2[], int noObjectives)
/* returns true if 'point1' dominates 'points2' with respect to the
	to the first 'noObjectives' objectives */
{
    int i;
    int betterInAnyObjective;

    betterInAnyObjective = 0;
    for (i = 0; i < noObjectives && point1[i] >= point2[i]; i++)
        if (point1[i] > point2[i])
            betterInAnyObjective = 1;
    return (i >= noObjectives && betterInAnyObjective);
} /* Dominates */

void Swap(double* front[], int i, int j)
{
    double* temp;

    temp = front[i];
    front[i] = front[j];
    front[j] = temp;
} /* Swap */

int FilterNondominatedSet(double* front[], int noPoints, int noObjectives)
/* all nondominated points regarding the first 'noObjectives' dimensions
	are collected; the points referenced by 'front[0..noPoints-1]' are
	considered; 'front' is resorted, such that 'front[0..n-1]' contains
	the nondominated points; n is returned */
{
    int i, j;
    int n;

    n = noPoints;
    i = 0;
    while (i < n) {
        j = i + 1;
        while (j < n) {
            if (Dominates(front[i], front[j], noObjectives)) {
                /* remove point 'j' */
                n--;
                Swap(front, j, n);
            }
            else if (Dominates(front[j], front[i], noObjectives)) {
                /* remove point 'i'; ensure that the point copied to index 'i'
	   is considered in the next outer loop (thus, decrement i) */
                n--;
                Swap(front, i, n);
                i--;
                break;
            }
            else
                j++;
        }
        i++;
    }
    return n;
} /* FilterNondominatedSet */

double SurfaceUnchangedTo(double* front[], int noPoints, int objective)
/* calculate next value regarding dimension 'objective'; consider
	points referenced in 'front[0..noPoints-1]' */
{
    int i;
    double minValue, value;

    if (noPoints < 1)
        ERROR("run-time error");
    minValue = front[0][objective];
    for (i = 1; i < noPoints; i++) {
        value = front[i][objective];
        if (value < minValue)
            minValue = value;
    }
    return minValue;
} /* SurfaceUnchangedTo */

int ReduceNondominatedSet(double* front[], int noPoints, int objective,
    double threshold)
/* remove all points which have a value <= 'threshold' regarding the
	dimension 'objective'; the points referenced by
	'front[0..noPoints-1]' are considered; 'front' is resorted, such that
	'front[0..n-1]' contains the remaining points; 'n' is returned */
{
    int n;
    int i;

    n = noPoints;
    for (i = 0; i < n; i++)
        if (front[i][objective] <= threshold) {
            n--;
            Swap(front, i, n);
        }
    return n;
} /* ReduceNondominatedSet */

double CalculateHypervolume(double* front[], int noPoints,
    int noObjectives)
{
    int n;
    double volume, distance;

    volume = 0;
    distance = 0;
    n = noPoints;
    while (n > 0) {
        int noNondominatedPoints;
        double tempVolume, tempDistance;

        noNondominatedPoints = FilterNondominatedSet(front, n, noObjectives - 1);
        tempVolume = 0;
        if (noObjectives < 3) {
            if (noNondominatedPoints < 1)
                ERROR("run-time error");
            tempVolume = front[0][0];
        }
        else
            tempVolume = CalculateHypervolume(front, noNondominatedPoints,
                noObjectives - 1);
        tempDistance = SurfaceUnchangedTo(front, n, noObjectives - 1);
        volume += tempVolume * (tempDistance - distance);
        distance = tempDistance;
        n = ReduceNondominatedSet(front, n, noObjectives - 1, distance);
    }
    return volume;
} /* CalculateHypervolume */

int ReadFront(double** frontPtr[], FILE* file, int noObjectives)
{
    int noPoints;
    int i;
    double value;

    /* check file and count points */
    noPoints = 0;
    while (!feof(file)) {
        for (i = 0; i < noObjectives && fscanf(file, "%lf", &value) != EOF; i++)
            ;
        if (i > 0 && i < noObjectives)
            ERROR("data in file incomplete");
        noPoints++;
    }
    /* allocate memory */
    *frontPtr = malloc(noPoints * sizeof(double*));
    if (*frontPtr == NULL)
        ERROR("memory allocation failed");
    for (i = 0; i < noPoints; i++) {
        (*frontPtr)[i] = malloc(noObjectives * sizeof(double));
        if ((*frontPtr)[i] == NULL)
            ERROR("memory allocation failed");
    }
    /* read data */
    rewind(file);
    noPoints = 0;
    while (!feof(file)) {
        for (i = 0; i < noObjectives; i++) {
            if (fscanf(file, "%lf", &value) != EOF)
                (*frontPtr)[noPoints][i] = value;
            else
                break;
        }
        if (i > 0 && i < noObjectives)
            ERROR("data in file incomplete");
        noPoints++;
    }
    if (noPoints < 1)
        ERROR("file contains no data");
    return noPoints;
} /* ReadFront */

int MergeFronts(double** frontPtr[], double* front1[], int sizeFront1,
    double* front2[], int sizeFront2, int noObjectives)
{
    int i, j;
    int noPoints;

    /* allocate memory */
    noPoints = sizeFront1 + sizeFront2;
    *frontPtr = malloc(noPoints * sizeof(double*));
    if (*frontPtr == NULL)
        ERROR("memory allocation failed");
    for (i = 0; i < noPoints; i++) {
        (*frontPtr)[i] = malloc(noObjectives * sizeof(double));
        if ((*frontPtr)[i] == NULL)
            ERROR("memory allocation failed");
    }
    /* copy points */
    noPoints = 0;
    for (i = 0; i < sizeFront1; i++) {
        for (j = 0; j < noObjectives; j++)
            (*frontPtr)[noPoints][j] = front1[i][j];
        noPoints++;
    }
    for (i = 0; i < sizeFront2; i++) {
        for (j = 0; j < noObjectives; j++)
            (*frontPtr)[noPoints][j] = front2[i][j];
        noPoints++;
    }

    return noPoints;
} /* MergeFronts */

void DeallocateFront(double** front, int noPoints)
{
    int i;

    if (front != NULL) {
        for (i = 0; i < noPoints; i++)
            if (front[i] != NULL)
                free(front[i]);
        free(front);
    }
} /* DeallocateFront */

#if 0
int  main(int  argc, char  *argv[])
{
  FILE      *file1, *file2;
  double    **front1, **front2, **front3;
  int       sizeFront1, sizeFront2, sizeFront3;
  int       redSizeFront1, redSizeFront2, redSizeFront3;
  double    volFront1, volFront2, volFront3;
  int       noObjectives;

  /* check parameters */
  if (argc < 3)  ERROR("missing arguments");
  sscanf(argv[1], "%d", &noObjectives);
  if (noObjectives < 2)  ERROR("invalid argument");
  file1 = fopen(argv[2], "r");
  if (file1 == NULL)  ERROR("cannot open file");
  if (argc == 4) {
    file2 = fopen(argv[3], "r");
    if (file2 == NULL)  ERROR("cannot open file");
  }
  /* read in data */
  sizeFront1 = ReadFront(&front1, file1, noObjectives);
  fclose(file1);
  if (argc == 4) {
    sizeFront2 = ReadFront(&front2, file2, noObjectives);
    sizeFront3 = MergeFronts(&front3, front1, sizeFront1, front2, sizeFront2,
			     noObjectives);
    fclose(file2);
  }
  /* calculate dominated hypervolume */
  redSizeFront1 = FilterNondominatedSet(front1, sizeFront1, noObjectives);
  volFront1 = CalculateHypervolume(front1, redSizeFront1, noObjectives);
  printf("%.10f ", volFront1);
  if (argc ==4 ) {
    redSizeFront2 = FilterNondominatedSet(front2, sizeFront2, noObjectives);
    volFront2 = CalculateHypervolume(front2, redSizeFront2, noObjectives);
    printf("%.10f ", volFront2);
    redSizeFront3 = FilterNondominatedSet(front3, sizeFront3, noObjectives);
    volFront3 = CalculateHypervolume(front3, redSizeFront3, noObjectives);
    printf("%.10f %.10f", volFront3 - volFront2,
	   volFront3 - volFront1);
  }
  DeallocateFront(front1, sizeFront1);
  if (argc ==4 ) {
    DeallocateFront(front2, sizeFront2);
    DeallocateFront(front3, sizeFront3);
  }
  printf("\n");
}
#endif
