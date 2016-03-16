MANUAL to software EHVI (c) Iris Hupkens, 2013
===================================================================

Implementation of algorithms first described in:

Iris Hupkens: Complexity Reduction and Validation of Computing the
Expected Hypervolume Improvement, Master's Thesis (with honors)
published as LIACS, Leiden University, The Netherlands, Internal
Report 2013-12, August, 2013
\url{http://www.liacs.nl/assets/Masterscripties/2013-12IHupkens.pdf}
Supervisors: Dr. Michael T.M. Emmerich, Dr. André H. Deutz

The code is organized as follows:

- ehvi_sliceupdate.cc contains the implementation of the
slice-update scheme

- ehvi_montecarlo.cc contains the implementation of the Monte Carlo
integration scheme

- ehvi_calculations.cc contains the implementations of the other
schemes (2-term, 5-term and 8-term) as well as the implementation
of the 2-dimensional calculation scheme.

-ehvi_multi.cc contains special implementations of the 5-term and
slice-update schemes which calculate the EHVI for multiple
individuals at the same time.

The files helper.cc and ehvi_hvol.cc contain functions which are
used by multiple update schemes. Hypervolume calculations are
implemented in ehvi_hvol.cc and the rest (such as the psi function)
is in helper.cc. There is also a file ehvi_consts.h which allows
the seed of the random number generator and the number of Monte
Carlo iterations to be changed.

The functions in these files can be used directly in C++ code, but
main.cc contains facilities to use it as a stand-alone command-line
application. To use it in this way, the software can be compiled
with gcc by unzipping the code into a directory and using the
command:

g++ -O3 -o EHVI -std=c++0x *.cc

If compiling it in a directory which also contains other .cc files
is desired, the following can be used instead:

g++ -O3 -o EHVI -std=c++0x main.cc helper.cc ehvi_sliceupdate.cc \
ehvi_multi.cc ehvi_montecarlo.cc ehvi_hvol.cc ehvi_calculations.cc

Running the software without command line arguments will allow a
test case to be entered into the terminal (with an arbitrary
population size and 1 candidate individual), which is then run
through the available calculation schemes.

Providing the name of a file as an argument will load that file and
perform EHVI calculations on it. The file should consist of the
following:

- A single integer representing \emph{n}

- n*3 floating point numbers representing the coordinates of P.

- 3 floating point numbers representing r

- An arbitrary number of individuals, represented by first 3
coordinates of their mean value, and then the 3 values of their
standard deviation in each dimension

No other text should be present in the file. All numbers should be
divided by whitespace, and additional whitespace is ignored.

The default scheme used will be the slice-update scheme, but other
schemes can be specified as arguments after the filename. The list
of possible schemes to request is:

2term 5term 8term sliceupdate montecarlo

However, only \emph{5term} and \emph{sliceupdate} can be used if
more than one candidate individual is provided.

4 8 8 2 11 6 7 9 5 8 14 3 9

0 0 0 6 6 6 3 3 3 5 2 4 1 3 6 1 7 2 3 5 3 2 3 5 2 8 3

Then the following command will calculate the EHVI for the four
candidate individuals using the 5-term scheme:

./EHVI multitest.txt 5term

And the output of the program will be:

Loading testcase from file... Calculating with 5-term scheme
(multi-version)... 47.24623199 11.21775781 8.935099634 19.88518203

Only the actual answers are output to stdout while the rest of the
output is written to stderr, making it possible to write the
answers to a file using simple shell commands.
