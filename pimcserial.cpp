#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;


// these are used to generate random numbers later.
int seed = 8;
std::random_device rd;
std::mt19937 e2( seed ? seed : rd());
std::uniform_real_distribution<> dist(0., 1.);
std::default_random_engine generator;

int n = 5; // num of particles
int M = 40; // num total timesteps
int dim = 2; // num spatial dimensions

double b = 3.0; // inverse temperature (name 'beta' gives a warning)
double dt = b/M; // imaginary timestep (related to beta)

double lambd = 6.0596; // hbar^2 / 2*m_He

double L = 10.0; // box size is -L to L in dim dimensions

int n_updates = 20000; // number of attempts at updating worldlines per step
int acceptances = 0; // tracker of how many updates get accepted


//functions
//double V_ext(std::vector<std::vector<double>>, int);
//double U(std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, int, int, int);
//void beadbybead_T(std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, int, int);
//void beadbybead_A(std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, int, int);
double V_ext(double**, int);
double U(double***, double**, int, int, int);
void beadbybead_T(double**, int, int);
void beadbybead_A(double***, double**, int, int);


int main(int argc, char* argv []) {
    std::srand(time(NULL));
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    auto start_time = std::chrono::steady_clock::now();
    
    // make array of all particle worldlines:
    // Q_ijk = (particle id i, time step j, dimension k)
    // put this back in eventually but vectors are easier to debug
    double*** Q = new double**[n];

    for(int i = 0; i < n; ++i){
        Q[i] = new double*[M];
        for(int j = 0; j < M; ++j){
            Q[i][j] = new double[dim];
        }
    }

    // // temporary vec of vec of vecs, easier to debug than pointers
    // std::vector<std::vector<std::vector<double>>> Q(n,std::vector<std::vector<double>>(M,std::vector<double>(dim)));

    // initialize particle positions somewhere in the box
    double rand_q;
    for(int i = 0; i < n; ++i) {
        for(int k = 0; k < dim; ++k) {
            rand_q = L*(2.*dist(e2) - 1.); //pseudorand. number btwn -L and L
            for(int j = 0; j < M; ++j) {
                Q[i][j][k] = rand_q;
            }
        }
    }

    // try to update particle worldlines

    // Q_test will hold one particle's new trial worldline:
    //std::vector<std::vector<double>> Q_test(M,std::vector<double>(dim));
    double** Q_test = new double*[M];
    for(int i = 0; i < M; ++i){
        Q_test[i] = new double[dim];
    }


    for(int updates = 0; updates < n_updates; updates++) {
    // try once to update a random bead on each worldline
        for(int i = 0; i < n; i++) {
            // fill in Q_test for all time slices
            for(int t = 0; t < M; t++) {
                for(int k = 0; k < dim; k++) {
                    Q_test[t][k] = Q[i][t][k];
                }
            }
            int t_upd = floor(dist(e2) * M);
            
            // T (bead-by-bead algorithm, as opposed to bisection) makes a worldline Q_test for particle i with one random bead displaced
            beadbybead_T(Q_test, i, t_upd); 

            // print statements to test how T updates particles:
            // if (i == 0 && updates > n_updates-20) {
            
            //     for (int j = 0; j < M; j++) {
            //         std::cout << Q[i][j][0] << "  " << std::flush;
            //     }
                
            //     std::cout << " | " << std::endl;

            //     for (int j = 0; j < M; j++) {
            //         std::cout << Q_test[j][0] << "  " << std::flush;
            //     }

            //     std::cout << " | " << std::endl;
            // }

            // A uses the Metropolis acceptance test to decide whether or not to accept the new worldline Q_test  
            beadbybead_A(Q, Q_test, i, t_upd); 

        }
    }

    std::cout << acceptances << " out of " << n*n_updates << " moves accepted." << std::endl;

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " << n << " particles.\n";


    // write an output file
    ofstream outFile("worldlines_serial.txt");

    // print array to file
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < dim; k++) {
                outFile << Q[i][j][k] << " ";
            }
            outFile << endl;
        }
        outFile << endl;
    }

    // close output file
    outFile.close();

    return 0;
}



// simple harmonic potential. modify for other dimensions + for pointers
// pass it some Q[i] (one particle's worldline)
double V_ext(double** Q, int t) {
    double totV = 0.;
    for (int k = 0; k < dim; k++) {
        totV += 0.2*pow(Q[t][k],2);
    }
    return totV;
}



// lennard jones potential for two particles
// needs Q[i] or Q_test (single particle worldline) and Q (for all other particles' worldlines for comparison)
double U(double*** Q, double** Q_test, int i, int j, int t) {
    double avg_sep = L/n;
    double r2 = 0.;
    for (int k = 0; k < dim; k++) {
        r2 += pow(Q[i][t][k]-Q_test[t][k],2);
    }
    double r6 = pow(avg_sep / r2, 3);
    return 4*(r6*r6 - r6);
}



// move a particle i at bead (time step) t according to gaussian dist, then write new pos.to Q_test
void beadbybead_T( double** Q_test, int i, int t) {
    double* r_mean = new double[dim];

    for (int k = 0; k < dim; k++) {
        // avg position of bead i+1 and bead i-1 ( modulos etc so we have PBC)
        r_mean[k] = 0.5*(Q_test[(M+t-1)%M][k]+Q_test[(t+1)%M][k]);

        // use a gaussian to update positions; it must have stdev lambda*dt and mean r_mean[k]
        std::normal_distribution<double> distrib(r_mean[k],dt*lambd);
        Q_test[t][k] = distrib(generator);
    }
}



// adjusts Q for a particle i and bead (time step) t
void beadbybead_A(double*** Q, double** Q_test, int i, int t) {
    double V_tot = 0;
    // calculate total difference in energy between configuration in Q and that if we used Q_test
    // aka V_tot = (energy of Q_test worldline) - (energy of current Q worldline)

    // single particle contribution
    V_tot += V_ext(Q_test, t) - V_ext(Q[i], t);

    // sum to get two particle contribution
    for (int l = 0; l < n; l++) {
        if (l != i) {
            V_tot += U(Q, Q_test, l, i, t) - U(Q, Q[i], l, i, t);
        }
    }

    //if(i==0){std::cout << V_tot << std::endl;}
    // instead of doing prob = min{1,e^(tau v' - v)}, just check if V_tot is positive
    // TODO: can speed this up a lot if we only update the correct dt rather than all M of them
    if (V_tot < 0) {
        for (int t = 0; t < M; t++) {
            for (int k = 0; k < dim; k++) {
            Q[i][t][k] = Q_test[t][k];
            }
        }
        acceptances += 1;
    }
    else {
        double prob = exp(-dt * V_tot);

        // accept move only if a random number is lower than acceptance prob
        if (dist(e2) < prob) {
            for (int t = 0; t < M; t++) {
                for (int k = 0; k < dim; k++) {
                    Q[i][t][k] = Q_test[t][k];
                }
            }
            acceptances += 1;
        }
    }
}

