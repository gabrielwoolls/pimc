#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <utility> // for std::pair used in `thermo_sample`
#include <omp.h>
#include <tuple> // for std::pair used in `thermo_sample`
#include <mpi.h>

#define OMP_NUM_THREADS 12

using namespace std;


// these are used to generate random numbers later.
// int seed = 0;
// std::random_device rd;
// std::mt19937 e2( seed ? seed : rd());
auto seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 e2(seed);
std::uniform_real_distribution<> dist(0., 1.);
std::default_random_engine generator;

int n = 100; // num of particles
int M = 40; // num total timesteps
int dim = 2; // num spatial dimensions

double b = 3.0; // inverse temperature (name 'beta' gives a warning)
double dt = b/M; // imaginary timestep (related to beta)

double lambd = 6.0596; // hbar^2 / 2*m_He

double L = 10.0; // box size is -L to L in dim dimensions

int n_updates = 20000; // number of attempts at updating worldlines per step
int acceptances = 0; // tracker of how many updates get accepted

double** energy_storage;

// OMP params
int barrier_wait = 1; // How many updates to do independently before manual barrier

int sims_per_rank = 3; // how many MCMC simulations each MPI rank should do

double min_radius = 1;

bool PRINT_THINGS = false;
bool SAVE_ENERGIES = false;


//functions
double V_ext(double**, int);
double U(double***, double**, int, int, int);
void beadbybead_T(double**, int, int);
void beadbybead_A(double***, double**, int, int);
double U_all_particles(double ***, int);
std::tuple<double, double> thermo_sample(double***);
void do_one_mc_sim(double*, double*, int*, int*, int, int);
double get_heat_cap(double*, double*);


int main(int argc, char* argv []) {


    // Init MPI
    int num_procs, rank, sim_num;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "BEGIN num procs: " << num_procs << std::endl;

    double* recv_buf_energy = NULL;
    double* recv_buf_cv = NULL;

    int *displs = new int[num_procs];
    int* counts = new int[num_procs];
    for (int r=0; r<num_procs; r++){
        counts[r] = 1; // only one energy (or Cv) will be sent at a time
    }

    if (rank==0) {
        recv_buf_energy = new double[num_procs * sims_per_rank];
        recv_buf_cv = new double[num_procs * sims_per_rank];
    }
    
    energy_storage = new double*[sims_per_rank];
    for (int sim=0; sim<sims_per_rank; sim++){
        energy_storage[sim] = new double[n_updates/10];
    }
    

    for (int s=0; s<sims_per_rank; s++){
        
        // sim_num = s + rank * sims_per_rank; // the 'id' of this simulation

        for (int r=0; r<num_procs; r++){
            displs[r] = s + r * sims_per_rank; // only one energy (or Cv) will be sent at a time
        }

        do_one_mc_sim(recv_buf_energy, recv_buf_cv, displs, counts, num_procs, s);
    }
    

    // do final energy + Cv analysis
    if (rank==0){
        std::cout << "energies: " << std::endl;
        for (int i=0; i<num_procs * sims_per_rank; i++){
            std::cout << recv_buf_energy[i] << endl;
        }

        // calculate full heat capacity (could just do this in python tbh)
        double cv_full = get_heat_cap(recv_buf_energy, recv_buf_cv);

        // write to output file
        ofstream outFile("observables.txt"); 

        // add (energy, cv_kin) data to file, for each simulation
        for (int i = 0; i < num_procs * sims_per_rank; i++) {
            outFile << recv_buf_energy[i] << " " << recv_buf_cv[i];
            outFile << endl;
        }

        // add the calculated total heat capacity in final line
        outFile << endl;
        outFile << cv_full;
        outFile.close();

        // while we're at it, print out in terminal
        std::cout << "Total heat capacity: " << cv_full << ", min radius " << min_radius;
    }

    
    MPI_Finalize();    
    return 0;
}


void do_one_mc_sim(double* recv_buf_energy, double* recv_buf_cv, int* displs, int* counts, int num_procs, int which_sim) {
    // do one simulation (at specified parameters beta, N, M, etc) with a 
    // single Markov initial condition, and return one energy + kinetic Cv

    std::srand(time(NULL));
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    // Set up OMP (and MPI?)
    omp_set_num_threads(OMP_NUM_THREADS);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto start_time = std::chrono::steady_clock::now();

    // make array of all particle worldlines:
    // Q_ijk = (particle id i, time step j, dimension k)
    double*** Q = new double**[n];

    #pragma omp parallel for
        for(int i = 0; i < n; ++i){
            Q[i] = new double*[M];
            for(int j = 0; j < M; ++j){
                Q[i][j] = new double[dim];
            }
        }


    // initialize particle positions somewhere in the box
    double rand_q;
    #pragma omp parallel for
        for(int i = 0; i < n; ++i) {
            for(int k = 0; k < dim; ++k) {
                seed = std::chrono::system_clock::now().time_since_epoch().count();
                std::mt19937 e2(seed);
                rand_q = L*(2.*dist(e2) - 1.); //pseudorand. number btwn -L and L
                for(int j = 0; j < M; ++j) {
                    Q[i][j][k] = rand_q;
                }
            }
        }

    // Metropolis-Hastings algo: try to update particle worldlines

    #pragma omp parallel
    {
        int my_id, num_threads;
        my_id = omp_get_thread_num();
        num_threads = omp_get_num_threads();


        // Q_test will hold one particle's new trial worldline:
        // Each OpenMP thread needs its own `Q_test`, hence why it's after the `#pragma`
        double** Q_test = new double*[M];
        for(int i = 0; i < M; ++i){
            Q_test[i] = new double[dim];
        }
        int t_upd;

        // Each OMP thread goes through its assigned particles and updates them
        // For now, split the particles evenly among threads. No reason to assume
        // things won't be load-balanced since these are independent particles
        int my_start, my_end;
        my_start = my_id * n / num_threads;
        my_end = (my_id + 1) * n / num_threads;
        if (my_id == num_threads - 1) {my_end = n;}

        // Each particle is updated n_updates times
        for(int updates = 0; updates < n_updates; updates++) {

            // Each thread goes through the particles it owns
            for( int loc_particle = my_start; loc_particle < my_end; loc_particle++) {

                // fill in Q_test for all time slices
                for(int t = 0; t < M; t++) {
                    for(int k = 0; k < dim; k++) {
                        Q_test[t][k] = Q[loc_particle][t][k];
                    }
                }

                t_upd = floor(dist(e2) * M);
                beadbybead_T(Q_test, loc_particle, t_upd); 

                // A uses the Metropolis acceptance test to decide whether or not to accept the new worldline Q_test  
                beadbybead_A(Q, Q_test, loc_particle, t_upd); 
            }

                if (updates % barrier_wait == 0) {
                    #pragma omp barrier
                    }

                if (SAVE_ENERGIES == true && updates % 10 == 0) {
                    double en, __;
                    tie(en, __) = thermo_sample(Q);
                    energy_storage[which_sim][updates/10] = en;
                }

            }
    }
    
    // the energy & kinetic Cv computed from this MC initial condition
    double energy_mc, cv_kin_mc;
    tie(energy_mc, cv_kin_mc) = thermo_sample(Q);

    if (PRINT_THINGS == true){
    std::cout << "rank " << rank << ", sim " << which_sim << ", energy " << energy_mc << std::endl;
    }

    // ** send energy + kinetic Cv to MPI rank 0
    MPI_Gatherv(&energy_mc, 1, MPI_DOUBLE, recv_buf_energy, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&cv_kin_mc, 1, MPI_DOUBLE, recv_buf_cv, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ** ----

    // ----
    // TODO?: send acceptances & total moves to MPI rank 0?
    // std::cout << acceptances << " out of " << n*n_updates << " moves accepted." << std::endl;
    // ----

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    // std::cout << "Simulation Time = " << seconds << " seconds for " << n << " particles.\n";

    // write an output file
    std::string filename = "worldlines_parallel" + std::to_string(rank) + ".txt";
    ofstream outFile(filename, std::ios::app);  // open in append mode -> NEED TO DELETE FILES BEFORE RUNNING

    // print array to file
    for (int i = 0; i < n; i++) {

        outFile << "particle " << i << ", simulation " << which_sim << endl;
        
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


    // **** Store energies in files, to check convergence ****
    if (SAVE_ENERGIES == true){
    filename = "mc_energies_rank_" + std::to_string(rank) + "sim_" + std::to_string(which_sim) + ".txt";
    outFile = ofstream(filename);
    for (int s; s < n_updates/10; s++){
        outFile << energy_storage[which_sim][s] << endl;
    }
    outFile.close();
    }
    

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
    double u = 4*(r6*r6 - r6);
    
    if (PRINT_THINGS == true && abs(r2)<=1e-8) {
        // cout << "r2 blew up for Qtest(a)=" << Q_test[t][k] << ", Qitk(b)=" << Q[i][t][k];
        cout << "r2=" << r2 << ", u=" << u << endl;
    }

    if (abs(r2) < min_radius){
        min_radius = r2;
    }


    return u;
}

// interparticle potential U for all particle pairs at time t
double U_all_particles(double*** Q_ijk, int t){
    double potential=0;
    double dV;
    for (int a=0; a<n; a++){
        for (int b=a+1; b<n; b++){
            // add potential between particles (a,b)
            dV = U(Q_ijk, Q_ijk[a], b, 0, t);

            if (isnan(dV)){
                cout << "Q_ijk(a): " << Q_ijk[a][t][0] << "," << Q_ijk[a][t][1] << endl;
                cout << "Q_ijk(b): " << Q_ijk[b][t][0] << "," << Q_ijk[b][t][1] << endl;
                cout << "|| for ptcls a=" << a << ", b=" << b << ", t=" << t << endl << endl;;
            }

            potential += dV;
        }
    }
    return potential;
}


// move a particle i at bead (time step) t according to gaussian dist, then write new pos.to Q_test
void beadbybead_T(double** Q_test, int i, int t) {
    double* r_mean = new double[dim];

    for (int k = 0; k < dim; k++) {
        // avg position of bead i+1 and bead i-1 ( modulos etc so we have PBC)
        r_mean[k] = 0.5*(Q_test[(M+t-1)%M][k]+Q_test[(t+1)%M][k]);

        // use a gaussian to update positions; it must have stdev lambda*dt and mean r_mean[k]
        std::normal_distribution<double> distrib(r_mean[k],std::sqrt(dt*lambd));
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

            if (isnan(U(Q, Q_test, l, i, t))){
                cout<<"U(q,qtest) blew up, ptcls" << i << ", " << l << endl;
            }

            if (isnan(U(Q, Q[i], l, i, t))){
                cout<<"U(q,qi) blew up, ptcls" << i << ", " << l << endl;
            }

            if (isnan(V_tot)){
                cout<<"V_tot blew up, ptcls" << i << ", " << l << endl;
            }                        
        }
    }

    //if(i==0){std::cout << V_tot << std::endl;}
    // instead of doing prob = min{1,e^(tau v' - v)}, just check if V_tot is positive
    if (V_tot < 0) {
        for (int k = 0; k < dim; k++) {
            Q[i][t][k] = Q_test[t][k];
            }
        acceptances += 1;
    }
    else {
        double prob = exp(-dt * V_tot);

        // accept move only if a random number is lower than acceptance prob
        if (dist(e2) < prob) {
            for (int k = 0; k < dim; k++) {
                    Q[i][t][k] = Q_test[t][k];
                }

            acceptances += 1;
        }
    }
}

std::tuple<double, double> thermo_sample(double*** Q_ijk){
// Takes a set of worldlines `Q_ijk` (i->particle, j->time step, k->spatial dim)
// and computes its contribution to the thermal energy U

    double kinetic = 0;
    double potential = 0;

    double u, cv = 0;

    for (int i=0; i<n; i++){
        for (int k=0; k<dim; k++){
            for (int j=0; j<M-1; j++){
                // kinetic term (R_{mu+1}-R_mu)^2
                kinetic += pow((Q_ijk[i][j+1][k]-Q_ijk[i][j][k]),2);
                
                // external harmonic potential
                potential += V_ext(Q_ijk[i], j);

                // if (isnan(potential)){
                //     cout << "harmonic potential blew up" << endl;
                // }

                // interparticle LJ potential
                potential += U_all_particles(Q_ijk, j);

                // if (isnan(potential)){
                //     cout << "U_LJ blew up" << endl;
                // }
            }
            // plus the periodic-boundary kinetic term connecting times 0 and M
            kinetic += pow((Q_ijk[i,M-1,k]-Q_ijk[i,0,k]),2);
        }
    }

    // kinetic *= M/(4*lambd*pow(b,2));
    // potential /= M;
    // return -(kinetic + potential);

    u = 0.5*dim*n*M/b -potential/M - kinetic*M/(4*lambd*pow(b,2));
    cv = -kinetic * M/(2*lambd*b);

    // if (std::isnan(u)){
    //     std::cout<< "nan energy: potential = " << potential << ", kinetic = " << kinetic << std::endl; 
    // }

    return std::make_tuple(u, cv);
    }

// compute heat capacity Cv from the 'kinetic' contribution + energy variance
double get_heat_cap(double* u_array, double* cv_kin_array){
    // u_array, cv_kin_array should be lists of energies & kinetic heat capacities, 
    // computed from different initial conditions (at the same beta)

    int size = sizeof(u_array) / sizeof(u_array[0]); // assume cv_kin_array is of same size

    double avg_u, avg_u2, cv_kinetic = 0;
    for (int i=0; i<size; i++){
        avg_u += u_array[i];
        avg_u2 += u_array[i]*u_array[i];
        cv_kinetic += cv_kin_array[i];
    }

    avg_u /= size;
    avg_u2 /= size;
    cv_kinetic /= size;

    double Cv = b * b * (avg_u2-pow(avg_u,2));
    Cv += cv_kinetic + 0.5*dim*n*M; // include the equipartition contribution as well

    return Cv;
}



