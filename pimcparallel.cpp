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

#define OMP_NUM_THREADS 20

using namespace std;


// these are used to generate random numbers later.
// int seed = 0;
// std::random_device rd;
// std::mt19937 e2( seed ? seed : rd());
auto seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 e2(seed);
std::uniform_real_distribution<> dist(0., 1.);
std::default_random_engine generator;

int n = 15; // num of particles
int M = 200; // num total timesteps
int dim = 2; // num spatial dimensions

double b = 1.0; // inverse temperature (name 'beta' gives a warning)
double dt = b/M; // imaginary timestep (related to beta)

double lambd = 6.0596; // hbar^2 / 2*m_He

double L = 15.0; // box size is -L to L in dim dimensions

int n_updates = 800; // number of attempts at updating worldlines per step
int acceptances = 0; // tracker of how many updates get accepted

int l = 5; // MULTILEVEL UPDATE UPDATES 2^l+1 BEADS AT ONCE
int upd_size = int(pow(2,l)+1);

double** energy_storage;

// OMP params
int barrier_wait = 1; // How many updates to do independently before manual barrier
int sims_per_rank = 4; // how many MCMC simulations each MPI rank should do
double min_radius = 1;

//functions
double V_ext(double**, int);
double U(double***, double**, int, int, int);

bool multilevel_update(double***, double**, int, int);
void multilevel_T(double**, int, int, int);
double delta_V(double***, double**, int, int);

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

    std::cout << "num procs: " << num_procs << std::endl;

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
        std::cout << "Total heat capacity: " << cv_full << ", min radius " << min_radius << std::endl;
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

    // Set up OMP and MPI
    omp_set_num_threads(OMP_NUM_THREADS);
    int num_threads;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto start_time = std::chrono::steady_clock::now();

    // calculate how many different non-overlapping time updates we can fit in M
    int t_bin_max = fmin(OMP_NUM_THREADS, floor(M / (upd_size-1))); // e.g. if we do 100 time steps with upd_size = 2^5+1, we can fit 3 time updates
    int max_attempts = OMP_NUM_THREADS / t_bin_max + 1; // if we have 20 threads and can fit 7 time bins we have 3 attempts at a successful update
    int max_pcls = ceil((double)n / t_bin_max);
    acceptances = 0;
    // make array of all particle worldlines:
    // Q_ijk = (particle id i, time step j, dimension k)
    // Q_test_hijk = (time bin h, attempt i, time step j, dimension k)
    // Q_test_pass = (time bin h, attempt i)
    double*** Q = new double**[n];
    double**** Q_test = new double***[t_bin_max];
    bool** Q_test_pass = new bool*[t_bin_max];

    // Q contains all particle worldlines
    #pragma omp parallel for
        for(int i = 0; i < n; ++i){
            Q[i] = new double*[M];
            for(int j = 0; j < M; ++j){
                Q[i][j] = new double[dim];
            }
        }

    // Q_test will hold several attempts at one particle's new trial worldline
    #pragma omp parallel for
        for(int h = 0; h < t_bin_max; h++) {
            Q_test[h] = new double**[max_attempts];
            for(int i = 0; i < max_attempts; i++) {
                Q_test[h][i] = new double*[M];
                for(int j = 0; j < M; ++j){
                    Q_test[h][i][j] = new double[dim];
                }
            }
        }
    
    // Q_test_pass (contains bools--whether or not a specific attempted worldline update passed the check)
    #pragma omp parallel for
        for(int h = 0; h < t_bin_max; h++) {
            Q_test_pass[h] = new bool[max_attempts];
            for(int i = 0; i < max_attempts; i++) {
                Q_test_pass[h][i] = false;
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

    // this just makes sure our initial conditions don't have any two particles too close (messed up energies)
    bool particles_too_close = true;
    while (particles_too_close) {
        particles_too_close = false;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dist2 = pow((Q[i][0][0] - Q[j][0][0]),2) + pow((Q[i][0][1] - Q[j][0][1]),2);
                if (dist2 < 1.2) {
                    particles_too_close = true;
                    // try new random position
                    for(int k = 0; k < dim; ++k) {
                        rand_q = L*(2.*dist(e2) - 1.); //pseudorand. number btwn -L and L
                        for(int t = 0; t < M; ++t) {
                            Q[j][t][k] = rand_q;
                        }
                    }
                    break;
                }
            }
            if (particles_too_close) {
                break;
            }
        }
    }

    // Metropolis-Hastings algo: try to update particle worldlines

    // due to a lack of better ideas, we generate all the t_upd's at the beginning in a big array
    int* t_upd = new int[n_updates];
    #pragma omp parallel for
        for(int i = 0; i < n_updates; i++) {
            t_upd[i] = floor(dist(e2) * M);
        }

    #pragma omp parallel
    {
        int my_id;
        my_id = omp_get_thread_num();
        num_threads = omp_get_num_threads();

        // try to fit max number of time updates into M beads
        int my_t_bin;
        int my_attempt;
        my_t_bin = my_id % t_bin_max;
        my_attempt = my_id / t_bin_max;

        int t_start;

        // Each OMP thread goes through its assigned particles and updates them
        // For now, split the particles evenly among threads. No reason to assume
        // things won't be load-balanced since these are independent particles
        int my_start, my_end;
        my_start = n * my_t_bin / t_bin_max; // = my_id * n / num_threads;
        my_end = n * (my_t_bin + 1) / t_bin_max; // (my_id + 1) * n / num_threads;

        // #pragma omp critical
        // {
        //     std::cout << "thread " << my_id << ", t_bin " << my_t_bin << ", attempt " << my_attempt << ", p_start " << my_start << std::endl;
        // }

        // Each particle is updated n_updates times
        for(int updates = 0; updates < n_updates; updates++) {

            // Each thread goes through the particles it owns
            // each particle is accepted by checking all threads' attempts before moving to next particle
            int loc_particle = my_start;
            for(int i = 0; i < max_pcls; i ++) {
                if(loc_particle < my_end) {
                    // fill in Q_test for all time slices
                    for(int t = 0; t < M; t++) {
                        for(int k = 0; k < dim; k++) {
                            Q_test[my_t_bin][my_attempt][t][k] = Q[loc_particle][t][k];
                        }
                    }

                    // attempt update (breaks if any level of multilevel algo not accepted)
                    t_start = (t_upd[updates] + my_t_bin * (upd_size-1))%M;

                    // multilevel algorithm now attempts an update in its Q_test and returns whether or not the update is valid
                    bool accept = multilevel_update(Q, Q_test[my_t_bin][my_attempt], loc_particle, t_start); 
                    Q_test_pass[my_t_bin][my_attempt] = accept;

                    if (my_attempt == 0) {
                        // one proc in each t bin will loop over each attempt, and if accepted, put it in Q and break to next particle
                        for(int attempt = 0; attempt < max_attempts; attempt++) {
                            if(Q_test_pass[my_t_bin][attempt] == true) {
                                for (int p = 1; p < upd_size-1; p++) {
                                    for (int k = 0; k < dim; k++) {
                                        Q[loc_particle][(t_start+p)%M][k] = Q_test[my_t_bin][attempt][(t_start+p)%M][k];
                                    }
                                }
                            acceptances += 1;
                            break;
                            }
                        }
                    }
                    loc_particle++;
                    Q_test_pass[my_t_bin][my_attempt] = false;
                }
                #pragma omp barrier
            }


            if (updates % barrier_wait == 0) {
                #pragma omp barrier
                }

            if (updates % 10 == 0) {
                double en, __;
                tie(en, __) = thermo_sample(Q);
                energy_storage[which_sim][updates/10] = en;
            }
        }
    }
    
    // the energy & kinetic Cv computed from this MC initial condition
    double energy_mc, cv_kin_mc;
    tie(energy_mc, cv_kin_mc) = thermo_sample(Q);

    // std::cout << "rank " << rank << ", sim " << which_sim << ", energy " << energy_mc << ", " << acceptances << " of " << n*n_updates << " moves accepted." << std::endl;

    // ** send energy + kinetic Cv to MPI rank 0
    MPI_Gatherv(&energy_mc, 1, MPI_DOUBLE, recv_buf_energy, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&cv_kin_mc, 1, MPI_DOUBLE, recv_buf_cv, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ** ----

    // ----
    // TODO?: send acceptances & total moves to MPI rank 0?
    //std::cout << acceptances << " out of " << n*n_updates << " moves accepted." << std::endl;
    // ----

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " << n << " particles.\n";

    // write an output file
    std::string filename = "worldlines_parallel" + std::to_string(rank) + ".txt";
    if(which_sim==0){std::remove(filename.c_str());}
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
    filename = "mc_energies_rank_" + std::to_string(rank) + "sim_" + std::to_string(which_sim) + ".txt";
    outFile = ofstream(filename);
    for (int s; s < n_updates/10; s++){
        outFile << energy_storage[which_sim][s] << endl;
    }
    outFile.close();

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
    double avg_sep = 4.5; //L/n;
    double r2 = 0.;
    for (int k = 0; k < dim; k++) {
        r2 += pow(Q[i][t][k]-Q_test[t][k],2);
    }
    double r6 = pow(avg_sep / r2, 3);
    double u = 4*(r6*r6 - r6);
    
    if (abs(r2)<=1e-8) {
        // cout << "r2 blew up for Qtest(a)=" << Q_test[t][k] << ", Qitk(b)=" << Q[i][t][k];
        // cout << "r2=" << r2 << ", u=" << u << endl;
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
            dV = U(Q_ijk, Q_ijk[a], b, -1, t);

            if (isnan(dV)){
                //cout << "Q_ijk(a): " << Q_ijk[a][t][0] << "," << Q_ijk[a][t][1] << endl;
                //cout << "Q_ijk(b): " << Q_ijk[b][t][0] << "," << Q_ijk[b][t][1] << endl;
                //cout << "|| for ptcls a=" << a << ", b=" << b << ", t=" << t << endl << endl;;
            }
            else {
                potential += dV;
            }
        }
    }
    return potential;
}


bool multilevel_update(double*** Q, double** Q_test, int i, int t) {
    // create array to hold delta_V for each successive level
    double* delta_V_updates = new double[l];
    double prob = 1.;
    // lvl will track which level of the update we are in
    // upd_size is how many particles are getting updated
    for(int lvl = 1; lvl <= l; lvl++) {
        // the beads that are fixed are bead t and bead t + 2^l
        // beads to be updated are t + (multiples of 2^(l-lvl))
        int spacing = int(pow(2.,l-lvl)); // spacing btwn beads getting updated this level
        for (int p = spacing/2; p < upd_size; p += spacing) {
            // update position of bead t+p in Q_test
            multilevel_T(Q_test, i, (t+p)%M, spacing);
            // compute contribution to V_tot for this bead's update
            delta_V_updates[lvl-1] += delta_V(Q, Q_test, i, (t+p)%M);
        }
        // decide whether or not to accept
        // this loop should account for contrib. from all levels prior to the current one
        for(int ll=0; ll < lvl-1; ll++) {
            prob *= exp(spacing/2 * dt * delta_V_updates[ll]);
        }
        // this will handle the current step
        prob *= exp(-spacing/2 * dt * delta_V_updates[lvl-1]);

        if (dist(e2) < prob) {
            if (lvl == l-1) {
                return true;
            }
        }
        else {
            return false;
        }
    }
    return false;
}

// do gaussian update of a single particle given the multilevel level l
// t must be the actual bead to update
// l is the level you're currently on
// spacing is spacing btwn two beads updated in same level
void multilevel_T(double** Q_test, int i, int t, int spacing) {
    double* r_mean = new double[dim];

    for (int k = 0; k < dim; k++) {
        // avg position of bead i+1 and bead i-1 (modulos etc so we have PBC)
        r_mean[k] = 0.5*(Q_test[(M+t-spacing/2)%M][k]+Q_test[(t+spacing/2)%M][k]);

        // use a gaussian to update positions; it must have stdev lambda*dt and mean r_mean[k]
        std::normal_distribution<double> distrib(r_mean[k],std::sqrt((spacing/2)*dt*lambd));
        Q_test[t%M][k] = distrib(generator);
    }
}

double delta_V(double*** Q, double** Q_test, int i, int t) {
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
    return V_tot;
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

// compute the heat capacity Cv from the 'kinetic' contribution + energy variance
// u_array, cv_kin_array should be lists of energies & kinetic heat capacities, 
// computed from different initial conditions (at the same beta)
double get_heat_cap(double* u_array, double* cv_kin_array){

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
