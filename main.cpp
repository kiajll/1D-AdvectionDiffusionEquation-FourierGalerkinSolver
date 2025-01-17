#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <complex>
#include <fftw3.h>
#include <fstream>
#include <chrono>

using namespace std;
using namespace chrono;

//################### 00. Function Prototype Declarations ####################
void compute_rhs(double a, double nu, double Lx, const vector <double> &kx, const fftw_complex *u_hat, vector<complex<double>> &du_hat_dt);

//########################### 01. Initializations ############################
int main(){

// 0.Parameters
    const double nu{0.001};                   // Kinematic viscosity
    const int N = pow(2,8);            // Number of spatial grid points, degree of freedom
    const double U0{1.0};                     // Amplitude of the initial velocity field
    const double a{1.0};                      // Convective velocity
    const int n{1};                           // Mode of initial wavenumber
    const double Lx{2.0*M_PI};                // Length of periodic domain
    const double k = (n* 2 * M_PI / Lx);      // Wave number of initial velocity field
    const double tf = 1;                      // Final time
    const double dt = 1e-4;                   // Time step
    const size_t Nt = static_cast<size_t>(round(tf / dt)); // Number of time steps

// 1.CFL evaluation    
double dx = Lx / N;
double CFL = a * dt / dx;
cout << "CFL number equals to: " << CFL <<endl;
if (CFL > 1.0) {
    cerr << "Warning: CFL condition violated! Reduce dt or increase N." << endl;
}
// 2. Spatial grid
    vector <double> x (N,0);
    for(int i{0} ; i<N ; ++i){               // i=0:N-1; (N-1 points + zero), where N's point is x[N-1] which is equal to x[0]
        
        x[i] = i*(Lx/N);                     // Lx divided to N-1 portion with N points
    }
    
// 3. Initial velocity field: u(x,0)= -U0 * sin(kx)
    vector <double> u (N,0);
    for(int i{0} ; i < N ; ++i){
        u[i] = -U0 * sin(k * x[i]);
    }
    // 3.1. Enforcing initial periodic condition
    u[N-1] = u[0];
    
    // 3.2. Checking output
    ofstream outFile1("velocity_field_Initial.csv");  // ofstream is a class from <fstream> library

    // Write header
    outFile1 << "x,velocity\n";
    if (!outFile1) {
        cerr << "Error: Could not open initial data file for writing!" << endl;
        return -1;
    }
    // Write data
    for (int i = 0; i < N; ++i) {
        outFile1 << x[i] << "," << u[i] << "\n";
    }
    outFile1.close();

    
//################### 02. Initialize FFTW PLAN ####################
// 02.01. Define wave numbers (kx)
    vector <double> kx (N);
    for (int i = 0; i < N; ++i) {
        kx[i] = (i < N / 2) ? i * 2.0 * M_PI / Lx : (i - N) * 2.0 * M_PI / Lx;
    }    
 
// 02.02 Fourier transform of Initial data
    fftw_complex *u_hat;
    fftw_complex *du_hat_dt;

// 02.03. Allocate memory for Fourier coefficients
    size_t fft_size = sizeof(fftw_complex) * N;
    
    u_hat = (fftw_complex*) fftw_malloc(fft_size);    // u_hat is a pointer of the type fftw_complex
    memset(u_hat, 0, fft_size);  //  Zero Initialize

    du_hat_dt = (fftw_complex*) fftw_malloc(fft_size);
    memset(du_hat_dt, 0, fft_size);  //  Zero Initialize

// 02.04. Create FFTW plans
    fftw_plan forward = fftw_plan_dft_r2c_1d(N, u.data(), u_hat, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_dft_c2r_1d(N, u_hat, u.data(), FFTW_ESTIMATE);
    
// 02.05. Perform forward FFT
    fftw_execute(forward);
   
// 02.06. Allocate memory for intermediate arrays
    fftw_complex *u_hat1 = (fftw_complex*) fftw_malloc(fft_size);
    memset(u_hat1, 0, fft_size);  //  Zero Initialize
    
    fftw_complex *u_hat2 = (fftw_complex*) fftw_malloc(fft_size);
    memset(u_hat2, 0, fft_size);

    fftw_complex *u_hat3 = (fftw_complex*) fftw_malloc(fft_size);
    memset(u_hat3, 0, fft_size);
// 02.06.01. Check memory allocation success for intermediate arrays
    if (!u_hat1 || !u_hat2 || !u_hat3) {
        cerr << "Error: Memory allocation failed for intermediate arrays (u_hat1 or u_hat2). Exiting program." << endl;
        return -1;
    }
    
//  02.07. Allocate intermediate arrays to improve performance 
    vector<complex<double>> R1(kx.size());
    vector<complex<double>> R2(kx.size());
    vector<complex<double>> R3(kx.size());

//####################### 03. Processing Stage ##########################    
// 03.01. Main time-stepping loop   
auto start = high_resolution_clock::now();
    
// 03.02. Time-stepping loop
for (size_t t_cuntr{0}; t_cuntr < Nt+1; ++t_cuntr){
// 03.02.01. RK3 stage 1
    memcpy(u_hat1, u_hat, fft_size);
    compute_rhs(a, nu, Lx, kx, u_hat1, R1);
    
// 03.02.02. RK3 stage 2
    for (int i = 0; i < N; ++i) {
            u_hat2[i][0] = u_hat[i][0] + 0.5 * dt * R1[i].real();
            u_hat2[i][1] = u_hat[i][1] + 0.5 * dt * R1[i].imag();
    }
    compute_rhs(a, nu, Lx, kx, u_hat2, R2);
    
// 03.02.03. RK3 stage 3
    for (int i = 0; i < N; ++i) {
            u_hat3[i][0] = u_hat[i][0] + 0.75 * dt * R2[i].real();
            u_hat3[i][1] = u_hat[i][1] + 0.75 * dt * R2[i].imag();
    }
    compute_rhs(a, nu, Lx, kx, u_hat3, R3);
    
// 03.02.04. Final stage
    for(int i = 0; i < N; ++i){
        u_hat[i][0] += (dt / 9.0) * (2.0 * R1[i].real() + 3.0 * R2[i].real() + 4.0 * R3[i].real() );
        u_hat[i][1] += (dt / 9.0) * (2.0 * R1[i].imag() + 3.0 * R2[i].imag() + 4.0 * R3[i].imag());
    }
    
}
//########################## End of Time-stepping loop ##############################

// 03.03. Transform back to Physical space
    fftw_execute(backward);
    
// 03.04. Enforcing final step periodic condition
    u[N-1] = u[0];
  
// 03.05. Normalize Final result
    for (int i=0; i< N; ++i){
        u[i] /= N;
    }
// 03.06. Clean up
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(u_hat);
    fftw_free(du_hat_dt);
    
// 03.07.01. Output results
cout << "\n ## Final results ## " << endl;
for (int i = 0; i < N; ++i) {
    cout << "x = " << x[i] << ", u = " << u[i] << endl;
}
    
// 03.07.02. Write data to a CSV file
   ofstream outFile2("velocity_field_final.csv");  //ofstream is a class from <fstream> library
    if (!outFile2) {
        cerr << "Error: Could not open file for writing!" << endl;
        return -1;
    }

    // Write header
    outFile2 << "x,velocity\n";

    // Write data
    for (int i = 0; i < N; ++i) {
        outFile2 << x[i] << "," << u[i] << "\n";
    }

    outFile2.close();
    cout << "\nFile 'velocity_field.csv' written successfully!" << endl;

// 03.08. End measuring time
auto end = chrono::high_resolution_clock::now();
auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
cout << "Elapsed time for the time-stepping loop: " << duration.count() << " ms" << endl;    

return 0;
}


//################### 04. Function Definitions #################### 
// 04.01 rhs function: du_hat_dt = compute_rhs(nu,Lx,k,u_hat)
void compute_rhs(double a, double nu, double Lx, const vector <double> &kx, const fftw_complex *u_hat, vector<complex<double>> &du_hat_dt){
    int N = kx.size();

    for (int i = 0; i < N; ++i) {
 
        // Diffusion term: -nu * k^2 * u_hat
        complex<double> diffusion_term = -nu * pow(kx[i],2) * complex<double>(u_hat[i][0], u_hat[i][1]);

        // Advection term: -i * a * k * u_hat
        complex<double> advection_term = -complex<double>(0, 1) * a * kx[i] * complex<double>(u_hat[i][0], u_hat[i][1]);

        // Total RHS term
        du_hat_dt[i] = diffusion_term + advection_term;
    }
}
