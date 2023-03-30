#include <iostream>
#define EIGEN_USE_BLIS
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include<string>
#include<cmath>
#include<random>
#include <fstream>
#include "json/json.h"
#include<iomanip>


using namespace Eigen;
//progress bar for making the program fancy
void progressbar(int i, int N, std::string prefix){
    int barWidth = 100;
    std::cout<<prefix<<"\t";
    std::cout << "[";
    int pos = (barWidth * i) / N;
    for (int i = 0; i < barWidth; ++i) {
        if (i <= pos) std::cout << "#";
        else std::cout << " ";
    }
    std::cout << "] " << (i * 100)/N  + 1<< " %\r";
    std::cout.flush();
    if(i==N-1){
        std::cout << std::endl;
    }
}

std::string dtos(double num, int bd, int ad){
    //=========================================================================
    //bd is number of digits before decimal point.
    //ad is number of digits after decimal point.
    //=========================================================================
    std::stringstream fract;
    fract <<std::setw(ad+2)<<std::setprecision(ad)<<std::fixed<<(num - floor(num));
    std::string frac = fract.str().erase(0,1);

    std::stringstream whole;
    whole << std::setfill('0')<<std::setw(bd)<<floor(num);
    std::string whl = whole.str();


    return (whl + frac);
}
std::string itos(int num, int bd){
    //=========================================================================
    //bd is number of digits in the int.
    //=========================================================================
    std::stringstream number;
    number<< std::setfill('0')<<std::setw(bd)<<num;
    return number.str();
}


double random_number(){

    double randnum = (float) rand()/RAND_MAX ;
    return randnum; 
}
std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd> K(int Nx, int Ny, int Nz) {
    int N = Nx * Ny * Nz;

    MatrixXd Kxa = MatrixXd::Zero(N, N);
    MatrixXd Kxb = MatrixXd::Zero(N, N);
    MatrixXd Kya = MatrixXd::Zero(N, N);
    MatrixXd Kyb = MatrixXd::Zero(N, N);
    MatrixXd Kza = MatrixXd::Zero(N, N);
    MatrixXd Kzb = MatrixXd::Zero(N, N);

    for (int i = 0; i < N; i++) {
        int x = i % Nx;
        int y = (i % (Nx * Ny)) / Nx;
        int z = i / (Nx * Ny);
        int xnbri = Nx * Ny * z + Nx * y + (x + 1) % Nx;
        int ynbri = Nx * Ny * z + Nx * ((y + 1) % Ny) + x;
        int znbri = Nx * Ny * ((z + 1) % Nz) + Nx * y + x;

        Kxa(i, xnbri) = -((x + 1) % 2);
        Kxa(xnbri, i) = -((x + 1) % 2);

        Kxb(i, xnbri) = -(x % 2);
        Kxb(xnbri, i) = -(x % 2);

        Kya(i, ynbri) = -((y + 1) % 2);
        Kya(ynbri, i) = -((y + 1) % 2);

        Kyb(i, ynbri) = -(y % 2);
        Kyb(ynbri, i) = -(y % 2);

        Kza(i, znbri) = -((z + 1) % 2);
        Kza(znbri, i) = -((z + 1) % 2);

        Kzb(i, znbri) = -(z % 2);
        Kzb(znbri, i) = -(z % 2);
    }

    return std::make_tuple(Kxa, Kxb, Kya, Kyb, Kza, Kzb);
}

Eigen::VectorXd V(int l,  double sigma, Eigen::MatrixXd s, double lamda, double U, double mu, double  Dtau ) {
    if(U>=0){
        Eigen::VectorXd lamda_sigma_sl_by_Dtau = (1.0/Dtau)*lamda*sigma*s.row(l);
        return lamda_sigma_sl_by_Dtau  - (mu - U/2.0)*Eigen::VectorXd::Ones(s.row(l).size());
    }
    else if(U<0){
        Eigen::VectorXd lamda_sigma_sl_by_Dtau = (1.0/Dtau)*lamda*sigma*s.row(l);
        return lamda_sigma_sl_by_Dtau  - (mu - U/2.0)*Eigen::VectorXd::Ones(s.row(l).size());
    }
}

Eigen::MatrixXd B(int l, double sigma,Eigen::MatrixXd s,double  lamda,double t,double  U, double mu,double  Dtau, std::vector<int> dim ) {
    int Nx = dim[0];
    int Ny = dim[1];
    int Nz = dim[2];
    int N  = Nx*Ny*Nz;

    auto [Kxa, Kxb, Kya, Kyb, Kza, Kzb] = K(Nx, Ny, Nz);

    Eigen::MatrixXd expKxa = Kxa * std::sinh(Dtau * t) + Eigen::MatrixXd::Identity(N, N) * std::cosh(Dtau * t);
    Eigen::MatrixXd expKxb = Kxb * std::sinh(Dtau * t) + Eigen::MatrixXd::Identity(N, N) * std::cosh(Dtau * t);
    Eigen::MatrixXd expKya = Kya * std::sinh(Dtau * t) + Eigen::MatrixXd::Identity(N, N) * std::cosh(Dtau * t);
    Eigen::MatrixXd expKyb = Kyb * std::sinh(Dtau * t) + Eigen::MatrixXd::Identity(N, N) * std::cosh(Dtau * t);
    Eigen::MatrixXd expKza = Kza * std::sinh(Dtau * t) + Eigen::MatrixXd::Identity(N, N) * std::cosh(Dtau * t);
    Eigen::MatrixXd expKzb = Kzb * std::sinh(Dtau * t) + Eigen::MatrixXd::Identity(N, N) * std::cosh(Dtau * t);

    Eigen::MatrixXd expKx = expKxa * expKxb;
    Eigen::MatrixXd expKy = expKya * expKyb;
    Eigen::MatrixXd expKz = expKza * expKzb;

    auto v = V(l, sigma, s, lamda, U, mu, Dtau);
    Eigen::MatrixXd expV = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; i++){
        expV(i, i) = std::exp(-Dtau*v(i));
    }

    Eigen::MatrixXd Blsigma = expKx * expKy * expKz * expV;
    return Blsigma;
}

Eigen::MatrixXd g( int l, double sigma, int M, Eigen::MatrixXd s, double lamda, double t,  double U, double mu, double Dtau, std::vector<int> dim  ) {
    int N = dim[0]*dim[1]*dim[2];
    Eigen::MatrixXd Alsigma = Eigen::MatrixXd::Identity(N, N);

    for (int i = l; i < M; i++) {
        Alsigma = B(i, sigma, s, lamda, t,  U, mu, Dtau, dim ) * Alsigma;
    }
    for (int i = 0; i < l; i++) {
        Alsigma = B(i, sigma, s, lamda, t,  U, mu, Dtau, dim ) * Alsigma;
    }

    return (Eigen::MatrixXd::Identity(N, N) + Alsigma).inverse();
}

double R(int l, double sigma, int i, int M, Eigen::MatrixXd s, double lamda, double t,  double U, double mu, double Dtau, std::vector<int> dim  ) {
    auto Glsigmaii = g(l,sigma, M, s, lamda, t,  U, mu, Dtau, dim )(i,i);

    return 1.0 + (1.0 - Glsigmaii )*(std::exp(-2.0*lamda*sigma*s(l,i)) - 1.0);
}

double filling(int M, Eigen::MatrixXd s, double lamda, double t, double U, double mu, double Dtau, std::vector<int> dim){
    double gsum= 0.0;
    int N = dim[0]*dim[1]*dim[2];
    for (int l=0; l<M; l++){ 
        auto Glup = g(l,1.0e0 , M, s, lamda, t,  U, mu, Dtau, dim);
        auto Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);
        gsum += (Glup + Gldn).trace();
    }

    return 2.0 - gsum/(M*N);
}

double xx_local_moment(int M, Eigen::MatrixXd s, double lamda, double t, double U, double mu, double Dtau, std::vector<int> dim){
    int N = dim[0]*dim[1]*dim[2];
    double gsum = 0; 
    for(int l = 0; l<M; l++){
        auto Glup = g(l,1.0e0 , M, s, lamda, t,  U, mu, Dtau, dim);
        auto Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);
        Eigen::ArrayXd Glup_diag = Glup.diagonal().array();
        Eigen::ArrayXd Gldn_diag = Gldn.diagonal().array();
        gsum += ((1.0e0 - Glup_diag)*Gldn_diag + (1.0e0 - Gldn_diag)*Glup_diag).sum();

    }
    return gsum/((float) N*M);
}


    

int main(int argc, char* argv[]) {
    std::string casename = "testrun";
    std::vector<int> dim {4, 4, 4};
    std::vector<int> Mlist {1, 2, 4, 6, 8};
    double Dtau = 0.01;
    double U = -4.0;
    double t = 1.0;
    double mu = 0.0;

    // Loading the parameters from JSON file
    std::string inp_file_name = argv[1];
    std::ifstream inp_file(inp_file_name);
    Json::Value params;
    inp_file >> params;

    // Overwrite default parameter values with the values from the JSON file
    
    if (!params["casename"].empty()) {
        casename = params["casename"].asString();
    }
    if (!params["dim"].empty()) {
        dim.clear();
        for (auto& val : params["dim"]) {
            dim.push_back(val.asInt());
        }
    }
    if (!params["Mlist"].empty()) {
        Mlist.clear();
        for (auto& val : params["Mlist"]) {
            Mlist.push_back(val.asInt());
        }
    }
    if (!params["Dtau"].empty()) {
        Dtau = params["Dtau"].asDouble();
    }
    if (!params["U"].empty()) {
        U = params["U"].asDouble();
    }
    if (!params["t"].empty()) {
        t = params["t"].asDouble();
    }
    if (!params["mu"].empty()) {
        mu = params["mu"].asDouble();
    }

    std::cout << "-----------------------------------------------------\n";
    std::cout << "Parameters obtained from: " << inp_file_name << "\n";
    std::cout << "-----------------------------------------------------\n";
    std::cout << "Casename: " << casename <<std::endl;
    std::cout << "dim   : ";
    for (auto& val : dim) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    std::cout << "t     : " << t << "\n";
    std::cout << "U     : " << U << "\n";
    std::cout << "mu    : " << mu << "\n";
    std::cout << "Mlist : ";
    for (auto& val : Mlist) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    std::cout << "Dtau  : " << Dtau << "\n";
    std::cout << "-----------------------------------------------------\n";

    std::string outfilename = casename + "_t" + dtos(t,1,3) + "_U" + dtos(U, 2,3) + ".dat"  ;
    std::ofstream outfile(outfilename);


    //------------------------------
    //parameters calculable from input parameters
    int N = dim[0] * dim[1] * dim[2];
    int Nsteps = 1500;
    double lamda = std::acosh(std::exp(std::abs(U) * Dtau / 2.0));

//    auto [Kxa, Kxb, Kya, Kyb, Kza, Kzb] = K(Nx, Ny, Nz);

//    std::cout << "Kxa:\n" << Kxa*Kxa << std::endl;
//    std::cout << "Kxb:\n" << Kxb*Kxb << std::endl;
//    std::cout << "Kya:\n" << Kya*Kya << std::endl;
//    std::cout << "Kyb:\n" << Kyb*Kyb << std::endl;
//    std::cout << "Kza:\n" << Kza*Kza << std::endl;
//    std::cout << "Kzb:\n" << Kzb*Kzb << std::endl;


    int seed = 3001;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,1); 

    //Initializing the H.S. spins
    for (auto& M : Mlist) {
        Eigen::MatrixXd s(M,N);
        for (int l=0; l<M; l++){
            for (int j=0; j<N; j++){
                s(l,j) = 2.0e0*dist(rng) - 1.0e0;
            }
        }

        double filling_avg = 0, filling_var = 0, filling_err = 0; 
        double xxlocalmom_avg = 0, xxlocalmom_var = 0, xxlocalmom_err = 0; 

        srand( (unsigned)time( NULL ) );
        for(int n=0; n<Nsteps; n++){
            for(int l=0; l<M; l++){
                for(int i=0; i<N; i++){
                    s(l,i) = -s(l,i);
                    auto Rup = R(l,1.0 ,i, M, s, lamda, t,  U, mu, Dtau, dim );
                    auto Rdn = R(l,-1.0,i, M, s, lamda, t,  U, mu, Dtau, dim );
                    double r = Rup*Rdn; 
                    double prob = r/(1.0e0 + r);
                    double randnum = random_number(); //Generating a random number between 0 and 1
                    if(randnum > prob ){
                        s(l,i) = -s(l,i);
                    }
                }
            }
            if (n < 500){
                progressbar(n,500,"Equilibriating : ");
            }
            else{
                double fill = filling(M, s, lamda, t, U, mu, Dtau, dim);
                filling_avg += fill;
                filling_var += fill*fill;

                double xxlocalmom = xx_local_moment(M, s, lamda, t, U, mu, Dtau, dim);
                xxlocalmom_avg += xxlocalmom; 
                xxlocalmom_var += xxlocalmom*xxlocalmom; 

                progressbar(n-500,Nsteps-500,"Measuring observables : ");
            }
        }

        filling_avg = filling_avg/((float)Nsteps - 500 );
        filling_var = filling_var/((float)Nsteps - 500 ) - filling_avg*filling_avg;
        filling_err = sqrt(filling_var/((float) Nsteps - 500 ));

        xxlocalmom_avg = xxlocalmom_avg/((float)Nsteps - 500 );
        xxlocalmom_var = xxlocalmom_var/((float)Nsteps - 500 ) - xxlocalmom_avg*xxlocalmom_avg;
        xxlocalmom_err = sqrt(xxlocalmom_var/((float) Nsteps - 500 ));

        std::cout<<"M: " << M << std::endl;
        std::cout<<"T: " << 1/(Dtau*M) << std::endl;
        std::cout<<"Filling avg: "<<filling_avg<<std::endl;
        std::cout<<"xxlocalmom avg: "<<xxlocalmom_avg<<std::endl;

        outfile << 1/(Dtau*M) <<" "<< filling_avg <<" "<< filling_err <<" "<< xxlocalmom_avg <<" "<< xxlocalmom_err << std::endl;

    }
    outfile.close();
    return 0;
}
