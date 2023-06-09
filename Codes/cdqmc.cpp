#include <iostream>
#define EIGEN_USE_BLAS
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
    int barWidth = 20;
    std::cout<<prefix<<" ";
    std::cout << "[";
    int pos = round((barWidth * (float)(i + 1.0)) / (float) N);
    for (int i = 0; i < barWidth; ++i) {
        if (i <= pos) std::cout << "#";
        else std::cout << " ";
    }
    std::cout << "] " << round(((float)(i + 1.0) * 100)/(float)N) << " %\r";
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
    if(U>0){
        Eigen::VectorXd lamda_sigma_sl_by_Dtau = (1.0/Dtau)*lamda*sigma*s.row(l);
        return lamda_sigma_sl_by_Dtau  - (mu - U/2.0)*Eigen::VectorXd::Ones(s.row(l).size());
    }
    Eigen::VectorXd lamda_sl_by_Dtau = (1.0/Dtau)*lamda*s.row(l);
    return lamda_sl_by_Dtau  - (mu - U/2.0)*Eigen::VectorXd::Ones(s.row(l).size());
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
    double factor = 1.0e0;
    //if(U<0){
    //    for (int i=0; i<N; i++){
    //        factor = factor * std::exp(-Dtau*U/4.0 - s(l,i)*lamda/2.0); 
    //    }
    //}


    Eigen::MatrixXd Blsigma = expKx * expKy * expKz * expV * factor;
    return Blsigma;
}

Eigen::MatrixXd g( int l, double sigma, int M, Eigen::MatrixXd s, double lamda, double t,  double U, double mu, double Dtau, std::vector<int> dim  ) {
    int N = dim[0]*dim[1]*dim[2];
    Eigen::MatrixXd Usigma = Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd Vsigma = Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd Dsigma = Eigen::MatrixXd::Identity(N, N);

    for (int i = 0; i < M; i++) {
        //Determine the value of l
        int ll = (l + i)%M;
        Usigma = B(ll, sigma, s, lamda, t,  U, mu, Dtau, dim ) * Usigma;
        int M0 = 4;
        if((i+1)%M0 == 0 and M%M0 == 0){
            //Decomposing
            Eigen::JacobiSVD<Eigen::MatrixXd> svd( Usigma*Dsigma, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Dsigma = svd.singularValues().asDiagonal();
            Usigma = svd.matrixU(); 
            Vsigma = svd.matrixV().transpose()*Vsigma;
        }

    }
    //return (Eigen::MatrixXd::Identity(N,N) + Usigma).inverse();
    Eigen::MatrixXd gsigma_inverse = Usigma.inverse()*Vsigma.inverse() + Dsigma;
 
    Eigen::JacobiSVD<Eigen::MatrixXd> svd( gsigma_inverse, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::ArrayXd Dsigma_diag        =  svd.singularValues();
    Eigen::VectorXd Dsigmainversediag = 1.0/Dsigma_diag;
    Eigen::MatrixXd Dsigmainverse     = Dsigmainversediag.asDiagonal();
    Usigma = Usigma*svd.matrixU();
    Vsigma = svd.matrixV().transpose()*Vsigma;

    return (Vsigma.inverse())*Dsigmainverse*(Usigma.inverse());
}

double R(int l, double sigma, int i, int M, Eigen::MatrixXd s, double lamda, double t,  double U, double mu, double Dtau, std::vector<int> dim  ) {

    auto Glsigmaii = g(l,sigma, M, s, lamda, t,  U, mu, Dtau, dim )(i,i);

    return 1.0 + (1.0 - Glsigmaii )*(std::exp(-2.0*lamda*sigma*s(l,i)) - 1.0);
}

double filling(int M, Eigen::MatrixXd s, double lamda, double t, double U, double mu, double Dtau, std::vector<int> dim){
    double gsum= 0.0;
    int N = dim[0]*dim[1]*dim[2];
    for (int l=0; l<M; l++){ 
        auto Glup = g(l, 1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);
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
    return gsum/((float) M*N);
}

double pair_corr_func(int M, Eigen::MatrixXd s, double lamda, double t, double U, double mu, double Dtau, std::vector<int> dim){
    int N = dim[0]*dim[1]*dim[2];
    double gsum = 0; 
    Eigen::MatrixXd delta = Eigen::MatrixXd::Identity(N,N);
    for(int l = 0; l<M; l++){
        Eigen::ArrayXXd Glup = /*delta*/ - g(l,1.0e0 , M, s, lamda, t,  U, mu, Dtau, dim);
        Eigen::ArrayXXd Gldn = /*delta*/ - g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);

        gsum += (Glup*Gldn).sum();
    }
    return gsum/((float) N*M);
}

double pauli_spin_succept(int M, Eigen::MatrixXd s, double lamda, double t, double U, double mu, double Dtau, std::vector<int> dim){
    int N = dim[0]*dim[1]*dim[2];
    double gsum = 0; 

    for(int l = 0; l<M; l++){
        Eigen::MatrixXd Glup = g(l,1.0e0 , M, s, lamda, t,  U, mu, Dtau, dim);
        Eigen::MatrixXd Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);

        gsum += (Glup + Gldn).trace()*(Glup + Gldn).trace() + (Glup + Gldn).trace() - 0.5*(Glup*Gldn).trace() - 0.5*(Gldn*Glup).trace();
    }
    return Dtau*gsum/((float) N);
}

double energy(int M, Eigen::MatrixXd s, double lamda, double t, double U, double mu, double Dtau, std::vector<int> dim){
    int N = dim[0]*dim[1]*dim[2];
    int Nx = dim[0];
    int Ny = dim[1];
    int Nz = dim[2];
    double en = 0; 
    double hopping1 = 0; 
    double hopping2 = 0; 
    double interac = 0; 

    for(int l = 0; l<M; l++){
        Eigen::MatrixXd Glup = g(l,1.0e0 , M, s, lamda, t,  U, mu, Dtau, dim);
        Eigen::MatrixXd Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);
        for (int i = 0; i<N; i++){
            int x = i % Nx;
            int y = (i % (Nx * Ny)) / Nx;
            int z = i / (Nx * Ny);
            int xnbri = Nx * Ny * z + Nx * y + (x + 1) % Nx;
            int ynbri = Nx * Ny * z + Nx * ((y + 1) % Ny) + x;
            int znbri = Nx * Ny * ((z + 1) % Nz) + Nx * y + x;
            interac  = U*(1.0 - Glup(i,i) - Gldn(i,i) + Glup(i,i)*Gldn(i,i));
            hopping1 = -(Glup(i,xnbri) + Glup(i,ynbri) + Glup(i,znbri) + Gldn(i,xnbri) + Gldn(i,ynbri) + Gldn(i,znbri));
            hopping2 = -(Glup(xnbri,i) + Glup(ynbri,i) + Glup(znbri,i) + Gldn(xnbri,i) + Gldn(ynbri,i) + Gldn(znbri,i));
            en += interac + hopping1 + hopping2 ;
        }

    }
    return en/((float) N*M);
}



double quick_run(int M, Eigen::MatrixXd s, double lamda, double t, double U, double mu, double Dtau, std::vector<int> dim){
    int MCsteps = 500;
    int Eqsteps = 200;
    int N = dim[0]*dim[1]*dim[2];
    double filling_avg = 0;
    double filling_var = 0;
    Eigen::MatrixXd delta = Eigen::MatrixXd::Identity(N,N);

    for(int n=0; n<MCsteps; n++){
        for(int l=0; l<M; l++){
            auto Glup = g(l, 1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );
            auto Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );
            for(int i=0; i<N; i++){
                //s(l,i) = -s(l,i);
                auto Gliiup = Glup(i,i);
                double Rup = 1.0 + (1.0 - Gliiup )*(std::exp(-2.0*lamda*s(l,i)) - 1.0);
                double Rdn = Rup;
                if(U>0){
                    auto Gliidn = Gldn(i,i);
                    Rdn = 1.0 + (1.0 - Gliidn )*(std::exp(2.0*lamda*s(l,i)) - 1.0);
                }
                double r = Rup*Rdn; 
                double prob = r*std::exp(2.0*lamda*s(l,i))/(1.0e0 + r*std::exp(2.0*lamda*s(l,i)));
                if(U>0){
                    prob = r/(1.0e0 + r);
                }
                double randnum = random_number(); //Generating a random number between 0 and 1
                if(randnum < prob ){
                    //Accepting the spin flip
                    s(l,i) = -s(l,i);
                    //updating the green's function 
                    auto glup = Glup; 
                    auto gldn = Gldn; 
                    double gammalupi = std::exp(-2.0*lamda*s(l,i)) - 1.0;

                    double gammaldni = gammalupi; 
                    if(U<0){
                        gammaldni = std::exp( 2.0*lamda*s(l,i)) - 1.0;
                    }
                    for(int j = 0; j< N; j++){
                        for (int k = 0; k< N; k++){
                            Glup(j,k) = glup(j,k) - (delta(j,i) - glup(j,i))*gammalupi*glup(i,k)/(1.0e0 + (1.0e0 - glup(i,i))*gammalupi) ;
                            Gldn(j,k) = gldn(j,k) - (delta(j,i) - gldn(j,i))*gammaldni*gldn(i,k)/(1.0e0 + (1.0e0 - gldn(i,i))*gammalupi) ;
                        }
                    }
                }
            }
        }
        if (n > Eqsteps){
            double fill  = filling(M, s, lamda, t, U, mu, Dtau, dim);
            filling_avg += fill;
            filling_var += fill*fill;
            progressbar(n-Eqsteps,MCsteps-Eqsteps,"Measuring Observables  ");
        }
        else{

            progressbar(n,Eqsteps,"Equilibriating:        ");
        }
    }
    return filling_avg/((float) MCsteps - Eqsteps);
}
double chemical_potential(double given_filling, int M, Eigen::MatrixXd s, double lamda, double t, double U, double Dtau, std::vector<int> dim){
    double mu1 = -16; 
    double mu2 = -3; 
    double tolerance = 0.005;

    double diff_filling_at_mu1 = quick_run(M, s,lamda, t, U,  mu1,  Dtau, dim) - given_filling;
    double diff_filling_at_mu2 = quick_run(M, s,lamda, t, U,  mu2,  Dtau, dim) - given_filling;
    //False Position Method:
    //double mu  = (mu1*diff_filling_at_mu2 - mu2*diff_filling_at_mu1)/(mu2 - mu1);
    //double diff_filling_at_false_point = quick_run(M, s,lamda, t, U,  mu,  Dtau, dim) - given_filling;
    //

    //if(diff_filling_at_mu1 * diff_filling_at_mu2 > 0){
    //    return std::numeric_limits<double>::quiet_NaN();
    //}
    //else{
    //    
    //    while(std::abs(diff_filling_at_false_point) > tolerance){
    //        mu = (mu1 + mu2)/2.0e0;
    //        diff_filling_at_false_point = quick_run(M, s,lamda, t, U,  mu,  Dtau, dim) - given_filling;
    //        if(diff_filling_at_false_point*diff_filling_at_mu1 < 0){
    //            mu2 = mu;
    //        }
    //        else if(diff_filling_at_false_point*diff_filling_at_mu2 < 0){
    //            mu1 = mu;
    //        }
    //        std::cout<<"mu: "<<dtos(mu,2,7)<< " \t filling: "<< dtos(diff_filling_at_false_point + given_filling, 2,3)<<std::endl;
    //    }
    //}
    //Secant Method: 
    double mu = mu2;
    double diff_filling_at_mu = quick_run(M, s,lamda, t, U,  mu,  Dtau, dim) - given_filling;
    int count=0;
    while(std::abs(diff_filling_at_mu) > tolerance){
        mu2 = mu;
        diff_filling_at_mu2 = diff_filling_at_mu;
        mu = mu - diff_filling_at_mu*(mu - mu1)/ (diff_filling_at_mu - diff_filling_at_mu1);
        mu1 = mu2;
        diff_filling_at_mu = quick_run(M, s,lamda, t, U,  mu,  Dtau, dim) - given_filling;
        diff_filling_at_mu1 = diff_filling_at_mu2;
        count+=1;
        std::cout.flush();
        std::cout<<"mu: "<<dtos(mu,2,7)<< " \t filling: "<< dtos(diff_filling_at_mu + given_filling, 2,3)<<" No. of steps: "<<count<<std::endl;
        
    }
    return mu;
}




int main(int argc, char* argv[]) {
    std::string casename = "testrun";
    std::vector<int> dim {4, 4, 4};
    std::vector<int> Mlist {1, 2, 4, 6, 8};
    double Dtau = 0.01;
    double U = 4.0;
    double t = 1.0;
    double mu = 0.0;
    double desired_filling = 1.0e0;
    bool adjust_filling = false;
    int MCsteps = 1500; 
    int Eqsteps = 500; 

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
    if (!params["filling"].empty()) {
        desired_filling = params["filling"].asDouble();
        adjust_filling = true;
    }
    if (!params["MCsteps"].empty()) {
        MCsteps = params["MCsteps"].asInt();
    }
    if (!params["Eqsteps"].empty()) {
        Eqsteps = params["Eqsteps"].asInt();
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
    std::cout << "t      : " << t << "\n";
    std::cout << "U      : " << U << "\n";
    std::cout << "mu     : " << mu << "\n";
    if(adjust_filling){
        std::cout << "filling: " << desired_filling << "\n";
    }
    else{
        std::cout << "mu     : " << mu << "\n";
    }
    std::cout << "Mlist  : ";
    for (auto& val : Mlist) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    std::cout << "Dtau   : " << Dtau << "\n";
    std::cout << "MCsteps: " << MCsteps << "\n";
    std::cout << "Eqsteps: " << Eqsteps << "\n";
    std::cout << "-----------------------------------------------------\n";

    std::string outfilename = casename + "_t" + dtos(t,1,3) + "_U" + dtos(U, 2,3) + ".dat"  ;
    std::ofstream outfile(outfilename);



    //------------------------------
    //parameters calculable from input parameters
    int N = dim[0] * dim[1] * dim[2];
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
        double T = 1/(M*Dtau);
        std::string runfilename = casename + "_t" + dtos(t,1,3) + "_U" + dtos(U, 2,3) + "_T" + dtos(T,1,3) +  ".dat"  ;
        std::ofstream runfile(runfilename);
        std::cout<<"M: " << M <<" " <<"T: " << T << std::endl;

        Eigen::MatrixXd s(M,N);
        for (int l=0; l<M; l++){
            for (int j=0; j<N; j++){
                s(l,j) = 2.0e0*dist(rng) - 1.0e0;
            }
        }
        if(adjust_filling){
            mu = chemical_potential(desired_filling,M, s,lamda, t, U, Dtau, dim);
        }

        double filling_avg = 0, filling_var = 0, filling_err = 0; 
        double xxlocalmom_avg = 0, xxlocalmom_var = 0, xxlocalmom_err = 0; 
        double zzlocalmom_avg = 0, zzlocalmom_var = 0, zzlocalmom_err = 0; 
        double pair_corrl_avg = 0, pair_corrl_var = 0, pair_corrl_err = 0; 
        double spinsuccpt_avg = 0, spinsuccpt_var = 0, spinsuccpt_err = 0; 
        double energy_avg = 0    , energy_var = 0    , energy_err = 0; 
        double spin_spin_corrn1_avg = 0    , spin_spin_corrn1_var = 0    , spin_spin_corrn1_err = 0; 
        double spin_spin_corrn2_avg = 0    , spin_spin_corrn2_var = 0    , spin_spin_corrn2_err = 0; 
        double spin_spin_corrn3_avg = 0    , spin_spin_corrn3_var = 0    , spin_spin_corrn3_err = 0; 

        int accepting_count=0; // To see how many times the  flipping of spin is accepted. 

        srand( (unsigned)time( NULL ) );
        Eigen::MatrixXd delta = Eigen::MatrixXd::Identity(N,N);
        for(int n=0; n<MCsteps; n++){
            for(int l=0; l<M; l++){

                auto Glup = g(l, 1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );
                auto Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );

                for(int i=0; i<N; i++){
                    s(l,i) = -s(l,i);
                    //auto Glup = g(l, 1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );
                    //auto Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );

                    auto Gliiup = Glup(i,i);
                    double Rup = 1.0 + (1.0 - Gliiup )*(std::exp(-2.0*lamda*s(l,i)) - 1.0);
                    double Rdn = Rup;
                    if(U>0){
                        auto Gliidn = Gldn(i,i);
                        Rdn = 1.0 + (1.0 - Gliidn )*(std::exp(2.0*lamda*s(l,i)) - 1.0);
                    }
                    //std::cout<<"Rup: "<<dtos(Rup,1,5)<<" Rdn: "<<dtos(Rdn,1,5)<<" r: "<<dtos(Rdn*Rup,1,5)<<std::endl;
                    double r = Rup*Rdn; 
                    //double prob = r/(1.0e0 + r);
                    double prob = r*std::exp(2.0*lamda*s(l,i))/(1.0e0 + r*std::exp(2.0*lamda*s(l,i)));
                    if(U>0){
                        prob = r/(1.0e0 + r);
                    }
                    double randnum = random_number(); //Generating a random number between 0 and 1
                                                      
                    if(randnum < prob ){
                        accepting_count++;
                        //s(l,i) = -s(l,i);
                        //updating the green's function 
                        if(U>8){
                            Glup = g(l, 1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );
                            Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );
                        }
                        else{

                            auto glup = Glup; 
                            auto gldn = Gldn; 
                            double gammalupi = std::exp(-2.0*lamda*s(l,i)) - 1.0;

                            double gammaldni = gammalupi; 
                            if(U>0){
                                gammaldni = std::exp( 2.0*lamda*s(l,i)) - 1.0;
                            }
                            for(int j = 0; j< N; j++){
                                for (int k = 0; k< N; k++){
                                    Glup(j,k) = glup(j,k) - (delta(j,i) - glup(j,i))*gammalupi*glup(i,k)/(1.0e0 + (1.0e0 - glup(i,i))*gammalupi) ;
                                    Gldn(j,k) = gldn(j,k) - (delta(j,i) - gldn(j,i))*gammaldni*gldn(i,k)/(1.0e0 + (1.0e0 - gldn(i,i))*gammalupi) ;
                                }
                            }
                        }
                    }
                    else{
                        s(l,i) = -s(l,i);
                    }
                }
            }
            if (n < Eqsteps){
                progressbar(n,Eqsteps,"Equilibriating:        ");
            }
            else{
                                // Measurement is going on here.
                double fill= 0.0;
                double xxlocalmom = 0;
                double zzlocalmom = 0;
                double pair_corrl = 0;
                double spinsuccpt = 0;
                double spin_spin_corrn1 = 0;
                double spin_spin_corrn2 = 0;
                double spin_spin_corrn3 = 0;
                double en = 0;
                double hopping1 = 0;
                double hopping2 = 0;
                double interac  = 0;
                int N = dim[0]*dim[1]*dim[2];
                int Nx = dim[0];
                int Ny = dim[1];
                int Nz = dim[2];

                Eigen::MatrixXd delta = Eigen::MatrixXd::Identity(N,N);

                for (int l=0; l<M; l++){
                    auto Glup = g(l, 1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);
                    auto Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);
                    Eigen::ArrayXd Glup_diag = Glup.diagonal().array();
                    Eigen::ArrayXd Gldn_diag = Gldn.diagonal().array();
                    Eigen::ArrayXXd Gluparr = /*delta*/ - g(l,1.0e0 , M, s, lamda, t,  U, mu, Dtau, dim);
                    Eigen::ArrayXXd Gldnarr = /*delta*/ - g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim);


                    fill += (Glup + Gldn).trace();
                    xxlocalmom += ((1.0e0 - Glup_diag)*Gldn_diag + (1.0e0 - Gldn_diag)*Glup_diag).sum();
                    zzlocalmom += ((1.0e0 - Glup_diag) - 2.0e0*(1.0e0 - Glup_diag)*(1.0e0 - Gldn_diag) + (1.0e0 - Gldn_diag)).sum();
                    pair_corrl += (Gluparr*Gldnarr).sum();
                    spinsuccpt += (Glup + Gldn).trace()*(Glup + Gldn).trace() + (Glup + Gldn).trace() - 0.5*(Glup*Gldn).trace() - 0.5*(Gldn*Glup).trace();

                    for (int i = 0; i<N; i++){
                        int x = i % Nx;
                        int y = (i % (Nx * Ny)) / Nx;
                        int z = i / (Nx * Ny);
                        int xnbri = Nx * Ny * z + Nx * y + (x + 1) % Nx;
                        int ynbri = Nx * Ny * z + Nx * ((y + 1) % Ny) + x;
                        int znbri = Nx * Ny * ((z + 1) % Nz) + Nx * y + x;
                        interac   = U*(1.0 - Glup(i,i) - Gldn(i,i) + Glup(i,i)*Gldn(i,i));
                        hopping1  = -(Glup(i,xnbri) + Glup(i,ynbri) + Glup(i,znbri) + Gldn(i,xnbri) + Gldn(i,ynbri) + Gldn(i,znbri));
                        hopping2  = -(Glup(xnbri,i) + Glup(ynbri,i) + Glup(znbri,i) + Gldn(xnbri,i) + Gldn(ynbri,i) + Gldn(znbri,i));
                        en += interac + hopping1 + hopping2 ;
                    }

                //Things that are calculated from the ising spin field..

                    for (int i=0; i<N; i++){
                        int x = i % Nx;
                        int y = (i % (Nx * Ny)) / Nx;
                        int z = i / (Nx * Ny);
                        int xnbr1i = Nx * Ny * z + Nx * y + (x + 1) % Nx;
                        int ynbr1i = Nx * Ny * z + Nx * ((y + 1) % Ny) + x;
                        int znbr1i = Nx * Ny * ((z + 1) % Nz) + Nx * y + x;
                        
                        int xynbr2i = Nx * Ny * z + Nx * ((y + 1) % Ny) + (x + 1) % Nx;
                        int zynbr2i = Nx * Ny * ((z + 1) % Nz) + Nx * ((y + 1) % Ny) + x;
                        int zxnbr2i = Nx * Ny * ((z + 1) % Nz) + Nx * y + (x + 1) % Nx;

                        int xyznbr3i = Nx * Ny * ((z + 1) % Nz) + Nx * ((y + 1) % Ny) + (x + 1) % Nx;

                        //spin_spin_corrn1 += s(l,i)*(s(l,xnbr1i) + s(l,ynbr1i) + s(l,znbr1i));
                        //spin_spin_corrn2 += s(l,i)*(s(l,xynbr2i) + s(l,zynbr2i) + s(l,zxnbr2i));
                        //spin_spin_corrn3 += s(l,i)*s(l,xyznbr3i); 
                        spin_spin_corrn1 += (-Glup(i,xnbr1i)*Gldn(xnbr1i,i)-Glup(i,xnbr1i)*Gldn(xnbr1i,i)-Glup(i,ynbr1i)*Gldn(ynbr1i,i)-Glup(i,ynbr1i)*Gldn(ynbr1i,i)-Glup(i,znbr1i)*Gldn(znbr1i,i)-Glup(i,znbr1i)*Gldn(znbr1i,i))/3.0e0;
                        spin_spin_corrn2 += (-Glup(i,xynbr2i)*Gldn(xynbr2i,i)-Glup(i,zynbr2i)*Gldn(zynbr2i,i)-Glup(i,zynbr2i)*Gldn(zynbr2i,i)-Glup(i,zynbr2i)*Gldn(zynbr2i,i)-Glup(i,zxnbr2i)*Gldn(zxnbr2i,i)-Glup(i,zxnbr2i)*Gldn(zxnbr2i,i))/3.0e0;
                        spin_spin_corrn3 += -Glup(i,xyznbr3i)*Gldn(xyznbr3i,i) - Gldn(i,xyznbr3i)*Glup(xyznbr3i,i); 
                    }
                }
                spin_spin_corrn1 = spin_spin_corrn1/((double) M*N);
                spin_spin_corrn2 = spin_spin_corrn2/((double) M*N);
                spin_spin_corrn3 = spin_spin_corrn3/((double) M*N);

                fill  = fill/((float) M*N);
                xxlocalmom = xxlocalmom/((float) M*N);
                zzlocalmom = zzlocalmom/((float) M*N);
                pair_corrl = pair_corrl/((float) N*M);
                en = en/((float) N*M);
                //==============================================================================
                //==============================================================================
                filling_avg += fill;
                filling_var += fill*fill;

                xxlocalmom_avg += xxlocalmom; 
                xxlocalmom_var += xxlocalmom*xxlocalmom; 

                zzlocalmom_avg += zzlocalmom; 
                zzlocalmom_var += zzlocalmom*zzlocalmom; 

                pair_corrl_avg += pair_corrl; 
                pair_corrl_var += pair_corrl*pair_corrl;

                spinsuccpt_avg += spinsuccpt; 
                spinsuccpt_var += spinsuccpt*spinsuccpt;

                energy_avg+= en;
                energy_var+= en*en;

                spin_spin_corrn1_avg += spin_spin_corrn1;
                spin_spin_corrn1_var += spin_spin_corrn1*spin_spin_corrn1;

                spin_spin_corrn2_avg += spin_spin_corrn2;
                spin_spin_corrn2_var += spin_spin_corrn2*spin_spin_corrn1;

                spin_spin_corrn3_avg += spin_spin_corrn3;
                spin_spin_corrn3_var += spin_spin_corrn3*spin_spin_corrn3;

                progressbar(n-Eqsteps,MCsteps-Eqsteps,"Measuring observables: ");

                runfile << 1/(Dtau*M) <<" "<< fill <<" "<< en << " "  << xxlocalmom <<" "<< zzlocalmom <<" "<< spin_spin_corrn1<<" "<< spin_spin_corrn2<<" "<< spin_spin_corrn3<<" " <<pair_corrl <<" "<<  spinsuccpt <<std::endl;
            }
        }
        runfile.close();
        std::ofstream spin_file("spin.dat");
        spin_file<<s<<std::endl;
        spin_file.close();
        filling_avg = filling_avg/((float)MCsteps - Eqsteps);
        filling_var = filling_var/((float)MCsteps - Eqsteps) - filling_avg*filling_avg;
        filling_err = sqrt(filling_var/((float) MCsteps - Eqsteps));

        energy_avg = energy_avg/((float)MCsteps - Eqsteps);
        energy_var = energy_var/((float)MCsteps - Eqsteps) - energy_avg*energy_avg;
        energy_err = sqrt(energy_var/((float) MCsteps - Eqsteps));

        xxlocalmom_avg = xxlocalmom_avg/((float)MCsteps - Eqsteps);
        xxlocalmom_var = xxlocalmom_var/((float)MCsteps - Eqsteps) - xxlocalmom_avg*xxlocalmom_avg;
        xxlocalmom_err = sqrt(xxlocalmom_var/((float) MCsteps - Eqsteps));

        zzlocalmom_avg = zzlocalmom_avg/((float)MCsteps - Eqsteps);
        zzlocalmom_var = zzlocalmom_var/((float)MCsteps - Eqsteps) - zzlocalmom_avg*zzlocalmom_avg;
        zzlocalmom_err = sqrt(zzlocalmom_var/((float) MCsteps - Eqsteps));

        spin_spin_corrn1_avg = spin_spin_corrn1_avg/((float)MCsteps - Eqsteps);
        spin_spin_corrn1_var = spin_spin_corrn1_var/((float)MCsteps - Eqsteps) - spin_spin_corrn1_avg*spin_spin_corrn1_avg;
        spin_spin_corrn1_err = sqrt(spin_spin_corrn1_var/((float) MCsteps - Eqsteps));

        spin_spin_corrn2_avg = spin_spin_corrn2_avg/((float)MCsteps - Eqsteps);
        spin_spin_corrn2_var = spin_spin_corrn2_var/((float)MCsteps - Eqsteps) - spin_spin_corrn2_avg*spin_spin_corrn2_avg;
        spin_spin_corrn2_err = sqrt(spin_spin_corrn2_var/((float) MCsteps - Eqsteps));

        spin_spin_corrn3_avg = spin_spin_corrn3_avg/((float)MCsteps - Eqsteps);
        spin_spin_corrn3_var = spin_spin_corrn3_var/((float)MCsteps - Eqsteps) - spin_spin_corrn3_avg*spin_spin_corrn3_avg;
        spin_spin_corrn3_err = sqrt(spin_spin_corrn3_var/((float) MCsteps - Eqsteps));

        pair_corrl_avg = pair_corrl_avg/((float)MCsteps - Eqsteps);
        pair_corrl_var = pair_corrl_var/((float)MCsteps - Eqsteps) - pair_corrl_avg*pair_corrl_avg;
        pair_corrl_err = sqrt(pair_corrl_var/((float) MCsteps - Eqsteps));

        spinsuccpt_avg = spinsuccpt_avg/((float)MCsteps - Eqsteps);
        spinsuccpt_var = spinsuccpt_var/((float)MCsteps - Eqsteps) - spinsuccpt_avg*spinsuccpt_avg;
        spinsuccpt_err = sqrt(spinsuccpt_var/((float) MCsteps - Eqsteps));

        std::cout << "-----------------------------------------------------\n";
        std::cout<<"Filling avg.   : "<<filling_avg<<std::endl;
        std::cout<<"<E>            : "<<energy_avg<<std::endl;
        std::cout<<"Chemical Pot.  : "<<mu<<std::endl;
        std::cout<<"xxlocalmom avg.: "<<xxlocalmom_avg<<std::endl;
        std::cout<<"zzlocalmom avg.: "<<zzlocalmom_avg<<std::endl;
        std::cout<<"<s^2>          : "<<3.0e0*xxlocalmom_avg/4.0e0<<std::endl;
        std::cout<<"spinspincorrn1 : "<<spin_spin_corrn1_avg<<std::endl;
        std::cout<<"spinspincorrn2 : "<<spin_spin_corrn2_avg<<std::endl;
        std::cout<<"spinspincorrn3 : "<<spin_spin_corrn3_avg<<std::endl;
        std::cout<<"<C_Delta>      : "<<pair_corrl_avg<<std::endl;
        std::cout<<"<chi>          : "<<spinsuccpt_avg<<std::endl;
        std::cout<<"accepting count: "<<accepting_count<<std::endl;
        std::cout << "=====================================================\n";

        outfile << 1/(Dtau*M) << " " << mu <<" "<< filling_avg <<" "<< filling_err << " " << energy_avg << " " << energy_err <<" "<< xxlocalmom_avg <<" "<< xxlocalmom_err <<" " << zzlocalmom_avg <<" " << zzlocalmom_err <<" "<< spin_spin_corrn1_avg<<" " << spin_spin_corrn1_err<<" "<< spin_spin_corrn2_avg<<" "<< spin_spin_corrn2_err<< " " <<spin_spin_corrn3_avg<<" "<< spin_spin_corrn3_err << " " << pair_corrl_avg <<" "<< pair_corrl_err <<" "<< spinsuccpt_avg <<" "<< spinsuccpt_err <<std::endl;

    }
    
    outfile.close();
    return 0;
}
