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
    Eigen::MatrixXd Rsigma = Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd Dsigma = Eigen::MatrixXd::Identity(N, N);

    for (int i = 0; i < M; i++) {
        //Determine the value of l
        int ll = (l + i)%M;
        Usigma = B(ll, sigma, s, lamda, t,  U, mu, Dtau, dim ) * Usigma;
        int M0 = 1;
        if((i+1)%M0 == 0){
            //Decomposing
    		Eigen::HouseholderQR<Eigen::MatrixXd> qr(Usigma*Dsigma);
    		Eigen::MatrixXd Q = qr.householderQ();
    		Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>(); 
            for (int j=0; j<N; j++){
                Dsigma(j,j) = R(j,j);
                for (int k = j; k<N; k++){
                    R(j,k) = R(j,k)/Dsigma(j,j);
                }
            }
            Usigma = Q; 
            Rsigma = R*Rsigma;
        }

    }
    //return (Eigen::MatrixXd::Identity(N,N) + Usigma).inverse();
    Eigen::MatrixXd gsigma_inverse = Usigma.inverse()*Rsigma.inverse() + Dsigma;
 
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(gsigma_inverse);
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>(); 
 
    for (int j=0; j<N; j++){
        Dsigma(j,j) = R(j,j);
        for (int k = j; k<N; k++){
            R(j,k) = R(j,k)/Dsigma(j,j);
        }
    }
    //std::cout<<Dsigma<<std::endl;
    Usigma = Usigma*Q;
    Rsigma = R*Rsigma;

    return (Rsigma.inverse())*(Dsigma.inverse())*(Usigma.inverse());
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
        double pair_corrl_avg = 0, pair_corrl_var = 0, pair_corrl_err = 0; 
        double spinsuccpt_avg = 0, spinsuccpt_var = 0, spinsuccpt_err = 0; 


        srand( (unsigned)time( NULL ) );
        Eigen::MatrixXd delta = Eigen::MatrixXd::Identity(N,N);
        for(int n=0; n<MCsteps; n++){
            for(int l=0; l<M; l++){

                auto Glup = g(l, 1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );
                auto Gldn = g(l,-1.0e0, M, s, lamda, t,  U, mu, Dtau, dim );

                for(int i=0; i<N; i++){
                    //s(l,i) = -s(l,i);
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
                        s(l,i) = -s(l,i);
                        //updating the green's function 
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
            }
            if (n < Eqsteps){
                progressbar(n,Eqsteps,"Equilibriating:        ");
            }
            else{
                double fill = filling(M, s, lamda, t, U, mu, Dtau, dim);
                filling_avg += fill;
                filling_var += fill*fill;

                double xxlocalmom = xx_local_moment(M, s, lamda, t, U, mu, Dtau, dim);
                xxlocalmom_avg += xxlocalmom; 
                xxlocalmom_var += xxlocalmom*xxlocalmom; 

                double pair_corrl = pair_corr_func(M, s, lamda, t, U, mu, Dtau, dim);
                pair_corrl_avg += pair_corrl; 
                pair_corrl_var += pair_corrl*pair_corrl;

                double spinsuccpt = pauli_spin_succept(M, s, lamda, t, U, mu, Dtau, dim);
                spinsuccpt_avg += spinsuccpt; 
                spinsuccpt_var += spinsuccpt*spinsuccpt;

                progressbar(n-Eqsteps,MCsteps-Eqsteps,"Measuring observables: ");

                runfile << 1/(Dtau*M) <<" "<< fill <<" "<<  xxlocalmom <<" "<< pair_corrl <<" "<<  spinsuccpt <<std::endl;
            }
        }
        runfile.close();
        std::ofstream spin_file("spin.dat");
        spin_file<<s<<std::endl;
        spin_file.close();
        filling_avg = filling_avg/((float)MCsteps - Eqsteps);
        filling_var = filling_var/((float)MCsteps - Eqsteps) - filling_avg*filling_avg;
        filling_err = sqrt(filling_var/((float) MCsteps - Eqsteps));

        xxlocalmom_avg = xxlocalmom_avg/((float)MCsteps - Eqsteps);
        xxlocalmom_var = xxlocalmom_var/((float)MCsteps - Eqsteps) - xxlocalmom_avg*xxlocalmom_avg;
        xxlocalmom_err = sqrt(xxlocalmom_var/((float) MCsteps - Eqsteps));

        pair_corrl_avg = pair_corrl_avg/((float)MCsteps - Eqsteps);
        pair_corrl_var = pair_corrl_var/((float)MCsteps - Eqsteps) - pair_corrl_avg*pair_corrl_avg;
        pair_corrl_err = sqrt(pair_corrl_var/((float) MCsteps - Eqsteps));

        spinsuccpt_avg = spinsuccpt_avg/((float)MCsteps - Eqsteps);
        spinsuccpt_var = spinsuccpt_var/((float)MCsteps - Eqsteps) - spinsuccpt_avg*spinsuccpt_avg;
        spinsuccpt_err = sqrt(spinsuccpt_var/((float) MCsteps - Eqsteps));

        std::cout << "-----------------------------------------------------\n";
        std::cout<<"Filling avg.   : "<<filling_avg<<std::endl;
        std::cout<<"Chemical Pot.  : "<<mu<<std::endl;
        std::cout<<"xxlocalmom avg.: "<<xxlocalmom_avg<<std::endl;
        std::cout<<"<s^2>          : "<<3.0e0*xxlocalmom_avg/4.0e0<<std::endl;
        std::cout<<"<C_Delta>      : "<<pair_corrl_avg<<std::endl;
        std::cout<<"<chi>          : "<<spinsuccpt_avg<<std::endl;
        std::cout << "=====================================================\n";

        outfile << 1/(Dtau*M) << " " << mu <<" "<< filling_avg <<" "<< filling_err <<" "<< xxlocalmom_avg <<" "<< xxlocalmom_err <<" "<< pair_corrl_avg <<" "<< pair_corrl_err <<" "<< spinsuccpt_avg <<" "<< spinsuccpt_err <<std::endl;

    }
    
    outfile.close();
    return 0;
}
