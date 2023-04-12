## Determinantal Quantum Monte Carlo
This repository contains the code for DQMC. 

----------------------------------------------
To compile the code follow these steps: 
* go to 'Codes' directory 
* change the permission of the 'compiler' file by typing the command: 
`chmod +x compiler`
then excute the compiler script by typing: 
`./compiler`

----------------------------------------------
To run the code: 
* go to 'Runs' directory.
* edit the in.dqmc.json file and save it.
* execute the code by typing: 
`./cdqmc in.dqmc.json`

----------------------------------------------
The in.dqmc.json file looks like this: 
----------------------------------------------
```json
{
    "casename": "testrun",
    "dim"     : [4,4,4],
    "t"       : 1.0,
    "U"       : 6,
    "mu"      : 3, 
    "//filling" : 0.5,
    "Mlist"   : [2,3,4,5,6,8,16],
    "Dtau"    : 0.125,
    "MCsteps" : 1500,
    "Eqsteps" :  500
}
```
----------------------------------------------
* 'dim' stands for dimensions [Nx,Ny,Nz]
* 't'   is the nearest neighbour hopping amplitude
* 'U'   is onsite interaction strength
* 'mu'  is the chemical potential
* 'filling' is the desired fractional filling of the system
* 'Mlist' contains a list of number of slices of beta. Temperature  = 1/(Dtau*M)
* 'Dtau'  width of the slice
* 'MCsteps' Total number of Monte Carlo steps
* 'Eqsteps' Number of steps to equilibriate the system

Note: while working with fixed filling, uncomment filling and comment out mu
