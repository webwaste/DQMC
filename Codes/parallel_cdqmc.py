import os
import numpy as np

def itos,digs):
    return((str(num)).zfill(digs))

#--------------------------------------------------------------------
def dtos(num,digs,dec):
    tmpstring = ""
    fmtstring = "%"+"."+str(dec)+"f"
    tmpstring=fmtstring%num
    return((str(tmpstring).zfill(digs+dec+1)))



casename = "reproduce_pair_correlation"
i = 0
Ulist    = [2, 4, 6, 8, 10, 12]
mulist   = [1, 2, 3, 4, 5, 6]
Tlist    = np.array([0.5, 1.0, 1.5, 2, 2.5, 3, 4])
for U in Ulist:
    mu = U / 2.0
    Dtau = 0.25/np.sqrt(U)
    Mlist = np.unique(np.round(1.0/(Tlist*Dtau)))
    pids = []
    print("U :", U)
    print("mu:", mu)
    inp_file_name = f"{casename}_U{dtos(U,2,3)}.in.json"
    out_file_name = f"{casename}_U{dtos(U,2,3)}.out.jnk"
    print("inp file :", inp_file_name)
    print("out file :", out_file_name)
    with open(inp_file_name, "w") as f:
        f.write("{\n")
        f.write(f'    "casename": "reproduce_local_moment",\n')
        f.write(f'    "dim": [4, 4, 4],\n')
        f.write(f'    "t": 1.0,\n')
        f.write(f'    "U": {U},\n')
        f.write(f'    "mu": {mu},\n')
        f.write(f'    "Mlist": {list(Mlist)} ,\n')
        f.write(f'    "Dtau": {Dtau},\n')
        f.write(f'    "MCsteps": 1500,\n')
        f.write(f'    "Eqsteps": 1000\n')
        f.write("}\n")
    np_start = i * 8
    np_end = (i + 1) * 8 - 1
    print(f"cpu assigned : {np_start} - {np_end}")
    os.environ["GOMP_CPU_AFFINITY"] = f"{np_start}-{np_end}"
    os.environ["BLIS_NUM_THREADS"] = "8"
    os.system(f"./cdqmc {inp_file_name} >& {out_file_name} &")
    pid = os.getpid()
    pids.append(pid)
    print("PID          :", pid)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    i += 1

