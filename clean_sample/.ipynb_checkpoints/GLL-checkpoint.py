import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import kwant
import time
from numpy import sqrt,pi,cos,sin,kron
from scipy.sparse.linalg import eigsh
from multiprocessing import Pool


sigma_0 = np.array([[1, 0],[0, 1]])
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

tau_0=sigma_0
tau_x=sigma_x
tau_y=sigma_y
tau_z=sigma_z


'''fixed param'''
hbar = 1.055e-34    #J.S   
me = 9.109e-31    #kg
meff = 0.03*me
delta0=0.12   #meV


a=1e-8 #m
L=300


gmu=1.446  #meV/T
meV= 1.6023e-22
t = hbar**2/(2*meff*a**2)/meV   
gamma = 0.15 #meV

alpha=0.8  #Rashba
mu_n=15   #meV  

#self energy
def self_eng(omega):
    return -gamma/sqrt(delta0**2-omega**2)*(omega*kron(tau_0,sigma_0)+delta0*kron(tau_x,sigma_0))


def make_sys(B,omega,mu):
    lat = kwant.lattice.square(a=1,norbs=4)
    sys = kwant.Builder()
    es=1e-3
    def system_shape(pos):
        return -L//2-es < pos[0]<L//2+es and 0-es < pos[1]<1
    
    def lead_left(pos):
        return 0-es < pos[1]<1 
    def lead_right(pos):
        return 0-es < pos[1]<1 
    
    def onsite_SC(site):
        return (2 * t-mu)*kron(tau_z,sigma_0)+ self_eng(omega)+ 1/2*gmu*B*kron(tau_0,sigma_x) 
    
    def hoppingx(site1,site2):
        x,y = site1.pos
        return  -t*kron(tau_z,sigma_0) + 1j*alpha/2*kron(tau_z,sigma_y) 


    sys[lat.shape(system_shape, (0, 0))]= onsite_SC

    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hoppingx

    #left lead
    left = kwant.Builder(kwant.TranslationalSymmetry((-1,0)))
    left[lat.shape(lead_left, (-L//2-10, 0))]= (2 * t-mu_n)*kron(tau_z,sigma_0)
    left[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t *kron(tau_z,sigma_0)
    sys.attach_lead(left)

    # right lead
    right = kwant.Builder(kwant.TranslationalSymmetry((1,0)))
    right[lat.shape(lead_right, (L//2+10, 0))]= (2 * t-mu_n)*kron(tau_z,sigma_0) 
    right[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t *kron(tau_z,sigma_0)
    sys.attach_lead(right)
    return sys

def local_cond(S):
    ree=0
    reh=0
    for i in range(len(S)//2):
        for j in range(len(S)//2):
            ree=ree+np.abs(S[i,j])**2
    for i in range(len(S)//2):
        for j in range(len(S)//2,len(S)):
             reh=reh+np.abs(S[i,j])**2
    # print(np.shape(S)[0]//2,ree,reh,ree+reh,2-ree+reh)
    return (np.shape(S)[0]//2-ree+reh)


def nonlocal_cond(S):
    ree=0
    reh=0
    for i in range(len(S)//2):
        for j in range(len(S)//2):
            ree=ree+np.abs(S[i,j])**2
    for i in range(len(S)//2):
        for j in range(len(S)//2,len(S)):
             reh=reh+np.abs(S[i,j])**2
    # print(np.shape(S)[0]//2,ree,reh)
    return (ree-reh)

def cal_local_conds(sys,omega):
    smatrix = kwant.smatrix(sys,energy=omega,check_hermiticity=True)
    cond=local_cond(np.array(smatrix.submatrix(0,0)))
    return cond


def cal_nonlocal_conds(sys,omega):
    smatrix = kwant.smatrix(sys,energy=omega,check_hermiticity=True)
    cond=nonlocal_cond(np.array(smatrix.submatrix(0,1)))
    return cond
    
def plot_cond(x,y,G,title):
    X, Y = np.meshgrid(x, y)
    Z = np.array(G).T
    plt.figure()
    plt.pcolormesh(X, Y, Z, cmap='bwr', shading='auto') 
    plt.title(title)
    plt.xlabel('B')
    plt.ylabel('bias')
    cbar = plt.colorbar()  
    cbar.set_label('2e^2/h', fontsize=12) 
    plt.show()


def main():
    es=1e-6
    mu=0.3
    B=np.linspace(0,0.8,11)
    bias=np.linspace(-0.12+es,0.12-es,11)
    
    
    '''GLL'''
    t1=time.time()
    GLL=np.zeros((len(B),len(bias)))
    for i in range(len(B)):
        for j in range(len(bias)):
            system = make_sys(B[i],bias[j],mu)
            system=system.finalized()
            GLL[i][j]=cal_local_conds(system,bias[j])
            
    t2=time.time()
    print(t2-t1)
    plot_cond(B,bias,GLL/2,'T=0,GLL')
    
    np.save('GLL.npy',GLL)
    plt.savefig("GLL.png") 

if __name__ == '__main__':
    main()

