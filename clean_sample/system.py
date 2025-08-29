import numpy as np
import matplotlib.pyplot as plt
import kwant
import time
from numpy import sqrt,pi,kron
from scipy.sparse.linalg import eigsh
import scipy.sparse.linalg as sla
    
# spin pauli matrix
sigma_0 = np.array([[1, 0],[0, 1]])
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

# particle-hole pauli matrix
tau_0 = sigma_0
tau_x = sigma_x
tau_y = sigma_y
tau_z = sigma_z


# Dimension analysis for t:
# hbar: [J·s] = [kg·m^2·s^-1]
# hbar^2: [kg^2·m^4·s^-2]
# denominator (2 * meff * a^2): [kg·m^2]
# => hbar^2 / (2 * meff * a^2): [kg·m^2·s^-2] = [J]
# divide by meV (1 meV in joules): [J]·[meV]/[J] = [meV]
# Therefore, t has the dimension of energy, in units of meV.

''' Fixed parameters '''
hbar  = 1.055e-34       # ℏ, reduced Planck constant [J·s]
me    = 9.109e-31       # m_e, free electron mass [kg]
meff  = 0.03 * me       # m*, effective electron mass [kg]
delta0 = 0.12           # Δ₀, parent superconducting gap [meV]

a     = 1e-8            # lattice constant / grid spacing [m]
L     = 300             # system length (in units of a) [dimensionless]

gmu   = 1.4470954503    # g·μ_B, effective Zeeman factor [meV/T]
meV   = 1.60217e-22     # 1 meV in joules [J]
t     = hbar**2 / (2 * meff * a**2) / meV   # hopping energy [meV]

gamma = 0.15            # γ, superconducting coupling strength [meV]
alpha = 0.8             # α, effective Rashba SOC (α/a) [meV]
V     = 15              # barrier potential at the edge [meV]

''' Tunable parameters '''
B     = 0.3             # external magnetic field [T]
mu    = 0.0             # chemical potential [meV]
omega = 0.0             # frequncy/energy [meV]

#self energy
# def self_eng(omega):
#     return -gamma/sqrt(delta0**2 - omega**2) * (omega * kron(tau_0,sigma_0) + delta0 * kron(tau_x,sigma_0))

def self_eng(omega):
    numerator   = omega * kron(tau_0, sigma_0) + delta0 * kron(tau_x, sigma_0)
    denominator = sqrt(delta0**2 - omega**2)
    return  -gamma * numerator / denominator


def sys_conds(B, omega, mu, mu_lead=2*t, t_lead=t):
    lat = kwant.lattice.square(a=1,norbs=4,name='wire')
    sys = kwant.Builder()
    es = 1e-3
    
    def system_shape(pos):
        return -1 - es < pos[0] < L + 1 + es and 0-es < pos[1] < 1
    def lead_left(pos):
        return 0-es < pos[1] < 1 
    def lead_right(pos):
        return 0-es < pos[1] < 1 
    
    def onsite_SC(site):

        #boundary onsite barrier
        if site.pos[0] == -1  or site.pos[0] == L + 1:
            return (2 * t_lead + V - mu_lead ) * kron(tau_z,sigma_0)
        else:
            return (2 * t - mu) * kron(tau_z,sigma_0) + self_eng(omega) + 1/2 * gmu * B * kron(tau_0,sigma_x) 
        
    def hoppingx(site1,site2):
        x1 = site1.pos[0]
        x2 = site2.pos[0]

        #boundary hopping 
        if x2 == -1 or x1 == L + 1:
            return -t_lead * kron(tau_z, sigma_0)
        else:
            return -t * kron(tau_z,sigma_0) + 1j * alpha/2 * kron(tau_z,sigma_y) 


    sys[lat.shape(system_shape, (0, 0))] = onsite_SC
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hoppingx

    #left lead
    left = kwant.Builder(kwant.TranslationalSymmetry((-1,0)))
    left[lat.shape(lead_left, (-2 , 0))] = (2 * t_lead - mu_lead) * kron(tau_z,sigma_0)
    left[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t_lead * kron(tau_z,sigma_0)
    sys.attach_lead(left)

    # right lead
    right = kwant.Builder(kwant.TranslationalSymmetry((1,0)))
    right[lat.shape(lead_right, (L+2 , 0))]= (2 * t_lead - mu_lead) * kron(tau_z,sigma_0) 
    right[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t_lead * kron(tau_z,sigma_0)
    sys.attach_lead(right)
    return sys

def sys_bands(B, omega, mu, PBC=True):
    if PBC:
        sys = kwant.Builder(kwant.TranslationalSymmetry((1,0)))
    else:
        sys = kwant.Builder()

    lat = kwant.lattice.square(a=1,norbs=4,name='wire')
    es = 1e-3
    
    def system_shape(pos):
        return -1  < pos[0] < L + 1  and  0-es < pos[1] < 1

    def onsite_SC(site):
        return (2 * t - mu) * kron(tau_z,sigma_0) + self_eng(omega) + 1/2 * gmu * B * kron(tau_0,sigma_x) 
    
    def hoppingx(site1,site2):
        x,y = site1.pos
        return  -t * kron(tau_z,sigma_0) + 1j * alpha/2 * kron(tau_z,sigma_y) 

    sys[lat.shape(system_shape, (0, 0))] = onsite_SC
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hoppingx
    return sys


def local_cond(S):
    ree = np.sum(np.abs(S[:len(S)//2, :len(S)//2])**2)
    reh = np.sum(np.abs(S[:len(S)//2, len(S)//2:])**2)
    # print(np.shape(S)[0]//2,ree,reh,ree+reh,2-ree+reh)
    return (len(S)//2 - ree + reh)

def nonlocal_cond(S):
    ree = np.sum(np.abs(S[:len(S)//2, :len(S)//2])**2)
    reh = np.sum(np.abs(S[:len(S)//2, len(S)//2:])**2)
    # print(np.shape(S)[0]//2,ree,reh)
    return (ree - reh)


def cal_local_conds(sys,omega):
    smatrix = kwant.smatrix(sys,energy=omega,check_hermiticity=True)
    cond=local_cond(np.array(smatrix.submatrix(0,0)))
    return cond

def cal_nonlocal_conds(sys,omega):
    smatrix = kwant.smatrix(sys,energy=omega,check_hermiticity=True)
    cond=nonlocal_cond(np.array(smatrix.submatrix(0,1)))
    return cond


