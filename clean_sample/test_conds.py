import numpy as np
import matplotlib.pyplot as plt
import kwant
import time
from numpy import sqrt,pi,cos,sin,kron
from scipy.sparse.linalg import eigsh
import scipy.sparse.linalg as sla

#pauli matrix
sigma_0 = np.array([[1, 0],[0, 1]])
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

tau_0 = sigma_0
tau_x = sigma_x
tau_y = sigma_y
tau_z = sigma_z


'''固定参数'''
hbar = 1.055e-34    #J.S  
me = 9.109e-31    #kg
meff = 0.03*me
delta0 = 0.12   #meV


a = 1e-8 #m
L = 300

#g·mu_B的值
gmu = 1.4470954503  #meV/T
meV = 1.60217e-22
t = hbar**2/(2*meff*a**2)/meV   #量纲为meV
gamma = 0.15 #meV
alpha = 0.8  #约化Rashba，等于alpha/a,单位为meV


#barrier on edge
V = 15 #meV

#self energy
def self_eng(omega):
    return -gamma/sqrt(delta0**2 - omega**2) * (omega * kron(tau_0,sigma_0) + delta0 * kron(tau_x,sigma_0))

def sys_conds(B,omega,mu,mu_lead,t_lead):
    lat = kwant.lattice.square(a=1,norbs=4)
    sys = kwant.Builder()
    es = 1e-3
    def system_shape(pos):
        return -L//2 - es < pos[0] < L//2 + es and 0-es < pos[1]<1
    def lead_left(pos):
        return 0-es < pos[1] < 1 
    def lead_right(pos):
        return 0-es < pos[1] < 1 
    
    def onsite_SC(site):
        if site.pos[0] == -L//2  or site.pos[0] == L//2:
            return (2 * t + V ) * kron(tau_z,sigma_0)
        else:
            return (2 * t - mu) * kron(tau_z,sigma_0) + self_eng(omega) + 1/2 * gmu * B * kron(tau_0,sigma_x) 
    def hoppingx(site1,site2):
        x,y = site1.pos
        return  -t * kron(tau_z,sigma_0) + 1j * alpha/2 * kron(tau_z,sigma_y) 


    sys[lat.shape(system_shape, (0, 0))] = onsite_SC
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hoppingx

    #left lead
    left = kwant.Builder(kwant.TranslationalSymmetry((-1,0)))
    left[lat.shape(lead_left, (-L//2-1 , 0))] = (2 * t_lead - mu_lead) * kron(tau_z,sigma_0)
    left[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t_lead * kron(tau_z,sigma_0)
    sys.attach_lead(left)

    # right lead
    right = kwant.Builder(kwant.TranslationalSymmetry((1,0)))
    right[lat.shape(lead_right, (L//2+1 , 0))]= (2 * t_lead - mu_lead) * kron(tau_z,sigma_0) 
    right[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t_lead * kron(tau_z,sigma_0)
    sys.attach_lead(right)
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


def plot_cond(x,y,G,title):
    X, Y = np.meshgrid(x, y)
    Z = np.array(G).T  
    plt.figure()
    plt.pcolormesh(X, Y, Z, cmap='bwr', shading='auto') 
    plt.title(title)
    plt.xlabel('B')
    plt.ylabel('bias')

    plt.xlim(np.min(x),np.max(x))
    plt.ylim(np.min(y),np.max(y))
    # 添加 colorbar 并标注
    cbar = plt.colorbar()  
    cbar.set_label('$2e^2/h$', fontsize=12)  # 设置 colorbar 的标注
    plt.show()



es = 1e-6
mu = 0
B = np.linspace(0,0.8,101)
bias = np.linspace(-0.1,0.1,101)


t1 = time.time()
GLL = np.zeros((len(B),len(bias)))
GLR = np.zeros((len(B),len(bias)))

mu_lead_list = np.linspace(0,1,1)
t_lead_list = np.linspace(0.1,2,1) * t

for mu_lead in mu_lead_list:
    for t_lead in t_lead_list:
        for i in range(len(B)):
            for j in range(len(bias)):
                system = sys_conds(B[i],bias[j],mu,mu_lead=mu_lead,t_lead=t_lead).finalized()
                GLL[i][j] = cal_local_conds(system,bias[j])

        plot_cond(B,bias,GLL/2,f'T=0,GLL,mu_lead={mu_lead},t_lead={t_lead}')



        #新的Ei
        bias_T=np.linspace(-0.1,0.1,201)
        #KT energy in 50mK
        KT= 4.31*1e-3 #meV

        #all params are numbers
        def dF(E,V,KT):
            return 1/(4*KT)*1/(np.cosh((E-V)/(2.*KT))**2)

        #新电导的维度
        g=np.zeros((len(bias_T),len(GLL[:,0])))

        # 矩形积分
        for i in range(len(bias_T)):
            for j in range(len(bias)):
                g[i,:] += GLL[:,j] * dF(bias[j],bias_T[i],KT) * (bias[2] - bias[1])

        plot_cond(B,bias_T,g/2,'T=50mK')

t2 = time.time()
print(t2-t1)