import multiprocessing as mp
from typing import ContextManager
import gpytorch
import math
import torch
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import distance_matrix
import numpy as np
import ot
#from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, WhiteNoiseKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import cos, nonzero, random
from torch import serialization
import pandas as pd
import time
import os
import netCDF4 as nc
algnm='MD-GP'
septnm='MinC-Known'
compnm='CGP-LCB'
knalgnm='MD-Known'
#print('x')
#change import to csv
lat=13
lon=10
vw=pd.read_csv('data_2016_2_%d_%d.csv' %(lat,lon), sep=',',header=None)
vw=np.asarray(vw[1:])
vw=vw[:,1:]
alt_level=[1600.04, 1459.58, 1328.43, 1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70, 566.49, 500.91, 440.58, 385.14, 334.22, 287.51, 244.68, 205.44, 169.50, 136.62, 106.54, 79.04, 53.92, 30.96, 10.00]
#print(vw.shape)
#from gpytorch.distributions.multivariate_normal import MultivariateNormal

title=int(time.time())%123456789
n_cores=25
from FRThelper import Tree
from Bayopthelper_real_2_wind_oldhyp_zerovr2_woday import FRTOBJECT
from runnerhelper_real_withbar_1216_oldhyp_fontadjusted_extrem_zerovr2_woday import runnerandplotter
#change bay in run when changing bay here**********



#kernel rbf or matern?
class RKHSModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(RKHSModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())+ScaleKernel(RBFKernel())#+ScaleKernel(RBFKernel(active_dims=torch.tensor([2])))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    def length_scale(self, value):
        self.covar_module.lengthscale = value 

# initialize likelihood and model
likelihood = GaussianLikelihood()


#k=RBFKernel(lengthscale=0.2)
#k(x,x).evaluate()

T=960
nd=int(T/24)
n=vw.shape[1]
#print(n)
n_e=24#+1
x_a=torch.tensor(alt_level[0:n])
#print(x_a)
e=torch.linspace(0,n_e-1,n_e)
#e=torch.rand(n_e)
x_s,e_s=torch.meshgrid(x_a,e)
d_s=torch.linspace(0,nd-1,nd)
x_s,e_s,d_s=torch.meshgrid(x_a,e,d_s)

d_in=torch.linspace(0,nd-1,nd)
train_x_e=torch.cat((x_s.reshape(n*n_e*nd,1),e_s.reshape(n*n_e*nd,1),d_s.reshape(n*n_e*nd,1)),1)
#normalization
#print(x_s)
#xmn=torch.mean(x_a)
#xsd=torch.std(x_a)
xmn=572.91#562.73#572.91#566.9#572.91#572.04#572.91#560
xsd=463.2#470.37#463.2#474.34#463.2#466.38#463.2#468.5
x_in=(x_s-xmn)/xsd
#emn=torch.mean(e)
#esd=torch.std(e)
emn=9.67#10.31#9.67#10.5#9.67#11.48#9.67#11.8
esd=6.4#6.615#6.4#6.8#6.4#6.7#6.4#7.0
e_in=(e_s-emn)/esd
dmn=19.5#4.42#4.38#4.42#4.41#4.42#4.21#4.42#4.11
dsd=11.5#2.9#2.86#2.9#2.8#2.9#2.66#2.9#2.87
#dmn=torch.mean(d_in)
#dsd=torch.std(d_in)
d_in=(d_s-dmn)/dsd
train_x_e_1=torch.cat((x_in.reshape(n*n_e*nd,1),e_in.reshape(n*n_e*nd,1),d_in.reshape(n*n_e*nd,1)),1)
#print(train_x_e)
#randomly initialize state and context space from (0,1)


#FRT
t_x_a=x_a
x_a=x_a.numpy()

d_x=distance_matrix(x_a.reshape(n,1),x_a.reshape(n,1))#find distance between all pairs of states
#print(d_x)
td_x=d_x+100000*np.identity(n)#add a large value to all the diagonal values 
mind=np.min(td_x)
d_x=(1/mind)*d_x#scale the distance matrix with inverse of the minimum distance so that the new minimum distance becomes 1 
diam=np.max(d_x)#maximum distance
Delta=np.power(2,np.ceil(np.log2(diam)))#find the nearest power of 2 to the maximum distance

print(mind,'mind')

minepsit=14
#pool = mp.Pool(14)
#in1, in2, in3, in4, in5, in6, in7= zip(*pool.starmap(Tree.main,[(x_a,n, cseed) for cseed in range(14) ]))
#mineps=10000000
#for i in range(5):
#  cseed=i
#  lofrt=Tree.main(x_a,n,cseed)
#  if lofrt[0]<mineps:
#    mineps=lofrt[0]
#    clofrt=lofrt
#eps=mineps
#acroot=lofrt[1]
#p=lofrt[2]
#vld=lofrt[3]
#ld=lofrt[4]
#u7f=lofrt[5]
#id_matrix=lofrt[6]
#minepsit=14
pool = mp.Pool(n_cores)
out1, out2, out3, out4, out5, out6, out7, out8= zip(*pool.starmap(Tree.main,[(x_a,n, cseed) for cseed in range(24) ]))

dfdsind=np.argmin(np.asarray(out8))  
eps=out1[dfdsind]
acroot=out2[dfdsind]
p=out3[dfdsind]
vld=out4[dfdsind]
ld=out5[dfdsind]
u7f=out6[dfdsind]
id_matrix=out7[dfdsind]
dfds=out8[dfdsind]



#epsind=np.argmin(np.asarray(out1))  
#eps=out1[epsind]
#acroot=out2[epsind]
#p=out3[epsind]
#vld=out4[epsind]
#ld=out5[epsind]
#u7f=out6[epsind]
#id_matrix=out7[epsind]
#dfds=out8[epsind]





#print(np.amin(d_g+np.amax(d_g)*np.identity(n)))
#eps=(u7f[0]/(2*(2*maxl+np.log2(n))))*(4/7)#minimum cost according to paper---to be changed according to his suggestion


#
a=int(np.floor(np.random.uniform(0,1)*n))
for u in range(0,n):
  un=p[tuple(vld[u,ld])]
  un.rprob=0#initialize zero prob at all states except initial state
p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
rootv=acroot
rootv.rprob=Tree.rprob_calc_nodes(rootv)
rootv.weight=Tree.define_weight_postorder(rootv,u7f[0])
Tree.update_constants_postorder(rootv)
nodeu=p[tuple(vld[a,ld])]
rootv=acroot
treedict={}
kn_root=Tree.clonetree_preorder(rootv,treedict)
kn_p={}
for u in range(n):
    nodeu=p[tuple(vld[u,ld])]
    kn_nodeu=treedict[nodeu]
    kn_p[tuple(vld[u,ld])]=kn_nodeu
kn_acroot=kn_root
nodeu=p[tuple(vld[a,ld])]
kn_nodeu=kn_p[tuple(vld[a,ld])]
while nodeu.parent!=None:
    #print(nodeu.id,nodeu.rprob)
    #print(kn_nodeu.id,nodeu.rprob)
    nodeu=nodeu.parent
    kn_nodeu=kn_nodeu.parent
#print(Tree.print_leaves_postorder(rootv))
#print(Tree.print_leaves_postorder(rootcpy))
#print(rootv.leaves,rootcpy.leaves)
#eps=np.amin(d_g+np.amax(d_g)*np.identity(n))#minimum cost according to paper---to be changed according to his suggestion
avg_mc=np.sum(d_x*mind)/(n*n)
eps=eps
torch.pi = torch.acos(torch.zeros(1)).item() * 2

pl_freq=20
cost_ratio=1.5
#comp_cost=torch.zeros(T)
#act_cost=torch.zeros(T)
#rand_cost=torch.zeros(T)
#servopt_cost=torch.zeros(T)
ini_x=a
beta=2
st_ti=np.floor((np.random.uniform(0,1,1)*(len(vw)-T)))
#print(st_ti)
st_ti=int(st_ti)
st_ti=786
vw=np.transpose(vw)
ls_1=12.94


stobj=FRTOBJECT(acroot,kn_acroot,p,kn_p,vld,ld,train_x_e,train_x_e_1,mind,n,n_e,eps,avg_mc,u7f,id_matrix,cost_ratio,beta,d_x,vw,ini_x)
#print(FRTOBJECT.FRTOT(stobj,T))
#print(mp.Pool(mp.cpu_count()))
f_n=n_cores
gam=1
st_ti_val=[3186]
#gam_val=[0.5,0.75,1,2,4,6,8,10,12,14,18,20,26,30]
gam_val=[0.5,0.75,1,2,3,4,6,8,10]
for st_ti in st_ti_val:
    comp_cost_mvt_avg, comp_cost_mvt_std=[],[]
    act_cost_mvt_avg, act_cost_mvt_std=[],[]
    rand_cost_mvt_avg, rand_cost_mvt_std=[],[]
    servopt_cost_mvt_avg, servopt_cost_mvt_std=[],[]
    stat_cost_mvt_avg, stat_cost_mvt_std=[],[]
    pier_cost_mvt_avg, pier_cost_mvt_std=[],[]
    kn_act_cost_mvt_avg, kn_act_cost_mvt_std=[],[]

    comp_cost_serv_avg, comp_cost_serv_std=[],[]
    act_cost_serv_avg, act_cost_serv_std=[],[]
    rand_cost_serv_avg, rand_cost_serv_std=[],[]
    servopt_cost_serv_avg, servopt_cost_serv_std=[],[]
    stat_cost_serv_avg, stat_cost_serv_std=[],[]
    pier_cost_serv_avg, pier_cost_serv_std=[],[]
    kn_act_cost_serv_avg, kn_act_cost_serv_std=[],[]


    comp_cost_tot_avg, comp_cost_tot_std=[],[]
    act_cost_tot_avg, act_cost_tot_std=[],[]
    rand_cost_tot_avg, rand_cost_tot_std=[],[]
    servopt_cost_tot_avg, servopt_cost_tot_std=[],[]
    stat_cost_tot_avg, stat_cost_tot_std=[],[]
    pier_cost_tot_avg, pier_cost_tot_std=[],[]
    kn_act_cost_tot_avg, kn_act_cost_tot_std=[],[]

    comp_power_tot_avg, comp_power_tot_std=[],[]
    act_power_tot_avg, act_power_tot_std=[],[]
    servopt_power_tot_avg, servopt_power_tot_std=[],[]
    stat_power_tot_avg, stat_power_tot_std=[],[]
    pier_power_tot_avg, pier_power_tot_std=[],[]
    kn_act_power_tot_avg, kn_act_power_tot_std=[],[]


    comp_reg_serv_avg, comp_reg_serv_std=[],[]
    act_reg_serv_avg, act_reg_serv_std=[],[]
    rand_reg_serv_avg, rand_reg_serv_std=[],[]
    stat_reg_serv_avg, stat_reg_serv_std=[],[]
    pier_reg_serv_avg, pier_reg_serv_std=[],[]
    kn_act_reg_serv_avg, kn_act_reg_serv_std=[],[]

    for gam in gam_val:
        cost_list=runnerandplotter(n_cores,f_n,gam,stobj,T,pl_freq,title,beta,minepsit,st_ti,lat,lon,ls_1)
        comp_cost_mvt_avg.append(cost_list[2][0])
        comp_cost_mvt_std.append(cost_list[3][0])
        act_cost_mvt_avg.append(cost_list[2][1])
        act_cost_mvt_std.append(cost_list[3][1])
        rand_cost_mvt_avg.append(cost_list[2][2])
        rand_cost_mvt_std.append(cost_list[3][2])
        servopt_cost_mvt_avg.append(cost_list[2][3])
        servopt_cost_mvt_std.append(cost_list[3][3])
        stat_cost_mvt_avg.append(cost_list[2][4])
        stat_cost_mvt_std.append(cost_list[3][4])
        pier_cost_mvt_avg.append(cost_list[2][5])
        pier_cost_mvt_std.append(cost_list[3][5])
        kn_act_cost_mvt_avg.append(cost_list[2][6])
        kn_act_cost_mvt_std.append(cost_list[3][6])

        comp_cost_serv_avg.append(cost_list[4][0])
        comp_cost_serv_std.append(cost_list[5][0])
        act_cost_serv_avg.append(cost_list[4][1])
        act_cost_serv_std.append(cost_list[5][1])
        rand_cost_serv_avg.append(cost_list[4][2])
        rand_cost_serv_std.append(cost_list[5][2])
        servopt_cost_serv_avg.append(cost_list[4][3])
        servopt_cost_serv_std.append(cost_list[5][3])
        stat_cost_serv_avg.append(cost_list[4][4])
        stat_cost_serv_std.append(cost_list[5][4])
        pier_cost_serv_avg.append(cost_list[4][5])
        pier_cost_serv_std.append(cost_list[5][5])
        kn_act_cost_serv_avg.append(cost_list[4][6])
        kn_act_cost_serv_std.append(cost_list[5][6])

        comp_cost_tot_avg.append(cost_list[0][0])
        comp_cost_tot_std.append(cost_list[1][0])
        act_cost_tot_avg.append(cost_list[0][1])
        act_cost_tot_std.append(cost_list[1][1])
        rand_cost_tot_avg.append(cost_list[0][2])
        rand_cost_tot_std.append(cost_list[1][2])
        servopt_cost_tot_avg.append(cost_list[0][3])
        servopt_cost_tot_std.append(cost_list[1][3])
        stat_cost_tot_avg.append(cost_list[0][4])
        stat_cost_tot_std.append(cost_list[1][4])
        pier_cost_tot_avg.append(cost_list[0][5])
        pier_cost_tot_std.append(cost_list[1][5])
        kn_act_cost_tot_avg.append(cost_list[0][6])
        kn_act_cost_tot_std.append(cost_list[1][6])

        comp_reg_serv_avg.append(cost_list[6][0])
        comp_reg_serv_std.append(cost_list[7][0])
        act_reg_serv_avg.append(cost_list[6][1])
        act_reg_serv_std.append(cost_list[7][1])
        rand_reg_serv_avg.append(cost_list[6][2])
        rand_reg_serv_std.append(cost_list[7][2])
        stat_reg_serv_avg.append(cost_list[6][3])
        stat_reg_serv_std.append(cost_list[7][3])
        pier_reg_serv_avg.append(cost_list[6][4])
        pier_reg_serv_std.append(cost_list[7][4])
        kn_act_reg_serv_avg.append(cost_list[6][5])
        kn_act_reg_serv_std.append(cost_list[7][5])

        comp_power_tot_avg.append(cost_list[8][0])
        comp_power_tot_std.append(cost_list[9][0])
        act_power_tot_avg.append(cost_list[8][1])
        act_power_tot_std.append(cost_list[9][1])
        servopt_power_tot_avg.append(cost_list[8][2])
        servopt_power_tot_std.append(cost_list[9][2])
        stat_power_tot_avg.append(cost_list[8][3])
        stat_power_tot_std.append(cost_list[9][3])
        pier_power_tot_avg.append(cost_list[8][4])
        pier_power_tot_std.append(cost_list[9][4])
        kn_act_power_tot_avg.append(cost_list[8][5])
        kn_act_power_tot_std.append(cost_list[9][5])


    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars


        # Plot predictive means as blue linenp.a
        ax.errorbar(gam_val, np.asarray(comp_cost_tot_avg), yerr=np.asarray(comp_cost_tot_std),color='b',capsize=10, marker='o')
        ax.errorbar(gam_val, np.asarray(act_cost_tot_avg), yerr=np.asarray(act_cost_tot_std),color= 'c',capsize=10, marker='s')

        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_cost_tot_avg),yerr=np.asarray(servopt_cost_tot_std),color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_cost_tot_avg),yerr=np.asarray(stat_cost_tot_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_cost_tot_avg),yerr=np.asarray(pier_cost_tot_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_cost_tot_avg),yerr=np.asarray(kn_act_cost_tot_std),color= 'tab:orange',capsize=10, marker='*')
        #print('why?')

        # Shade between the lower and upper confidence bounds
        #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        #ax.set_ylim([-1, 2])
        #ax.legend([ compnm, algnm,'Known Service Optimal Cost', 'Stationary', 'Current State Optimal Cost', knalgnm])
        ax.legend([ compnm, algnm,septnm, 'Stationary', knalgnm],prop={'size': 25})
        plt.title('Total Cost (Service + Movement)',fontsize=25)
        #print('why?')
        #plt.ylabel('Cumulative Cost',fontsize=25)
        plt.xlabel(r'Range of $\rho$', fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_tot.csv' %(T,title,st_ti,lat,lon,ls_1), (np.asarray(comp_cost_tot_avg), np.asarray(comp_cost_tot_std), np.asarray(act_cost_tot_avg), np.asarray(act_cost_tot_std), np.asarray(servopt_cost_tot_avg), np.asarray(servopt_cost_tot_std), np.asarray(stat_cost_tot_avg), np.asarray(stat_cost_tot_std), np.asarray(kn_act_cost_tot_avg), np.asarray(kn_act_cost_tot_std)), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_tot.png' %(T,title,st_ti,lat,lon,ls_1))
        #print('why?')

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        #print('why?')

        # Plot predictive means as blue line
        ax.errorbar(gam_val, np.asarray(comp_cost_mvt_avg), yerr=np.asarray(comp_cost_mvt_std),color='b',capsize=10, marker='o')
        #print(gam_val,comp_cost_mvt_avg.numpy(),comp_cost_mvt_std.numpy())
        ax.errorbar(gam_val, np.asarray(act_cost_mvt_avg), yerr=np.asarray(act_cost_mvt_std),color= 'c',capsize=10, marker='s')

        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.ndumpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_cost_mvt_avg),yerr=np.asarray(servopt_cost_mvt_std),color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_cost_mvt_avg),yerr=np.asarray(stat_cost_mvt_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_cost_mvt_avg),yerr=np.asarray(pier_cost_mvt_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_cost_mvt_avg),yerr=np.asarray(kn_act_cost_mvt_std),color= 'tab:orange',capsize=10, marker='*')
        #print('why?')

        # Shade between the lower and upper confidence bounds
        #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        #ax.set_ylim([-1, 2])
        #ax.legend([ 'CGP-LCB', algnm,'Known Service Optimal Cost', 'Stationary', 'Current State Optimal Cost', knalgnm])
        ax.legend([ compnm, algnm,septnm, 'Stationary', knalgnm],prop={'size': 25})
        plt.title('Movement Energy Loss',fontsize=25 )
        #print('why?')
        #plt.ylabel('Cumulative Cost', fontsize=25)
        plt.xlabel(r'Range of $\rho$', fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_mvt.csv' %(T,title,st_ti,lat,lon,ls_1), (np.asarray(comp_cost_mvt_avg), np.asarray(comp_cost_mvt_std), np.asarray(act_cost_mvt_avg), np.asarray(act_cost_mvt_std), np.asarray(servopt_cost_mvt_avg), np.asarray(servopt_cost_mvt_std), np.asarray(stat_cost_mvt_avg), np.asarray(stat_cost_mvt_std), np.asarray(kn_act_cost_mvt_avg), np.asarray(kn_act_cost_mvt_std)), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_mvt.png' %(T,title,st_ti,lat,lon,ls_1))
        #print('why?')

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars


        # Plot predictive means as blue line
        ax.errorbar(gam_val, np.asarray(comp_cost_serv_avg), yerr=np.asarray(comp_cost_serv_std),color='b',capsize=10, marker='o')
        ax.errorbar(gam_val, np.asarray(act_cost_serv_avg), yerr=np.asarray(act_cost_serv_std),color= 'c',capsize=10, marker='s')

        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_cost_serv_avg),yerr=np.asarray(servopt_cost_serv_std),color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_cost_serv_avg),yerr=np.asarray(stat_cost_serv_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_cost_serv_avg),yerr=np.asarray(pier_cost_serv_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_cost_serv_avg),yerr=np.asarray(kn_act_cost_serv_std),color= 'tab:orange',capsize=10, marker='*')

        #print('why?')
        # Shade between the lower and upper confidence bounds
        #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        #ax.set_ylim([-1, 2])
        #ax.legend([ compnm, algnm,'Known Service Optimal Cost', 'Stationary', 'Current State Optimal Cost', knalgnm])
        ax.legend([ compnm, algnm,septnm, 'Stationary', knalgnm],prop={'size': 25})
        plt.title('Service Cost',fontsize=25)
        #plt.ylabel('Cumulative Cost', fontsize=25)
        plt.xlabel(r'Range of $\rho$', fontsize=25)
        #plt.show()
        #print('why?')
        np.savetxt('newdirfont1216extrem/data/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_serv.csv' %(T,title,st_ti,lat,lon,ls_1), (np.asarray(comp_cost_serv_avg), np.asarray(comp_cost_serv_std), np.asarray(act_cost_serv_avg), np.asarray(act_cost_serv_std), np.asarray(servopt_cost_serv_avg), np.asarray(servopt_cost_serv_std), np.asarray(stat_cost_serv_avg), np.asarray(stat_cost_serv_std), np.asarray(kn_act_cost_serv_avg), np.asarray(kn_act_cost_serv_std)), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_serv.png' %(T,title,st_ti,lat,lon,ls_1))
        #print('why?')

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars


        # Plot predictive means as blue line
        ax.errorbar(gam_val, np.asarray(comp_reg_serv_avg), yerr=np.asarray(comp_reg_serv_std),color='b',capsize=10, marker='o')
        ax.errorbar(gam_val, np.asarray(act_reg_serv_avg), yerr=np.asarray(act_reg_serv_std),color= 'c',capsize=10, marker='s')

        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        #ax.errorbar(gam_val, servopt_cost_serv_avg.numpy(),yerr=servopt_cost_serv_std.numpy(),color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_reg_serv_avg),yerr=np.asarray(stat_reg_serv_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_reg_serv_avg),yerr=np.asarray(pier_reg_serv_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_reg_serv_avg),yerr=np.asarray(kn_act_reg_serv_std),color= 'tab:orange',capsize=10, marker='*')
        #print('why?')

        # Shade between the lower and upper confidence bounds
        #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        #ax.set_ylim([-1, 2])
        #ax.legend([ compnm, algnm,'Known Service Optimal Cost', 'Stationary', 'Current State Optimal Cost', 'Known-MD'])
        ax.legend([ compnm, algnm, 'Stationary', knalgnm],prop={'size': 25})
        plt.title('Service Cost Regret', fontsize=25 )
        #print('why?')
        #plt.ylabel('Cumulative Regret',fontsize=25)
        plt.xlabel(r'Range of $\rho$', fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_reg.csv' %(T,title,st_ti,lat,lon,ls_1), (np.asarray(comp_reg_serv_avg), np.asarray(comp_reg_serv_std), np.asarray(act_reg_serv_avg), np.asarray(act_reg_serv_std),  np.asarray(stat_reg_serv_avg), np.asarray(stat_reg_serv_std), np.asarray(kn_act_reg_serv_avg), np.asarray(kn_act_reg_serv_std)), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_reg.png' %(T,title,st_ti,lat,lon,ls_1))
        #print('why?')


    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars


        # Plot predictive means as blue linenp.a
        ax.errorbar(gam_val, np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std),color='b',capsize=10, marker='o')
        ax.errorbar(gam_val, np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std),color= 'c',capsize=10, marker='s')

        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_power_tot_avg),yerr=np.asarray(servopt_power_tot_std),color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_power_tot_avg),yerr=np.asarray(stat_power_tot_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_power_tot_avg),yerr=np.asarray(pier_power_tot_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_power_tot_avg),yerr=np.asarray(kn_act_power_tot_std),color= 'tab:orange',capsize=10, marker='*')
        #print('why?')

        # Shade between the lower and upper confidence bounds
        #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        #ax.set_ylim([-1, 2])
        #ax.legend([ compnm, algnm,'Known Service Optimal Cost', 'Stationary', 'Current State Optimal Cost', knalgnm])
        ax.legend([ compnm, algnm,septnm, 'Stationary', knalgnm], prop={'size': 25})
        plt.title('Total Energy Generated', fontsize=25)
        #print('why?')
        #plt.ylabel('Cumulative Energy',fontsize=25)
        plt.xlabel(r'Range of $\rho$', fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.csv' %(T,title,st_ti,lat,lon,ls_1), (np.asarray(comp_power_tot_avg), np.asarray(comp_power_tot_std), np.asarray(act_power_tot_avg), np.asarray(act_power_tot_std), np.asarray(servopt_power_tot_avg), np.asarray(servopt_power_tot_std), np.asarray(stat_power_tot_avg), np.asarray(stat_power_tot_std), np.asarray(kn_act_power_tot_avg), np.asarray(kn_act_power_tot_std)), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.png' %(T,title,st_ti,lat,lon,ls_1))
        #print('why?')


