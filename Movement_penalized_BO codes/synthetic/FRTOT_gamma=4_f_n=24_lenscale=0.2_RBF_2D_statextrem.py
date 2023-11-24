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
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import cos, nonzero, random
from torch import serialization
import time
import os
#from gpytorch.distributions.multivariate_normal import MultivariateNormal

title=int(time.time())%123456789
n_cores=12
from FRThelper2D import Tree
from Bayopthelper_2D import FRTOBJECT
len_scale=0.2
kernel_choice=0
if kernel_choice==0:
    k=RBFKernel()
    kout="rbf"
else:
    k=MaternKernel(len_scale=len_scale)
    kout="matern"
class RKHSModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(RKHSModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = k
    
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


n_1=20
n_e=40
dim=2
n=pow(n_1,dim)
x_a_2D=torch.linspace(0,1,n_1)
y_a_2D=torch.linspace(0,1,n_1)
x_a_2D_tr,y_a_2D_tr=torch.meshgrid(x_a_2D,y_a_2D)
x_a=torch.cat((x_a_2D_tr.reshape(n_1*n_1,1),y_a_2D_tr.reshape(n_1*n_1,1)),1)
#print(x_a)
e=torch.rand(n_e)
x_s,y_s,e_s=torch.meshgrid(x_a_2D,y_a_2D,e)
train_x_e=torch.cat((x_s.reshape(n*n_e,1),y_s.reshape(n*n_e,1),e_s.reshape(n*n_e,1)),1)
#randomly initialize state and context space from (0,1)
#print(train_x_e)

#FRT
t_x_a=x_a
x_a=x_a.numpy()
map_2d={i:x_a[i] for i in range(n)}
#print(map_2d)
d_x=distance_matrix(x_a,x_a)#find distance between all pairs of states
#print(d_x)
td_x=d_x+100000*np.identity(n)#add a large value to all the diagonal values 
mind=np.min(td_x)
d_x=(1/mind)*d_x#scale the distance matrix with inverse of the minimum distance so that the new minimum distance becomes 1 
diam=np.max(d_x)#maximum distance
Delta=np.power(2,np.ceil(np.log2(diam)))#find the nearest power of 2 to the maximum distance



minepsit=24
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
out1, out2, out3, out4, out5, out6, out7, out8= zip(*pool.starmap(Tree.main,[(x_a,n, cseed,dim) for cseed in range(minepsit) ]))
dfdsind=np.argmin(np.asarray(out8))  
eps=out1[dfdsind]
acroot=out2[dfdsind]
p=out3[dfdsind]
vld=out4[dfdsind]
ld=out5[dfdsind]
u7f=out6[dfdsind]
id_matrix=out7[dfdsind]
dfds=out8[dfdsind]





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

T=500
pl_freq=20
cost_ratio=1.5
#comp_cost=torch.zeros(T)
#act_cost=torch.zeros(T)
#rand_cost=torch.zeros(T)
#servopt_cost=torch.zeros(T)
ini_x=a
beta=2
#len_scale=0.2
#k=MaternKernel()
stobj=FRTOBJECT(acroot,kn_acroot,p,kn_p,vld,ld,train_x_e,mind,n,n_e,eps,avg_mc,u7f,id_matrix,cost_ratio,beta,d_x,len_scale,k,map_2d,ini_x)
#print(FRTOBJECT.FRTOT(stobj,T))
#print(mp.Pool(mp.cpu_count()))
pool = mp.Pool(n_cores)
f_n=24
gam=4
k.__has_lengthscale=True
k.lengthscale=0.2
print(k.lengthscale,'lengthscale')
#k=RBFKernel(lengthscale=len_scale)
#cov_em=k(train_x_e,train_x_e).evaluate()+0.001*torch.eye(n*n_e)#noise aded to make sure matrix is positive definite
out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14 = zip(*pool.starmap(FRTOBJECT.FRTOT_fixgamma,[(stobj,T,gam,i) for i in range(0,f_n)]))
comp_cost_mvt=torch.stack(out1)
act_cost_mvt=torch.stack(out2)
rand_cost_mvt=torch.stack(out3)
servopt_cost_mvt=torch.stack(out4)
stat_cost_mvt=torch.stack(out5)
pier_cost_mvt=torch.stack(out11)
kn_act_cost_mvt=torch.stack(out13)
comp_cost_serv=torch.stack(out6)
act_cost_serv=torch.stack(out7)
rand_cost_serv=torch.stack(out8)
servopt_cost_serv=torch.stack(out9)
stat_cost_serv=torch.stack(out10)
pier_cost_serv=torch.stack(out12)
kn_act_cost_serv=torch.stack(out14)




cclid=torch.topk(stat_cost_serv[:,T-1],f_n-2).indices
ccsid=torch.topk(stat_cost_serv[:,T-1],f_n-2,largest=False).indices
#print(cclid,ccsid)
#print(np.intersect1d(cclid,ccsid))
ne_ind=np.intersect1d(ccsid,cclid)
comp_cost_mvt=comp_cost_mvt[ne_ind,:]
act_cost_mvt=act_cost_mvt[ne_ind,:]
stat_cost_mvt=stat_cost_mvt[ne_ind,:]
servopt_cost_mvt=servopt_cost_mvt[ne_ind,:]
rand_cost_mvt=rand_cost_mvt[ne_ind,:]
pier_cost_mvt=pier_cost_mvt[ne_ind,:]
kn_act_cost_mvt=kn_act_cost_mvt[ne_ind,:]
comp_cost_serv=comp_cost_serv[ne_ind,:]
act_cost_serv=act_cost_serv[ne_ind,:]
stat_cost_serv=stat_cost_serv[ne_ind,:]
servopt_cost_serv=servopt_cost_serv[ne_ind,:]
rand_cost_serv=rand_cost_serv[ne_ind,:]
pier_cost_serv=pier_cost_serv[ne_ind,:]
kn_act_cost_serv=kn_act_cost_serv[ne_ind,:]
#print(comp_cost_mvt,comp_cost_mvt.shape)
comp_cost_mvt_avg, comp_cost_mvt_std=comp_cost_mvt.mean(0),comp_cost_mvt.std(0)
act_cost_mvt_avg, act_cost_mvt_std=act_cost_mvt.mean(0),act_cost_mvt.std(0)
rand_cost_mvt_avg, rand_cost_mvt_std=rand_cost_mvt.mean(0),rand_cost_mvt.std(0)
servopt_cost_mvt_avg, servopt_cost_mvt_std=servopt_cost_mvt.mean(0),servopt_cost_mvt.std(0)
stat_cost_mvt_avg, stat_cost_mvt_std=stat_cost_mvt.mean(0),stat_cost_mvt.std(0)
pier_cost_mvt_avg, pier_cost_mvt_std=pier_cost_mvt.mean(0),pier_cost_mvt.std(0)
kn_act_cost_mvt_avg, kn_act_cost_mvt_std=kn_act_cost_mvt.mean(0),kn_act_cost_mvt.std(0)

comp_cost_serv_avg, comp_cost_serv_std=comp_cost_serv.mean(0),comp_cost_serv.std(0)
act_cost_serv_avg, act_cost_serv_std=act_cost_serv.mean(0),act_cost_serv.std(0)
rand_cost_serv_avg, rand_cost_serv_std=rand_cost_serv.mean(0),rand_cost_serv.std(0)
servopt_cost_serv_avg, servopt_cost_serv_std=servopt_cost_serv.mean(0),servopt_cost_serv.std(0)
stat_cost_serv_avg, stat_cost_serv_std=stat_cost_serv.mean(0),stat_cost_serv.std(0)
pier_cost_serv_avg, pier_cost_serv_std=pier_cost_serv.mean(0),pier_cost_serv.std(0)
kn_act_cost_serv_avg, kn_act_cost_serv_std=kn_act_cost_serv.mean(0),kn_act_cost_serv.std(0)


comp_cost_tot_avg, comp_cost_tot_std=(comp_cost_mvt+comp_cost_serv).mean(0),(comp_cost_mvt+comp_cost_serv).std(0)
act_cost_tot_avg, act_cost_tot_std=(act_cost_mvt+act_cost_serv).mean(0),(act_cost_serv+act_cost_mvt).std(0)
rand_cost_tot_avg, rand_cost_tot_std=(rand_cost_mvt+rand_cost_serv).mean(0),(rand_cost_serv+rand_cost_mvt).std(0)
servopt_cost_tot_avg, servopt_cost_tot_std=(servopt_cost_mvt+servopt_cost_serv).mean(0),(servopt_cost_mvt+servopt_cost_serv).std(0)
stat_cost_tot_avg, stat_cost_tot_std=(stat_cost_mvt+stat_cost_serv).mean(0),(stat_cost_serv+stat_cost_mvt).std(0)
pier_cost_tot_avg, pier_cost_tot_std=(pier_cost_mvt+pier_cost_serv).mean(0),(pier_cost_serv+pier_cost_mvt).std(0)
kn_act_cost_tot_avg, kn_act_cost_tot_std=(kn_act_cost_mvt+kn_act_cost_serv).mean(0),(kn_act_cost_serv+kn_act_cost_mvt).std(0)


comp_reg_serv_avg, comp_reg_serv_std=(comp_cost_serv-servopt_cost_serv).mean(0),(comp_cost_serv-servopt_cost_serv).std(0)
act_reg_serv_avg, act_reg_serv_std=(act_cost_serv-servopt_cost_serv).mean(0),(act_cost_serv-servopt_cost_serv).std(0)
rand_reg_serv_avg, rand_reg_serv_std=(rand_cost_serv-servopt_cost_serv).mean(0),(rand_cost_serv-servopt_cost_serv).std(0)
stat_reg_serv_avg, stat_reg_serv_std=(stat_cost_serv-servopt_cost_serv).mean(0),(stat_cost_serv-servopt_cost_serv).std(0)
pier_reg_serv_avg, pier_reg_serv_std=(pier_cost_serv-servopt_cost_serv).mean(0),(pier_cost_serv-servopt_cost_serv).std(0)
kn_act_reg_serv_avg, kn_act_reg_serv_std=(kn_act_cost_serv-servopt_cost_serv).mean(0),(kn_act_cost_serv-servopt_cost_serv).std(0)

print(comp_cost_tot_avg[T-1],act_cost_tot_avg[T-1],rand_cost_tot_avg[T-1],servopt_cost_tot_avg[T-1],stat_cost_tot_avg[T-1],pier_cost_tot_avg[T-1], kn_act_cost_tot_avg[T-1] )
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    
   
    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_tot_avg.numpy()[0::int(T/pl_freq)], yerr=comp_cost_tot_std.numpy()[0::int(T/pl_freq)],color='b',capsize=10, marker='o')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_tot_avg.numpy()[0::int(T/pl_freq)], yerr=act_cost_tot_std.numpy()[0::int(T/pl_freq)],color= 'c',capsize=10, marker='s')

    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=servopt_cost_tot_std.numpy()[0::int(T/pl_freq)],color= 'y',capsize=10, marker='^')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=stat_cost_tot_std.numpy()[0::int(T/pl_freq)],color= 'm',capsize=10, marker='d')
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=pier_cost_tot_std.numpy()[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=kn_act_cost_tot_std.numpy()[0::int(T/pl_freq)],color= 'tab:orange',capsize=10, marker='*')
    

    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    #ax.legend([ 'CGP-LCB', 'CGP-LCB-MD','Known Service Optimal Cost', 'Stationary', 'Current State Optimal Cost', 'Known-MD'])
    ax.legend([ 'CGP-LCB', 'MD-CGP-LCB','CGP-Known', 'Stationary', 'MD-Known'],  prop={'size': 25})
    plt.title('Average Total Cost (Service + Movement)', fontsize=25)
    plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=25)
    #plt.show()
    np.savetxt('data/t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_tot_2D.csv' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale), (comp_cost_tot_avg.numpy(), comp_cost_tot_std.numpy(), act_cost_tot_avg.numpy(), act_cost_tot_std.numpy(), servopt_cost_tot_avg.numpy(), servopt_cost_tot_std.numpy(), stat_cost_tot_avg.numpy(), stat_cost_tot_std.numpy(), kn_act_cost_tot_avg.numpy(), kn_act_cost_tot_std.numpy()), delimiter=',')

    plt.savefig('t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_tot_2D.png' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale))



with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
   
    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_serv_avg.numpy()[0::int(T/pl_freq)], yerr=comp_cost_serv_std.numpy()[0::int(T/pl_freq)],color='b',capsize=10, marker='o')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_serv_avg.numpy()[0::int(T/pl_freq)], yerr=act_cost_serv_std.numpy()[0::int(T/pl_freq)],color= 'c',capsize=10, marker='s')

    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_serv_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_serv_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg.numpy()[0::int(T/pl_freq)],yerr=servopt_cost_serv_std.numpy()[0::int(T/pl_freq)],color= 'y',capsize=10, marker='^')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg.numpy()[0::int(T/pl_freq)],yerr=stat_cost_serv_std.numpy()[0::int(T/pl_freq)],color= 'm',capsize=10, marker='d')
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg.numpy()[0::int(T/pl_freq)],yerr=pier_cost_serv_std.numpy()[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg.numpy()[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std.numpy()[0::int(T/pl_freq)],color= 'tab:orange',capsize=10, marker='*')
    

    
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    #ax.legend([ 'Comparison cost', 'Actual cost','Known Service Optimal Cost','Stationary Cost', 'Current State Optimal Cost', 'Known Actual Cost'])
    ax.legend([ 'CGP-LCB', 'MD-CGP-LCB','CGP-Known', 'Stationary', 'MD-Known'],  prop={'size': 25})
    plt.title('Average Service Cost', fontsize=25)
    plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=25)
    #plt.show()
    np.savetxt('data/t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_serv_2D.csv' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale), (comp_cost_serv_avg.numpy(), comp_cost_serv_std.numpy(), act_cost_serv_avg.numpy(), act_cost_serv_std.numpy(), servopt_cost_serv_avg.numpy(), servopt_cost_serv_std.numpy(), stat_cost_serv_avg.numpy(), stat_cost_serv_std.numpy(), kn_act_cost_serv_avg.numpy(), kn_act_cost_serv_std.numpy()), delimiter=',')

    plt.savefig('t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_serv_2D.png' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale))


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
   
    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_mvt_avg.numpy()[0::int(T/pl_freq)], yerr=comp_cost_mvt_std.numpy()[0::int(T/pl_freq)],color='b',capsize=10, marker='o')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_mvt_avg.numpy()[0::int(T/pl_freq)], yerr=act_cost_mvt_std.numpy()[0::int(T/pl_freq)],color= 'c',capsize=10, marker='s')

    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_mvt_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_mvt_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_mvt_avg.numpy()[0::int(T/pl_freq)],yerr=servopt_cost_mvt_std.numpy()[0::int(T/pl_freq)],color= 'y',capsize=10, marker='^')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_mvt_avg.numpy()[0::int(T/pl_freq)],yerr=stat_cost_mvt_std.numpy()[0::int(T/pl_freq)],color= 'm',capsize=10, marker='d')
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_mvt_avg.numpy()[0::int(T/pl_freq)],yerr=pier_cost_mvt_std.numpy()[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_mvt_avg.numpy()[0::int(T/pl_freq)],yerr=kn_act_cost_mvt_std.numpy()[0::int(T/pl_freq)],color= 'tab:orange',capsize=10, marker='*')
    
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    #ax.legend([ 'Comparison cost', 'Actual cost','Known Service Optimal Cost', 'Stationary Cost', 'Current State Optimal Cost', 'Known Actual Cost'])
    #plt.title('T=800,f_n=49,n=500--Movement Cost')
    ax.legend([ 'CGP-LCB', 'MD-CGP-LCB','CGP-Known', 'Stationary', 'MD-Known'],  prop={'size': 25})
    plt.title('Average Movement Cost', fontsize=25)
    plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=25)
    #plt.show()
    #plt.savefig('t=800_n=500_mvt_avg.png')
    np.savetxt('data/t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_mvt_2D.csv' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale), (comp_cost_mvt_avg.numpy(), comp_cost_mvt_std.numpy(), act_cost_mvt_avg.numpy(), act_cost_mvt_std.numpy(), servopt_cost_mvt_avg.numpy(), servopt_cost_mvt_std.numpy(), stat_cost_mvt_avg.numpy(), stat_cost_mvt_std.numpy(), kn_act_cost_mvt_avg.numpy(), kn_act_cost_mvt_std.numpy()), delimiter=',')

    plt.savefig('t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_mvt_2D.png' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale))


     
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    
   
    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_reg_serv_avg.numpy()[0::int(T/pl_freq)], yerr=comp_reg_serv_std.numpy()[0::int(T/pl_freq)],color='b',capsize=10, marker='o')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_reg_serv_avg.numpy()[0::int(T/pl_freq)], yerr=act_reg_serv_std.numpy()[0::int(T/pl_freq)],color= 'c',capsize=10, marker='s')

    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_reg_serv_avg.numpy()[0::int(T/pl_freq)],yerr=rand_reg_serv_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_reg_serv_avg.numpy()[0::int(T/pl_freq)],yerr=stat_reg_serv_std.numpy()[0::int(T/pl_freq)], color='m',capsize=10, marker='d' )
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_reg_serv_avg.numpy()[0::int(T/pl_freq)],yerr=pier_reg_serv_std.numpy()[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_reg_serv_avg.numpy()[0::int(T/pl_freq)],yerr=kn_act_reg_serv_std.numpy()[0::int(T/pl_freq)],color= 'tab:orange',capsize=10, marker='*')
    

    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    #ax.legend([ 'Comparison cost', 'Actual cost','Known Service Optimal Cost','Stationary Cost', 'Current State Optimal Cost', 'Known Actual Cost'])
    ax.legend([ 'CGP-LCB', 'MD-CGP-LCB', 'Stationary', 'MD-Known'], prop={'size': 25})
    plt.title('Average Service Cost Regret', fontsize=25)
    
    plt.ylabel('Cumulative Regret', fontsize=25)
    plt.xlabel('Timesteps', fontsize=25)
    #plt.show()
    np.savetxt('data/t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_reg_2D.csv' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale), (comp_reg_serv_avg.numpy(), comp_reg_serv_std.numpy(), act_reg_serv_avg.numpy(), act_reg_serv_std.numpy(),  stat_reg_serv_avg.numpy(), stat_reg_serv_std.numpy(), kn_act_reg_serv_avg.numpy(), kn_act_reg_serv_std.numpy()), delimiter=',')

    plt.savefig('t=%d_f_n=%d_n=%d_gam=%f_Kernel=%s_f_rep=3_%d_mineps=%f_c_n=%d_len_scale=%f_reg_2D.png' %(T,f_n, n,gam,kout,title,eps,minepsit,len_scale))



