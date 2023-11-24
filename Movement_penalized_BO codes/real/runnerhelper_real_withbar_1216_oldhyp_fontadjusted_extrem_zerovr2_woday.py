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

from FRThelper import Tree
from Bayopthelper_real_2_wind_oldhyp_zerovr2_woday import FRTOBJECT
algnm='MD-GP'
septnm='MinC-Known'
compnm='CGP-LCB'
knalgnm='MD-Known'

def runnerandplotter(n_cores,f_n,gam,stobj,T,pl_freq,title,beta,minepsit,st_ti,lat,lon,ls_1):
    pool = mp.Pool(n_cores)
    n=stobj.n
    eps=stobj.eps

    len_scale=0.2
    #k=RBFKernel(lengthscale=len_scale)
    #cov_em=k(train_x_e,train_x_e).evaluate()+0.001*torch.eye(n*n_e)#noise aded to make sure matrix is positive definite
    out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16, out17, out18, out19, out20 = zip(*pool.starmap(FRTOBJECT.FRTOT_fixgamma,[(stobj,T,gam,i,st_ti,ls_1) for i in range(0,f_n)]))
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

    comp_power_serv=torch.stack(out15)
    act_power_serv=torch.stack(out16)
    servopt_power_serv=torch.stack(out17)
    stat_power_serv=torch.stack(out18)
    pier_power_serv=torch.stack(out19)
    kn_act_power_serv=torch.stack(out20)



    extrem=3
    cclid=torch.topk(stat_power_serv[:,T-1],f_n-extrem).indices
    ccsid=torch.topk(stat_power_serv[:,T-1],f_n-extrem,largest=False).indices
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


    comp_power_serv=comp_power_serv[ne_ind,:]
    act_power_serv=act_power_serv[ne_ind,:]
    stat_power_serv=stat_power_serv[ne_ind,:]
    servopt_power_serv=servopt_power_serv[ne_ind,:]
    pier_power_serv=pier_power_serv[ne_ind,:]
    kn_act_power_serv=kn_act_power_serv[ne_ind,:]
    #print(comp_cost_mvt[:,T-1])
    np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_boxplt.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (comp_cost_mvt[:,T-1].numpy(),comp_cost_serv[:,T-1].numpy(),comp_power_serv[:,T-1].numpy(), act_cost_mvt[:,T-1].numpy(),act_cost_serv[:,T-1].numpy(),act_power_serv[:,T-1].numpy(), servopt_cost_mvt[:,T-1].numpy(), servopt_cost_serv[:,T-1].numpy(), servopt_power_serv[:,T-1].numpy(), stat_cost_mvt[:,T-1].numpy(), stat_cost_serv[:,T-1].numpy(), stat_power_serv[:,T-1].numpy(), kn_act_cost_mvt[:,T-1].numpy(), kn_act_cost_serv[:,T-1].numpy(), kn_act_power_serv[:,T-1].numpy()), delimiter=',')
    #print(comp_cost_mvt[:,T-1])


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

    
    
    comp_power_tot_avg, comp_power_tot_std=(-comp_cost_mvt+comp_power_serv).mean(0),(-comp_cost_mvt+comp_power_serv).std(0)
    act_power_tot_avg, act_power_tot_std=(-act_cost_mvt+act_power_serv).mean(0),(-act_cost_mvt+act_power_serv).std(0)
    #rand_power_tot_avg, rand_cost_tot_std=(rand_cost_mvt+rand_cost_serv).mean(0),(rand_cost_serv+rand_cost_mvt).std(0)
    servopt_power_tot_avg, servopt_power_tot_std=(-servopt_cost_mvt+servopt_power_serv).mean(0),(-servopt_cost_mvt+servopt_power_serv).std(0)
    stat_power_tot_avg, stat_power_tot_std=(-stat_cost_mvt+stat_power_serv).mean(0),(-stat_cost_mvt+stat_power_serv).std(0)
    pier_power_tot_avg, pier_power_tot_std=(-pier_cost_mvt+pier_power_serv).mean(0),(-pier_cost_mvt+pier_power_serv).std(0)
    kn_act_power_tot_avg, kn_act_power_tot_std=(-kn_act_cost_mvt+kn_act_power_serv).mean(0),(-kn_act_cost_mvt+kn_act_power_serv).std(0)

       
    
    tot_avg= [comp_cost_tot_avg[T-1],act_cost_tot_avg[T-1],rand_cost_tot_avg[T-1],servopt_cost_tot_avg[T-1],stat_cost_tot_avg[T-1],pier_cost_tot_avg[T-1], kn_act_cost_tot_avg[T-1] ]
    tot_std= [comp_cost_tot_std[T-1],act_cost_tot_std[T-1],rand_cost_tot_std[T-1],servopt_cost_tot_std[T-1],stat_cost_tot_std[T-1],pier_cost_tot_std[T-1], kn_act_cost_tot_std[T-1] ]
    mvt_avg= [comp_cost_mvt_avg[T-1],act_cost_mvt_avg[T-1],rand_cost_mvt_avg[T-1],servopt_cost_mvt_avg[T-1],stat_cost_mvt_avg[T-1],pier_cost_mvt_avg[T-1], kn_act_cost_mvt_avg[T-1] ]
    mvt_std= [comp_cost_mvt_std[T-1],act_cost_mvt_std[T-1],rand_cost_mvt_std[T-1],servopt_cost_mvt_std[T-1],stat_cost_mvt_std[T-1],pier_cost_mvt_std[T-1], kn_act_cost_mvt_std[T-1] ]
    serv_avg= [comp_cost_serv_avg[T-1],act_cost_serv_avg[T-1],rand_cost_serv_avg[T-1],servopt_cost_serv_avg[T-1],stat_cost_serv_avg[T-1],pier_cost_serv_avg[T-1], kn_act_cost_serv_avg[T-1] ]
    serv_std= [comp_cost_serv_std[T-1],act_cost_serv_std[T-1],rand_cost_serv_std[T-1],servopt_cost_serv_std[T-1],stat_cost_serv_std[T-1],pier_cost_serv_std[T-1], kn_act_cost_serv_std[T-1] ]
    reg_avg= [comp_reg_serv_avg[T-1],act_reg_serv_avg[T-1],rand_reg_serv_avg[T-1],stat_reg_serv_avg[T-1],pier_reg_serv_avg[T-1], kn_act_reg_serv_avg[T-1] ]
    reg_std= [comp_reg_serv_std[T-1],act_reg_serv_std[T-1],rand_reg_serv_std[T-1],stat_reg_serv_std[T-1],pier_reg_serv_std[T-1], kn_act_reg_serv_std[T-1] ]
    
    ptot_avg= [comp_power_tot_avg[T-1],act_power_tot_avg[T-1],servopt_power_tot_avg[T-1],stat_power_tot_avg[T-1],pier_power_tot_avg[T-1], kn_act_power_tot_avg[T-1] ]
    ptot_std= [comp_power_tot_std[T-1],act_power_tot_std[T-1],servopt_power_tot_std[T-1],stat_power_tot_std[T-1],pier_power_tot_std[T-1], kn_act_power_tot_std[T-1] ]
    
    #tot_avg_dt= [comp_cost_tot_avg.numpy(),act_cost_tot_avg.numpy(),rand_cost_tot_avg.numpy(),servopt_cost_tot_avg.numpy(),stat_cost_tot_avg.numpy(),pier_cost_tot_avg.numpy(), kn_act_cost_tot_avg.numpy() ]
    #tot_std_dt= [comp_cost_tot_std.numpy(),act_cost_tot_std.numpy(),rand_cost_tot_std.numpy(),servopt_cost_tot_std.numpy(),stat_cost_tot_std.numpy(),pier_cost_tot_std.numpy(), kn_act_cost_tot_std.numpy() ]
    #mvt_avg_dt= [comp_cost_mvt_avg.numpy(),act_cost_mvt_avg.numpy(),rand_cost_mvt_avg.numpy(),servopt_cost_mvt_avg.numpy(),stat_cost_mvt_avg.numpy(),pier_cost_mvt_avg.numpy(), kn_act_cost_mvt_avg.numpy() ]
    #mvt_std_dt= [comp_cost_mvt_std.numpy(),act_cost_mvt_std.numpy(),rand_cost_mvt_std.numpy(),servopt_cost_mvt_std.numpy(),stat_cost_mvt_std.numpy(),pier_cost_mvt_std.numpy(), kn_act_cost_mvt_std.numpy() ]
    #serv_avg_dt= [comp_cost_serv_avg.numpy(),act_cost_serv_avg.numpy(),rand_cost_serv_avg.numpy(),servopt_cost_serv_avg.numpy(),stat_cost_serv_avg.numpy(),pier_cost_serv_avg.numpy(), kn_act_cost_serv_avg.numpy() ]
    #serv_std_dt= [comp_cost_serv_std.numpy(),act_cost_serv_std.numpy(),rand_cost_serv_std.numpy(),servopt_cost_serv_std.numpy(),stat_cost_serv_std.numpy(),pier_cost_serv_std.numpy(), kn_act_cost_serv_std.numpy() ]
    #reg_avg_dt= [comp_reg_serv_avg.numpy(),act_reg_serv_avg.numpy(),rand_reg_serv_avg.numpy(),stat_reg_serv_avg.numpy(),pier_reg_serv_avg.numpy(), kn_act_reg_serv_avg.numpy() ]
    #reg_std_dt= [comp_reg_serv_std.numpy(),act_reg_serv_std.numpy(),rand_reg_serv_std.numpy(),stat_reg_serv_std.numpy(),pier_reg_serv_std.numpy(), kn_act_reg_serv_std.numpy() ]
    
    #ptot_avg_dt= [comp_power_tot_avg.numpy(),act_power_tot_avg.numpy(),servopt_power_tot_avg.numpy(),stat_power_tot_avg.numpy(),pier_power_tot_avg.numpy(), kn_act_power_tot_avg.numpy() ]
    #ptot_std_dt= [comp_power_tot_std.numpy(),act_power_tot_std.numpy(),servopt_power_tot_std.numpy(),stat_power_tot_std.numpy(),pier_power_tot_std.numpy(), kn_act_power_tot_std.numpy() ]
    
    #al_dt=[np.asarray(tot_avg_dt),np.asarray(tot_std_dt),np.asarray(mvt_avg_dt),np.asarray(mvt_std_dt),np.asarray(serv_avg_dt),np.asarray(serv_std_dt),np.asarray(reg_avg_dt),np.asarray(reg_std_dt),np.asarray(ptot_avg_dt),np.asarray(ptot_std_dt)]
    #np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_all.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (np.asarray(al_dt)), delimiter=',')

    
    print(comp_cost_tot_avg[T-1],act_cost_tot_avg[T-1],rand_cost_tot_avg[T-1],servopt_cost_tot_avg[T-1],stat_cost_tot_avg[T-1],pier_cost_tot_avg[T-1], kn_act_cost_tot_avg[T-1] )
    print(comp_power_tot_avg[T-1],act_power_tot_avg[T-1],servopt_power_tot_avg[T-1],stat_power_tot_avg[T-1],pier_power_tot_avg[T-1], kn_act_power_tot_avg[T-1] )
    
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
        ax.legend([ compnm, algnm ,septnm, 'Stationary', knalgnm], prop={'size': 22})
        plt.title('Total Cost (Service + Movement)',fontsize=25)
        ##plt.ylabel('Cumulative cost',fontsize=25)
        plt.xlabel('Timesteps',fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_tot.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (comp_cost_tot_avg.numpy(), comp_cost_tot_std.numpy(), act_cost_tot_avg.numpy(), act_cost_tot_std.numpy(), servopt_cost_tot_avg.numpy(), servopt_cost_tot_std.numpy(), stat_cost_tot_avg.numpy(), stat_cost_tot_std.numpy(), kn_act_cost_tot_avg.numpy(), kn_act_cost_tot_std.numpy()), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_tot.png' %(T,gam,title,st_ti,lat,lon,ls_1))



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
        ax.legend([ compnm, algnm ,septnm, 'Stationary', knalgnm], prop={'size': 22})
        plt.title('Service Cost', fontsize=25 )
        #plt.ylabel('Cumulative cost', fontsize=25)
        plt.xlabel('Timesteps', fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_serv.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (comp_cost_serv_avg.numpy(), comp_cost_serv_std.numpy(), act_cost_serv_avg.numpy(), act_cost_serv_std.numpy(), servopt_cost_serv_avg.numpy(), servopt_cost_serv_std.numpy(), stat_cost_serv_avg.numpy(), stat_cost_serv_std.numpy(), kn_act_cost_serv_avg.numpy(), kn_act_cost_serv_std.numpy()), delimiter=',')


        plt.savefig('newdirfont1216extrem/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_serv.png' %(T,gam,title,st_ti,lat,lon,ls_1))


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
        ax.legend([ compnm, algnm ,septnm, 'Stationary',knalgnm], prop={'size': 22})
        #plt.title('T=800,f_n=49,n=500--Movement Cost')
        plt.title('Movement Energy Loss', fontsize=25)
        #plt.ylabel('Cumulative Cost',fontsize=25)
        plt.xlabel('Timesteps',fontsize=25)
        #plt.show()
        #plt.savefig('t=800_n=500_mvt_avg.png')
        np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_mvt.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (comp_cost_mvt_avg.numpy(), comp_cost_mvt_std.numpy(), act_cost_mvt_avg.numpy(), act_cost_mvt_std.numpy(), servopt_cost_mvt_avg.numpy(), servopt_cost_mvt_std.numpy(), stat_cost_mvt_avg.numpy(), stat_cost_mvt_std.numpy(), kn_act_cost_mvt_avg.numpy(), kn_act_cost_mvt_std.numpy()), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_mvt.png' %(T,gam,title,st_ti,lat,lon,ls_1))


        
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
        ax.legend([ compnm, algnm ,septnm, 'Stationary', knalgnm], prop={'size': 22})
        plt.title('Service Cost Regret', fontsize=25)
        
        #plt.ylabel('Cumulative Regret', fontsize=25)
        plt.xlabel('Timesteps',fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_reg.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (comp_reg_serv_avg.numpy(), comp_reg_serv_std.numpy(), act_reg_serv_avg.numpy(), act_reg_serv_std.numpy(),  stat_reg_serv_avg.numpy(), stat_reg_serv_std.numpy(), kn_act_reg_serv_avg.numpy(), kn_act_reg_serv_std.numpy()), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_reg.png' %(T,gam,title,st_ti,lat,lon,ls_1))
    
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        
    
        # Plot predictive means as blue line
        ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_power_tot_avg.numpy()[0::int(T/pl_freq)], yerr=comp_power_tot_std.numpy()[0::int(T/pl_freq)],color='b',capsize=10, marker='o')
        ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_power_tot_avg.numpy()[0::int(T/pl_freq)], yerr=act_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'c',capsize=10, marker='s')

        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_power_tot_avg.numpy()[0::int(T/pl_freq)],yerr=servopt_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'y',capsize=10, marker='^')
        ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_power_tot_avg.numpy()[0::int(T/pl_freq)],yerr=stat_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'm',capsize=10, marker='d')
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_power_tot_avg.numpy()[0::int(T/pl_freq)],yerr=pier_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
        ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_power_tot_avg.numpy()[0::int(T/pl_freq)],yerr=kn_act_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'tab:orange',capsize=10, marker='*')
        

        # Shade between the lower and upper confidence bounds
        #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        #ax.set_ylim([-1, 2])
        ax.legend([ compnm, algnm ,septnm, 'Stationary', knalgnm], prop={'size': 22})
        plt.title('Total Energy Generated', fontsize=25)
        #plt.ylabel('Cumulative Energy', fontsize=25)
        plt.xlabel('Timesteps',fontsize=25)
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (comp_power_tot_avg.numpy(), comp_power_tot_std.numpy(), act_power_tot_avg.numpy(), act_power_tot_std.numpy(), servopt_power_tot_avg.numpy(), servopt_power_tot_std.numpy(), stat_power_tot_avg.numpy(), stat_power_tot_std.numpy(), kn_act_power_tot_avg.numpy(), kn_act_power_tot_std.numpy()), delimiter=',')

        plt.savefig('newdirfont1216extrem/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.png' %(T,gam,title,st_ti,lat,lon,ls_1))

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(3)

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.bar(X + 0.00, comp_power_tot_avg.numpy()[int(T/3)-1::int(T/3)], color = 'b', width = 0.25)
        ax.bar(X + 0.25, act_power_tot_avg.numpy()[int(T/3)-1::int(T/3)], color = 'c', width = 0.25)
        ax.bar(X + 0.50,  stat_power_tot_avg.numpy()[int(T/3)-1::int(T/3)] , color = 'm', width = 0.25)
        plt.title('Total Energy Generated', fontsize=25)
        #plt.ylabel('Energy', fontsize=25)
        plt.xlabel('Timesteps', fontsize=25)
        plt.xticks(X+0.25, ( '320', '640', '960'))
        plt.yticks(np.arange(0, 5000, 1000))
        plt.legend(labels=[compnm, algnm ,'Stationary'],  prop={'size': 22})
    
        # Plot predictive means as blue line
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_power_tot_avg.numpy()[0::int(T/pl_freq)], yerr=comp_power_tot_std.numpy()[0::int(T/pl_freq)],color='b',capsize=10, marker='o')
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_power_tot_avg.numpy()[0::int(T/pl_freq)], yerr=act_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'c',capsize=10, marker='s')

        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_power_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_power_tot_avg.numpy()[0::int(T/pl_freq)],yerr=stat_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'm',capsize=10, marker='d')
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_power_tot_avg.numpy()[0::int(T/pl_freq)],yerr=pier_power_tot_std.numpy()[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
        # Shade between the lower and upper confidence bounds
        #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        #ax.set_ylim([-1, 2]
        #plt.show()
        np.savetxt('newdirfont1216extrem/data/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.csv' %(T,gam,title,st_ti,lat,lon,ls_1), (comp_power_tot_avg.numpy(),  act_power_tot_avg.numpy(), stat_power_tot_avg.numpy()), delimiter=',')

        plt.savefig('newdirfont1216extrem/gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.png' %(gam,title,st_ti,lat,lon,ls_1))

    
    return [tot_avg,tot_std,mvt_avg,mvt_std,serv_avg,serv_std,reg_avg,reg_std,ptot_avg,ptot_std]

    