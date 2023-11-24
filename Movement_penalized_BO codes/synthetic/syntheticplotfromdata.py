from logging import debug
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch

algnm="$\\bf{GP-MD}$"
septnm='MinC-Known'
compnm='CGP-LCB'
knalgnm='MD-Known'
servpltle='Movement Cost'#Service Cost,Total Cost
wplt='mvt'#serv,tot
pl_freq=20
fold_name='data'
#fl='t=500_f_n=24_n=400_gam=4.000000_Kernel=rbf_f_rep=3_38141459_mineps=14.000000_c_n=24_len_scale=0.200000_tot_2D'
gam=4
rand_int=38179084
T=500


cgp_col='#2CBDFE'
gpmd_col='#661D98'

copw=pd.read_csv('%s/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,gam,rand_int,wplt), sep=',',header=None)
copw=np.asarray(copw)

comp_cost_serv_avg=copw[0]
comp_cost_serv_std=copw[1]
act_cost_serv_avg=copw[2]
act_cost_serv_std=copw[3]
servopt_cost_serv_avg=copw[4]
servopt_cost_serv_std=copw[5]  
stat_cost_serv_avg=copw[6]
stat_cost_serv_std=copw[7]
kn_act_cost_serv_avg=copw[8]
kn_act_cost_serv_std=copw[9]

#CB91_Blue = ‘#2CBDFE’-act
#CB91_Green = ‘#47DBCD’
#CB91_Pink = ‘#F3A0F2’-stat
#CB91_Purple = ‘#9D2EC5’
#CB91_Violet = ‘#661D98’-comp
#CB91_Amber = ‘#F5B14C-kn_act

lwdth=4#2.5
lsze=30
tkse=26
asze=40
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars

    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_serv_avg[0::int(T/pl_freq)], yerr=act_cost_serv_std[0::int(T/pl_freq)],color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_serv_avg[0::int(T/pl_freq)], yerr=comp_cost_serv_std[0::int(T/pl_freq)],color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_serv_avg[0::int(T/pl_freq)],yerr=rand_cost_serv_std[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^', linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d', linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*', linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([ algnm ,compnm,septnm, 'Stationary', knalgnm], prop={'size': lsze},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze)
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    #plt.show()
    plt.savefig('pltfrdata/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_len_scale=0.200000_%s_2D.png' %(gam,rand_int,wplt))



gam=2
rand_int=38179053
copw=pd.read_csv('%s/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,gam,rand_int,wplt), sep=',',header=None)
copw=np.asarray(copw)

comp_cost_serv_avg=copw[0]
comp_cost_serv_std=copw[1]
act_cost_serv_avg=copw[2]
act_cost_serv_std=copw[3]
servopt_cost_serv_avg=copw[4]
servopt_cost_serv_std=copw[5]  
stat_cost_serv_avg=copw[6]
stat_cost_serv_std=copw[7]
kn_act_cost_serv_avg=copw[8]
kn_act_cost_serv_std=copw[9]



with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars

    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_serv_avg[0::int(T/pl_freq)], yerr=act_cost_serv_std[0::int(T/pl_freq)],color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_serv_avg[0::int(T/pl_freq)], yerr=comp_cost_serv_std[0::int(T/pl_freq)],color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_serv_avg[0::int(T/pl_freq)],yerr=rand_cost_serv_std[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^', linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d', linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*', linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([ algnm ,compnm,septnm, 'Stationary', knalgnm], prop={'size': lsze},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze)
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    #plt.show()
    plt.savefig('pltfrdata/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_len_scale=0.200000_%s_2D.png' %(gam,rand_int,wplt))


gam=1
rand_int=38179052

copw=pd.read_csv('%s/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,gam,rand_int,wplt), sep=',',header=None)
copw=np.asarray(copw)

comp_cost_serv_avg=copw[0]
comp_cost_serv_std=copw[1]
act_cost_serv_avg=copw[2]
act_cost_serv_std=copw[3]
servopt_cost_serv_avg=copw[4]
servopt_cost_serv_std=copw[5]  
stat_cost_serv_avg=copw[6]
stat_cost_serv_std=copw[7]
kn_act_cost_serv_avg=copw[8]
kn_act_cost_serv_std=copw[9]



with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars

    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_serv_avg[0::int(T/pl_freq)], yerr=act_cost_serv_std[0::int(T/pl_freq)],color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_serv_avg[0::int(T/pl_freq)], yerr=comp_cost_serv_std[0::int(T/pl_freq)],color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_serv_avg[0::int(T/pl_freq)],yerr=rand_cost_serv_std[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^', linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d', linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*', linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([  algnm ,compnm, septnm, 'Stationary', knalgnm], prop={'size': lsze},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze)
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    #plt.show()
    plt.savefig('pltfrdata/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_len_scale=0.200000_%s_2D.png' %(gam,rand_int,wplt))


gam=0.5
rand_int=38179051
copw=pd.read_csv('%s/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,gam,rand_int,wplt), sep=',',header=None)
copw=np.asarray(copw)

comp_cost_serv_avg=copw[0]
comp_cost_serv_std=copw[1]
act_cost_serv_avg=copw[2]
act_cost_serv_std=copw[3]
servopt_cost_serv_avg=copw[4]
servopt_cost_serv_std=copw[5]  
stat_cost_serv_avg=copw[6]
stat_cost_serv_std=copw[7]
kn_act_cost_serv_avg=copw[8]
kn_act_cost_serv_std=copw[9]



with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars

    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_serv_avg[0::int(T/pl_freq)], yerr=act_cost_serv_std[0::int(T/pl_freq)],color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_serv_avg[0::int(T/pl_freq)], yerr=comp_cost_serv_std[0::int(T/pl_freq)],color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_serv_avg[0::int(T/pl_freq)],yerr=rand_cost_serv_std[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^', linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d', linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*', linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([ algnm , compnm, septnm, 'Stationary', knalgnm], prop={'size': lsze},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze)
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    #plt.show()
    plt.savefig('pltfrdata/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_len_scale=0.200000_%s_2D.png' %(gam,rand_int,wplt))

gam=0.25
rand_int=38179051
copw=pd.read_csv('%s/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,gam,rand_int,wplt), sep=',',header=None)
copw=np.asarray(copw)

comp_cost_serv_avg=copw[0]
comp_cost_serv_std=copw[1]
act_cost_serv_avg=copw[2]
act_cost_serv_std=copw[3]
servopt_cost_serv_avg=copw[4]
servopt_cost_serv_std=copw[5]  
stat_cost_serv_avg=copw[6]
stat_cost_serv_std=copw[7]
kn_act_cost_serv_avg=copw[8]
kn_act_cost_serv_std=copw[9]



with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars

    # Plot predictive means as blue line
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], act_cost_serv_avg[0::int(T/pl_freq)], yerr=act_cost_serv_std[0::int(T/pl_freq)],color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], comp_cost_serv_avg[0::int(T/pl_freq)], yerr=comp_cost_serv_std[0::int(T/pl_freq)],color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
    
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_serv_avg[0::int(T/pl_freq)],yerr=rand_cost_serv_std[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^', linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d', linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*', linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([ algnm , compnm, septnm, 'Stationary', knalgnm], prop={'size': lsze},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze)
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    #plt.show()
    plt.savefig('pltfrdata/t=500_f_n=24_n=400_gam=%f_Kernel=rbf_f_rep=3_%d_len_scale=0.200000_%s_2D.png' %(gam,rand_int,wplt))

