import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch

algnm="$\\bf{GP-MD}$"
septnm='MinC-Known'
compnm='CGP-LCB'
knalgnm='MD-Known'
lwdth=4#2.5
servpltle='Average Total Energy'#Movement cost,Service Cost
wplt='totpow'#mvt,serv
pl_freq=20
fold_name='newdirfont1216extrem/data'
init_t=3186
lat=12
lon=16
ls_1=6.85
T=960
gam=4
rand_int=38102385#38048720
copw=pd.read_csv('%s/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.csv' %(fold_name,T,gam,rand_int,init_t,lat,lon,ls_1,wplt), sep=',',header=None)
copw=np.asarray(copw)
if wplt=='totpow':
    for t in range(0,T):
        copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)
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

cgp_col='#2CBDFE'
gpmd_col='#661D98'
lsze=22
tkse=30
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
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*',linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([ algnm , compnm, septnm, 'Stationary', knalgnm], prop={'size': lsze+6},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze )
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    #plt.show()
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.png' %(T,gam,rand_int,init_t,lat,lon,ls_1,wplt))



init_t=2226
lat=13
lon=10
ls_1=6.85
T=960
#gam=4
rand_int=38102414#38048660
copw=pd.read_csv('%s/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.csv' %(fold_name,T,gam,rand_int,init_t,lat,lon,ls_1,wplt), sep=',',header=None)
copw=np.asarray(copw)
if wplt=='totpow':
    for t in range(0,T):
        copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)
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
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*',linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-1, 2])
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([ algnm , compnm, septnm, 'Stationary', knalgnm], prop={'size': lsze+7.5},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze )
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    #plt.show()
    plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.png' %(T,gam,rand_int,init_t,lat,lon,ls_1,wplt))


copw=pd.read_csv('%s/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.csv' %(fold_name,T,gam,rand_int,init_t,lat,lon,ls_1), sep=',',header=None)
copw=np.asarray(copw)
#copw=copw[:,1:]
#print(copw)
comp_power_tot_avg=copw[0]
act_power_tot_avg=copw[1]
stat_power_tot_avg=copw[2]


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    X=np.arange(3)

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.bar(X + 0.25, act_power_tot_avg[int(T/3)-1::int(T/3)], color = gpmd_col, width = 0.25)
    ax.bar(X + 0.00, comp_power_tot_avg[int(T/3)-1::int(T/3)], color = cgp_col, width = 0.25)
    ax.bar(X + 0.50,  stat_power_tot_avg[int(T/3)-1::int(T/3)] , color = '#F3A0F2', width = 0.25)
    ax.xaxis.set_label_coords(.5, -.07)
    plt.title('Total Energy Generated', fontsize=asze)
    #plt.ylabel('Energy', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    plt.xticks(X+0.25, ( '320', '640', '960'))
    plt.yticks(np.arange(0, 5000, 1000))
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    plt.legend(labels=[algnm , compnm, 'Stationary'],  prop={'size': lsze+8})
    #plt.show()
    plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.png' %(T,gam,rand_int,init_t,lat,lon,ls_1))




init_t=2226
lat=18
lon=26
ls_1=12.94
T=960
#gam=4
rand_int=38102441#38048690
copw=pd.read_csv('%s/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.csv' %(fold_name,T,gam,rand_int,init_t,lat,lon,ls_1,wplt), sep=',',header=None)
copw=np.asarray(copw)
if wplt=='totpow':
    for t in range(0,T):
        copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)
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
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*',linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([0, 6])#foravgtotpow
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([algnm , compnm, septnm, 'Stationary', knalgnm], prop={'size': lsze+6},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze)
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    #plt.show()
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.png' %(T,gam,rand_int,init_t,lat,lon,ls_1,wplt))



copw=pd.read_csv('%s/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.csv' %(fold_name,T,gam,rand_int,init_t,lat,lon,ls_1), sep=',',header=None)
copw=np.asarray(copw)
#copw=copw[:,1:]
#print(copw)
comp_power_tot_avg=copw[0]
act_power_tot_avg=copw[1]
stat_power_tot_avg=copw[2]


with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(3)

        # Get upper and lower confidence bounds
        #lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.bar(X + 0.25, act_power_tot_avg[int(T/3)-1::int(T/3)], color = gpmd_col, width = 0.25)
        ax.bar(X + 0.00, comp_power_tot_avg[int(T/3)-1::int(T/3)], color = cgp_col, width = 0.25)
        
        ax.bar(X + 0.50,  stat_power_tot_avg[int(T/3)-1::int(T/3)] , color = '#F3A0F2', width = 0.25)
        ax.xaxis.set_label_coords(.5, -.07)
        plt.title('Total Energy Generated', fontsize=asze)
        #plt.ylabel('Energy', fontsize=25)
        plt.xlabel('Timesteps', fontsize=asze-5)
        plt.xticks(X+0.25, ( '320', '640', '960'))
        plt.yticks(np.arange(0, 3000, 1000))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        plt.legend(labels=[ algnm , compnm, 'Stationary'],  prop={'size': lsze+12})
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.png' %(T,gam,rand_int,init_t,lat,lon,ls_1))





init_t=3186
lat=13
lon=10
ls_1=12.94
T=960
#gam=4
rand_int=38102414#38048721
copw=pd.read_csv('%s/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.csv' %(fold_name,T,gam,rand_int,init_t,lat,lon,ls_1,wplt), sep=',',header=None)
copw=np.asarray(copw)
if wplt=='totpow':
    for t in range(0,T):
        copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)
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
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], servopt_cost_serv_avg[0::int(T/pl_freq)],yerr=servopt_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= 'y',capsize=10, marker='^',linewidth=lwdth)
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], stat_cost_serv_avg[0::int(T/pl_freq)],yerr=stat_cost_serv_std[0::int(T/pl_freq)],color= '#F3A0F2',capsize=10, marker='d',linewidth=lwdth)
    #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], pier_cost_serv_avg[0::int(T/pl_freq)],yerr=pier_cost_serv_std[0::int(T/pl_freq)],color= 'k',capsize=10, marker='x')
    ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], kn_act_cost_serv_avg[0::int(T/pl_freq)],yerr=kn_act_cost_serv_std[0::int(T/pl_freq)],ls='dashed',color= '#F5B14C',capsize=10, marker='*',linewidth=lwdth)
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([3, 8.5])#avgtotpow
    ax.xaxis.set_label_coords(.5, -.07)
    ax.legend([  algnm , compnm, septnm, 'Stationary', knalgnm], prop={'size': lsze+6.5},labelspacing=0.2)
    plt.title(servpltle, fontsize=asze)
    #plt.ylabel('Cumulative cost', fontsize=25)
    plt.xlabel('Timesteps', fontsize=asze-5)
    #plt.show()
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.png' %(T,gam,rand_int,init_t,lat,lon,ls_1,wplt))

