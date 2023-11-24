import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch

algnm="$\\bf{GP-MD}$"
septnm='MinC-Known'
compnm='CGP-LCB'
knalgnm='MD-Known'
tkse=18
lwdth=2.5
servpltle='Total Energy Generated'
wplt='totpow'
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

comp_power_tot_avg=copw[0]
comp_power_tot_std=copw[1]
act_power_tot_avg=copw[2]
act_power_tot_std=copw[3]
servopt_power_tot_avg=copw[4]
servopt_power_tot_std=copw[5]  
stat_power_tot_avg=copw[6]
stat_power_tot_std=copw[7]
kn_act_power_tot_avg=copw[8]
kn_act_power_tot_std=copw[9]

#CB91_Blue = ‘#2CBDFE’-act
#CB91_Green = ‘#47DBCD’
#CB91_Pink = ‘#F3A0F2’-stat
#CB91_Purple = ‘#9D2EC5’
#CB91_Violet = ‘#661D98’-comp
#CB91_Amber = ‘#F5B14C-kn_act

cgp_col='#2CBDFE'
gpmd_col='#661D98'

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    X=np.arange(3)

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.bar(X + 0.25, act_power_tot_avg[int(T/3)-1::int(T/3)], yerr=act_power_tot_std[int(T/3)-1::int(T/3)], color = gpmd_col, width = 0.25)
    ax.bar(X + 0.00, comp_power_tot_avg[int(T/3)-1::int(T/3)], yerr=comp_power_tot_std[int(T/3)-1::int(T/3)],color = cgp_col, width = 0.25)
    ax.bar(X + 0.50,  stat_power_tot_avg[int(T/3)-1::int(T/3)] , yerr=stat_power_tot_std[int(T/3)-1::int(T/3)], color = '#F3A0F2', width = 0.25)
    plt.title('Total Energy Generated', fontsize=25)
    #plt.ylabel('Energy', fontsize=25)
    plt.xlabel('Timesteps', fontsize=25)
    plt.ylim(bottom=1000)
    plt.xticks(X+0.25, ( '320', '640', '960'))
    plt.yticks(np.arange(1000, 5000, 1000))
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    plt.legend(labels=[algnm ,compnm,'Stationary'],  prop={'size': 25})
    #plt.show()
    plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.png' %(T,gam,rand_int,init_t,lat,lon,ls_1))


init_t=2226
lat=13
lon=10
ls_1=6.85
T=960
#gam=4
rand_int=38102414#38048660
copw=pd.read_csv('%s/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_%s.csv' %(fold_name,T,gam,rand_int,init_t,lat,lon,ls_1,wplt), sep=',',header=None)
copw=np.asarray(copw)


comp_power_tot_avg=copw[0]
comp_power_tot_std=copw[1]
act_power_tot_avg=copw[2]
act_power_tot_std=copw[3]
servopt_power_tot_avg=copw[4]
servopt_power_tot_std=copw[5]  
stat_power_tot_avg=copw[6]
stat_power_tot_std=copw[7]
kn_act_power_tot_avg=copw[8]
kn_act_power_tot_std=copw[9]


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    X=np.arange(3)

    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.bar(X + 0.25, act_power_tot_avg[int(T/3)-1::int(T/3)], yerr=act_power_tot_std[int(T/3)-1::int(T/3)], color = gpmd_col, width = 0.25)
    ax.bar(X + 0.00, comp_power_tot_avg[int(T/3)-1::int(T/3)], yerr=comp_power_tot_std[int(T/3)-1::int(T/3)],color = cgp_col, width = 0.25)
    ax.bar(X + 0.50,  stat_power_tot_avg[int(T/3)-1::int(T/3)] , yerr=stat_power_tot_std[int(T/3)-1::int(T/3)], color = '#F3A0F2', width = 0.25)
    plt.title('Total Energy Generated', fontsize=25)
    #plt.ylabel('Energy', fontsize=25)
    plt.xlabel('Timesteps', fontsize=25)
    plt.ylim(bottom=1000)
    plt.xticks(X+0.25, ( '320', '640', '960'))
    plt.yticks(np.arange(1000, 5000, 1000))
    plt.xticks(fontsize=tkse)
    plt.yticks(fontsize=tkse)
    plt.legend(labels=[algnm ,compnm,'Stationary'],  prop={'size': 25})
    #plt.show()
    plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_gam=%f_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_bartotpow.png' %(T,gam,rand_int,init_t,lat,lon,ls_1))
