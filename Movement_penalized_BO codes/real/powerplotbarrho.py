from distutils.log import error
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch

algnm="$\\bf{GP-MD}$"
septnm='MinC-Known'
compnm='CGP-LCB'
knalgnm='MD-Known'
servpltle='Average Total Power'
tkse=30
lwdth=2.5
lsze=18
pl_freq=20
fold_name='newdirfont1216extrem/data'
ch=1
gam_val=[0.5,0.75,1,2,3,4,6,8,10]
st_gm=3
gam_val=gam_val[st_gm:]
init_t=3186
lat=12
lon=16
ls_1=6.85
T=960
#gam=1
rand_int=38102385#38048720
copw=pd.read_csv('%s/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.csv' %(fold_name,T,rand_int,init_t,lat,lon,ls_1), sep=',',header=None)
copw=np.asarray(copw)
#for t in range(0,T):
#    copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)

comp_power_tot_avg=copw[0,st_gm:]
comp_power_tot_std=copw[1,st_gm:]
act_power_tot_avg=copw[2,st_gm:]
act_power_tot_std=copw[3,st_gm:]
servopt_power_tot_avg=copw[4,st_gm:]
servopt_power_tot_std=copw[5,st_gm:]  
stat_power_tot_avg=copw[6,st_gm:]
stat_power_tot_std=copw[7,st_gm:]
kn_act_power_tot_avg=copw[8,st_gm:]
kn_act_power_tot_std=copw[9,st_gm:]


#CB91_Blue = ‘#2CBDFE’-act
#CB91_Green = ‘#47DBCD’
#CB91_Pink = ‘#F3A0F2’-stat
#CB91_Purple = ‘#9D2EC5’
#CB91_Violet = ‘#661D98’-comp
#CB91_Amber = ‘#F5B14C-kn_act

cgp_col='#2CBDFE'
gpmd_col='#661D98'

if ch==0:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        # Plot predictive means as blue linenp.a
        ax.errorbar(gam_val, np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std),color= '2CBDFE',capsize=10, marker='s',linewidth=lwdth)
        ax.errorbar(gam_val, np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std),color='661D98',capsize=10, marker='o',linewidth=lwdth)
        
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_power_tot_avg),yerr=np.asarray(servopt_power_tot_std),ls='dashed',color= 'y',capsize=10, marker='^',linewidth=lwdth)
        ax.errorbar(gam_val, np.asarray(stat_power_tot_avg),yerr=np.asarray(stat_power_tot_std),color= 'm',capsize=10, marker='d',linewidth=lwdth)
        #ax.errorbar(gam_val, np.asarray(pier_power_tot_avg),yerr=np.asarray(pier_power_tot_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_power_tot_avg),yerr=np.asarray(kn_act_power_tot_std),ls='dashed',color= 'tab:orange',capsize=10, marker='*',linewidth=lwdth)
        ax.legend([  algnm, compnm, septnm, 'Stationary', knalgnm], prop={'size': 25})
        plt.title('Total Energy Generated', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=25)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))

else:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(len(gam_val))
        ax.bar(X + 0.15,  np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std), color = gpmd_col, width = 0.15,linewidth=lwdth)
        ax.bar(X + 0.00,  np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std), color = cgp_col, width = 0.15,linewidth=lwdth)
        
        ax.bar(X + 0.30,  np.asarray(stat_power_tot_avg), yerr=np.asarray(stat_power_tot_std), color = '#F3A0F2', width = 0.15,linewidth=lwdth)
        ax.bar(X + 0.45,  np.asarray(servopt_power_tot_avg), yerr=np.asarray(servopt_power_tot_std), ls='dashed', facecolor='None',edgecolor = 'y', color='white',  hatch='//',  width = 0.15,linewidth=lwdth)
        ax.bar(X + 0.60,  np.asarray(kn_act_power_tot_avg), yerr=np.asarray(kn_act_power_tot_std), ls='dashed',facecolor='None',edgecolor = '#F5B14C', color='white', hatch='//', width = 0.15,linewidth=lwdth)
        
        ax.xaxis.set_label_coords(.5, -.05)
        plt.title('Total Energy Generated', fontsize=40)
        #plt.ylabel('Energy', fontsize=25)
        plt.ylim(bottom=2000)
        plt.xlabel(r'$\rho$', fontsize=40)
        plt.xticks(X+0.30, gam_val)
        plt.yticks(np.arange(2000, 7000, 1000))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        plt.legend(labels=[ algnm , compnm, 'Stationary',septnm,knalgnm ],  prop={'size': lsze+8},labelspacing=0.2)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))



init_t=2226
lat=13
lon=10
ls_1=6.85
T=960
#gam=1
rand_int=38102414#38048660
copw=pd.read_csv('%s/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.csv' %(fold_name,T,rand_int,init_t,lat,lon,ls_1), sep=',',header=None)
copw=np.asarray(copw)
#for t in range(0,T):
#    copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)

comp_power_tot_avg=copw[0,st_gm:]
comp_power_tot_std=copw[1,st_gm:]
act_power_tot_avg=copw[2,st_gm:]
act_power_tot_std=copw[3,st_gm:]
servopt_power_tot_avg=copw[4,st_gm:]
servopt_power_tot_std=copw[5,st_gm:]  
stat_power_tot_avg=copw[6,st_gm:]
stat_power_tot_std=copw[7,st_gm:]
kn_act_power_tot_avg=copw[8,st_gm:]
kn_act_power_tot_std=copw[9,st_gm:]


if ch==0:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        # Plot predictive means as blue linenp.a
        ax.errorbar(gam_val, np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std),color= 'c',capsize=10, marker='s')
        ax.errorbar(gam_val, np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std),color='b',capsize=10, marker='o')
        
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_power_tot_avg),yerr=np.asarray(servopt_power_tot_std),ls='dashed',color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_power_tot_avg),yerr=np.asarray(stat_power_tot_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_power_tot_avg),yerr=np.asarray(pier_power_tot_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_power_tot_avg),yerr=np.asarray(kn_act_power_tot_std),ls='dashed',color= 'tab:orange',capsize=10, marker='*')
        ax.legend([ algnm, compnm, septnm, 'Stationary', knalgnm], prop={'size': 25})
        plt.title('Total Energy Generated', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=25)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))

else:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(len(gam_val))
        ax.bar(X + 0.15,  np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std), color = gpmd_col, width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.00,  np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std), color = cgp_col, width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.30,  np.asarray(stat_power_tot_avg), yerr=np.asarray(stat_power_tot_std), color = '#F3A0F2', width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.45,  np.asarray(servopt_power_tot_avg), yerr=np.asarray(servopt_power_tot_std), ls='dashed', facecolor='None',edgecolor = 'y', color='white',  hatch='//',  width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.60,  np.asarray(kn_act_power_tot_avg), yerr=np.asarray(kn_act_power_tot_std), ls='dashed',facecolor='None',edgecolor = '#F5B14C', color='white', hatch='//', width = 0.15, linewidth=lwdth)
        ax.xaxis.set_label_coords(.5, -.05)
        
        plt.title('Total Energy Generated', fontsize=40)
        #plt.ylabel('Energy', fontsize=25)
        plt.ylim(bottom=2000)
        
        #plt.title('Total Energy Generated', fontsize=40)
        #plt.ylabel('Energy', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=40)
        plt.xticks(X+0.30, gam_val)
        plt.yticks(np.arange(2000, 7000, 1000))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        plt.legend(labels=[ algnm , compnm, 'Stationary',septnm,knalgnm ],  prop={'size': lsze+6},labelspacing=0.2)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))





init_t=2226
lat=18
lon=26
ls_1=12.94
T=960
#gam=1
rand_int=38102441#38048690
copw=pd.read_csv('%s/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.csv' %(fold_name,T,rand_int,init_t,lat,lon,ls_1), sep=',',header=None)
copw=np.asarray(copw)
#for t in range(0,T):
#    copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)

comp_power_tot_avg=copw[0,st_gm:]
comp_power_tot_std=copw[1,st_gm:]
act_power_tot_avg=copw[2,st_gm:]
act_power_tot_std=copw[3,st_gm:]
servopt_power_tot_avg=copw[4,st_gm:]
servopt_power_tot_std=copw[5,st_gm:]  
stat_power_tot_avg=copw[6,st_gm:]
stat_power_tot_std=copw[7,st_gm:]
kn_act_power_tot_avg=copw[8,st_gm:]
kn_act_power_tot_std=copw[9,st_gm:]


if ch==0:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        # Plot predictive means as blue linenp.a
        ax.errorbar(gam_val, np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std),color= 'c',capsize=10, marker='s')
        ax.errorbar(gam_val, np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std),color='b',capsize=10, marker='o')
        
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_power_tot_avg),yerr=np.asarray(servopt_power_tot_std),ls='dashed',color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_power_tot_avg),yerr=np.asarray(stat_power_tot_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_power_tot_avg),yerr=np.asarray(pier_power_tot_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_power_tot_avg),yerr=np.asarray(kn_act_power_tot_std),ls='dashed',color= 'tab:orange',capsize=10, marker='*')
        ax.legend([ algnm, compnm, septnm, 'Stationary', knalgnm], prop={'size': 25})
        plt.title('Total Energy Generated', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=25)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))

else:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(len(gam_val))
        ax.bar(X + 0.15,  np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std), color = gpmd_col, width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.00,  np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std), color = cgp_col, width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.30,  np.asarray(stat_power_tot_avg), yerr=np.asarray(stat_power_tot_std), color = '#F3A0F2', width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.45,  np.asarray(servopt_power_tot_avg), yerr=np.asarray(servopt_power_tot_std), ls='dashed', facecolor='None',edgecolor = 'y', color='white',  hatch='//',  width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.60,  np.asarray(kn_act_power_tot_avg), yerr=np.asarray(kn_act_power_tot_std), ls='dashed',facecolor='None',edgecolor = '#F5B14C', color='white', hatch='//', width = 0.15, linewidth=lwdth)
        ax.xaxis.set_label_coords(.5, -.04)
        
        plt.title('Total Energy Generated', fontsize=40)
        #plt.ylabel('Energy', fontsize=25)
        plt.ylim(bottom=1000)
        #plt.ylabel('Energy', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=40)
        plt.xticks(X+0.30, gam_val)
        plt.yticks(np.arange(1000, 4000, 1000))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        plt.legend(labels=[ algnm , compnm, 'Stationary',septnm,knalgnm ],  prop={'size': lsze+8}, labelspacing=0.2)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))




init_t=3186
lat=13
lon=10
ls_1=12.94
T=960
#gam=1
rand_int=38102414#38048721
copw=pd.read_csv('%s/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_woday_totpow.csv' %(fold_name,T,rand_int,init_t,lat,lon,ls_1), sep=',',header=None)
copw=np.asarray(copw)
#for t in range(0,T):
#    copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)

comp_power_tot_avg=copw[0,st_gm:]
comp_power_tot_std=copw[1,st_gm:]
act_power_tot_avg=copw[2,st_gm:]
act_power_tot_std=copw[3,st_gm:]
servopt_power_tot_avg=copw[4,st_gm:]
servopt_power_tot_std=copw[5,st_gm:]  
stat_power_tot_avg=copw[6,st_gm:]
stat_power_tot_std=copw[7,st_gm:]
kn_act_power_tot_avg=copw[8,st_gm:]
kn_act_power_tot_std=copw[9,st_gm:]


if ch==0:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        # Plot predictive means as blue linenp.a
        ax.errorbar(gam_val, np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std),color= 'c',capsize=10, marker='s')
        ax.errorbar(gam_val, np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std),color='b',capsize=10, marker='o')
        
        #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(gam_val, np.asarray(servopt_power_tot_avg),yerr=np.asarray(servopt_power_tot_std),ls='dashed',color= 'y',capsize=10, marker='^')
        ax.errorbar(gam_val, np.asarray(stat_power_tot_avg),yerr=np.asarray(stat_power_tot_std),color= 'm',capsize=10, marker='d')
        #ax.errorbar(gam_val, np.asarray(pier_power_tot_avg),yerr=np.asarray(pier_power_tot_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(gam_val, np.asarray(kn_act_power_tot_avg),yerr=np.asarray(kn_act_power_tot_std),ls='dashed',color= 'tab:orange',capsize=10, marker='*')
        ax.legend([ algnm, compnm, septnm, 'Stationary', knalgnm], prop={'size': 25})
        plt.title('Total Energy Generated', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=25)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))

else:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(len(gam_val))
        ax.bar(X + 0.15,  np.asarray(act_power_tot_avg), yerr=np.asarray(act_power_tot_std), color = gpmd_col, width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.00,  np.asarray(comp_power_tot_avg), yerr=np.asarray(comp_power_tot_std), color = cgp_col, width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.30,  np.asarray(stat_power_tot_avg), yerr=np.asarray(stat_power_tot_std), color = '#F3A0F2', width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.45,  np.asarray(servopt_power_tot_avg), yerr=np.asarray(servopt_power_tot_std), ls='dashed', facecolor='None',edgecolor = 'y', color='white',  hatch='//',  width = 0.15, linewidth=lwdth)
        ax.bar(X + 0.60,  np.asarray(kn_act_power_tot_avg), yerr=np.asarray(kn_act_power_tot_std), ls='dashed',facecolor='None',edgecolor = '#F5B14C', color='white', hatch='//', width = 0.15, linewidth=lwdth)
        ax.xaxis.set_label_coords(.5, -.04)
        
        plt.title('Total Energy Generated', fontsize=40)
        #plt.ylabel('Energy', fontsize=25)
        plt.ylim(bottom=2000)
        plt.xlabel(r'$\rho$', fontsize=40)
        plt.xticks(X+0.30, gam_val)
        plt.yticks(np.arange(2000, 7000, 1000))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        plt.legend(labels=[ algnm , compnm, 'Stationary',septnm,knalgnm ],  prop={'size': lsze+8},labelspacing=0.2)
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))



