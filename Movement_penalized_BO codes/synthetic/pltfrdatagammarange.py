from distutils.log import error
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch

algnm="$\\bf{GP-MD}$"
septnm='MinC-Known'
compnm='CGP-LCB'
knalgnm='MD-Known'
lsze=30
tkse=26
asze=40
lwdth=4#2.5
pl_freq=20
fold_name='data'
ch=0
#gam_val=[0.5,0.75,1,2,3,4,6,8,10]
gam_val=[0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 10, 15, 20]

st_gm=0
gam_val=gam_val[st_gm:]
alpha_values=[]
for gam in gam_val:
  alpha_values.append(gam/(1+gam))
wplt='serv'
T=800
#gam=1
#t=800_f_n=10_n=400__Kernel=rbf_f_rep=2_38133805_mineps=14.000000_c_n=24_len_scale=0.200000_serv_2D
rand_int=38180793#38133805#38048720
copw=pd.read_csv('%s/t=800_f_n=10_n=400__Kernel=rbf_f_rep=2_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,rand_int,wplt), sep=',',header=None)
copw=np.asarray(copw)
#for t in range(0,T):
#    copw[:,t]=copw[:,t]/t
#copw=copw[:,1:]
#print(copw)

comp_cost_serv_avg=copw[0,st_gm:]
comp_cost_serv_std=copw[1,st_gm:]
act_cost_serv_avg=copw[2,st_gm:]
act_cost_serv_std=copw[3,st_gm:]
servopt_cost_serv_avg=copw[4,st_gm:]
servopt_cost_serv_std=copw[5,st_gm:]  
stat_cost_serv_avg=copw[6,st_gm:]
stat_cost_serv_std=copw[7,st_gm:]
kn_act_cost_serv_avg=copw[8,st_gm:]
kn_act_cost_serv_std=copw[9,st_gm:]


#CB91_Blue = ‘#2CBDFE’-act
#CB91_Green = ‘#47DBCD’
#CB91_Pink = ‘#F3A0F2’-stat
#CB91_Purple = ‘#9D2EC5’
#CB91_Violet = ‘#661D98’-comp
#CB91_Amber = ‘#F5B14C-kn_act
cgp_col='#2CBDFE'
gpmd_col='#661D98'

servpltle='Service Cost'
if ch==0:
    with torch.no_grad():
    # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        ax.errorbar(alpha_values, np.asarray(act_cost_serv_avg), yerr=np.asarray(act_cost_serv_std),color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
        ax.errorbar(alpha_values, np.asarray(comp_cost_serv_avg), yerr=np.asarray(comp_cost_serv_std),color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
        
            #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(alpha_values, np.asarray(stat_cost_serv_avg),yerr=np.asarray(stat_cost_serv_std),color= '#F3A0F2',capsize=10, marker='^',linewidth=lwdth)
        ax.errorbar(alpha_values, np.asarray(servopt_cost_serv_avg),yerr=np.asarray(servopt_cost_serv_std), ls='dashed',color= 'y',capsize=10, marker='d',linewidth=lwdth)
            #ax.errorbar(alpha_values, np.asarray(pier_cost_serv_avg),yerr=np.asarray(pier_cost_serv_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(alpha_values, np.asarray(kn_act_cost_serv_avg),yerr=np.asarray(kn_act_cost_serv_std),ls='dashed',color= '#F5B14C',capsize=10, marker='*',linewidth=lwdth)
        ax.xaxis.set_label_coords(.5, -.05)    
        ax.legend([ algnm,compnm, 'Stationary', septnm, knalgnm], prop={'size': lsze},labelspacing=0.2)
        plt.title(servpltle, fontsize=asze)

        #plt.ylabel(servpltle, fontsize=20)
        plt.xlabel(r'Range of $\frac{\rho}{1+\rho}$', fontsize=asze-5)
        plt.xticks(np.arange(0,6)*0.2,('0','0.2','0.4','0.6','0.8','1'))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        plt.show()
        #plt.savefig('pltfrdata/t=800_f_n=10_n=400__Kernel=rbf_f_rep=2_%d_len_scale=0.200000_%s_2D.png' %(rand_int,wplt))

else:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(len(gam_val))
        ax.bar(X + 0.15,  np.asarray(act_cost_serv_avg), yerr=np.asarray(act_cost_serv_std), color = gpmd_col, width = 0.15)
        ax.bar(X + 0.00,  np.asarray(comp_cost_serv_avg), yerr=np.asarray(comp_cost_serv_std), color = cgp_col, width = 0.15)
        ax.bar(X + 0.30,  np.asarray(stat_cost_serv_avg), yerr=np.asarray(stat_cost_serv_std), color = '#F3A0F2', width = 0.15)
        ax.bar(X + 0.45,  np.asarray(servopt_cost_serv_avg), yerr=np.asarray(servopt_cost_serv_std), ls='dashed', facecolor='None',edgecolor = 'y', color='white',  hatch='//',  width = 0.15)
        ax.bar(X + 0.60,  np.asarray(kn_act_cost_serv_avg), yerr=np.asarray(kn_act_cost_serv_std), ls='dashed',facecolor='None',edgecolor = '#F5B14C', color='white', hatch='//', width = 0.15)
        
        
        plt.title('Total Energy Generated', fontsize=25)
        #plt.ylabel('Energy', fontsize=25)
        plt.ylim(bottom=1000)
        plt.xlabel(r'$\rho$', fontsize=25)
        plt.xticks(X+0.30, gam_val)
        plt.yticks(np.arange(1000, 7000, 1000))
        plt.legend(labels=[algnm ,compnm,'Stationary',septnm,knalgnm ],  prop={'size': 15})
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))



wplt='mvt'
copw1=pd.read_csv('%s/t=800_f_n=10_n=400__Kernel=rbf_f_rep=2_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,rand_int,wplt), sep=',',header=None)
copw1=np.asarray(copw1)

comp_cost_mvt_avg=copw1[0,st_gm:]
comp_cost_mvt_std=copw1[1,st_gm:]
act_cost_mvt_avg=copw1[2,st_gm:]
act_cost_mvt_std=copw1[3,st_gm:]
servopt_cost_mvt_avg=copw1[4,st_gm:]
servopt_cost_mvt_std=copw1[5,st_gm:]  
stat_cost_mvt_avg=copw1[6,st_gm:]
stat_cost_mvt_std=copw1[7,st_gm:]
kn_act_cost_mvt_avg=copw1[8,st_gm:]
kn_act_cost_mvt_std=copw1[9,st_gm:]

servpltle='Movement Cost'
if ch==0:
    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        ax.errorbar(alpha_values, np.asarray(act_cost_mvt_avg), yerr=np.asarray(act_cost_mvt_std),color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
        ax.errorbar(alpha_values, np.asarray(comp_cost_mvt_avg), yerr=np.asarray(comp_cost_mvt_std),color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
        
            #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(alpha_values, np.asarray(stat_cost_mvt_avg),yerr=np.asarray(stat_cost_mvt_std),color= '#F3A0F2',capsize=10, marker='^',linewidth=lwdth)
        ax.errorbar(alpha_values, np.asarray(servopt_cost_mvt_avg),yerr=np.asarray(servopt_cost_mvt_std),ls='dashed',color= 'y',capsize=10, marker='d',linewidth=lwdth)
            #ax.errorbar(alpha_values, np.asarray(pier_cost_mvt_avg),yerr=np.asarray(pier_cost_mvt_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(alpha_values, np.asarray(kn_act_cost_mvt_avg),yerr=np.asarray(kn_act_cost_mvt_std),ls='dashed',color= '#F5B14C',capsize=10, marker='*',linewidth=lwdth)
        ax.xaxis.set_label_coords(.5, -.05)    
        ax.legend([ algnm, compnm, 'Stationary', septnm, knalgnm], prop={'size': lsze},labelspacing=0.2)
        plt.title(servpltle, fontsize=asze)

        #plt.ylabel(servpltle, fontsize=20)
        plt.xlabel(r'Range of $\frac{\rho}{1+\rho}$', fontsize=asze-5)
        plt.xticks(np.arange(0,6)*0.2,('0','0.2','0.4','0.6','0.8','1'))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        plt.show()
        #plt.savefig('pltfrdata/t=800_f_n=10_n=400__Kernel=rbf_f_rep=2_%d_len_scale=0.200000_%s_2D.png' %(rand_int,wplt))

       
else:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(len(gam_val))
        ax.bar(X + 0.15,  np.asarray(act_cost_mvt_avg), yerr=np.asarray(act_cost_mvt_std), color = gpmd_col, width = 0.15)
        ax.bar(X + 0.00,  np.asarray(comp_cost_mvt_avg), yerr=np.asarray(comp_cost_mvt_std), color = cgp_col, width = 0.15)
        ax.bar(X + 0.30,  np.asarray(stat_cost_mvt_avg), yerr=np.asarray(stat_cost_mvt_std), color = '#F3A0F2', width = 0.15)
        ax.bar(X + 0.45,  np.asarray(servopt_cost_mvt_avg), yerr=np.asarray(servopt_cost_mvt_std), ls='dashed', facecolor='None',edgecolor = 'y', color='white',  hatch='//',  width = 0.15)
        ax.bar(X + 0.60,  np.asarray(kn_act_cost_mvt_avg), yerr=np.asarray(kn_act_cost_mvt_std), ls='dashed',facecolor='None',edgecolor = '#F5B14C', color='white', hatch='//', width = 0.15)
        
        
        plt.title('Total Energy Generated', fontsize=25)
        #plt.ylabel('Energy', fontsize=25)
        plt.ylim(bottom=1000)
        
        plt.title('Total Energy Generated', fontsize=25)
        #plt.ylabel('Energy', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=25)
        plt.xticks(X+0.30, gam_val)
        plt.yticks(np.arange(1000, 7000, 1000))
        plt.legend(labels=[algnm ,compnm,'Stationary',septnm,knalgnm ],  prop={'size': 15})
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))

wplt='tot'
copw1=pd.read_csv('%s/t=800_f_n=10_n=400__Kernel=rbf_f_rep=2_%d_mineps=14.000000_c_n=24_len_scale=0.200000_%s_2D.csv' %(fold_name,rand_int,wplt), sep=',',header=None)
copw1=np.asarray(copw1)

comp_cost_mvt_avg=copw1[0,st_gm:]
comp_cost_mvt_std=copw1[1,st_gm:]
act_cost_mvt_avg=copw1[2,st_gm:]
act_cost_mvt_std=copw1[3,st_gm:]
servopt_cost_mvt_avg=copw1[4,st_gm:]
servopt_cost_mvt_std=copw1[5,st_gm:]  
stat_cost_mvt_avg=copw1[6,st_gm:]
stat_cost_mvt_std=copw1[7,st_gm:]
kn_act_cost_mvt_avg=copw1[8,st_gm:]
kn_act_cost_mvt_std=copw1[9,st_gm:]

servpltle='Total Cost'
if ch==0:
    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        ax.errorbar(alpha_values, np.asarray(act_cost_mvt_avg), yerr=np.asarray(act_cost_mvt_std),color= gpmd_col,capsize=10, marker='s',linewidth=lwdth)
        ax.errorbar(alpha_values, np.asarray(comp_cost_mvt_avg), yerr=np.asarray(comp_cost_mvt_std),color=cgp_col,capsize=10, marker='o',linewidth=lwdth)
        
            #ax.errorbar(np.arange(0,T)[0::int(T/pl_freq)], rand_cost_tot_avg.numpy()[0::int(T/pl_freq)],yerr=rand_cost_tot_std.numpy()[0::int(T/pl_freq)], color='r',capsize=10, marker='v' )
        ax.errorbar(alpha_values, np.asarray(stat_cost_mvt_avg),yerr=np.asarray(stat_cost_mvt_std),color= '#F3A0F2',capsize=10, marker='^',linewidth=lwdth)
        ax.errorbar(alpha_values, np.asarray(servopt_cost_mvt_avg),yerr=np.asarray(servopt_cost_mvt_std),ls='dashed',color= 'y',capsize=10, marker='d',linewidth=lwdth)
            #ax.errorbar(alpha_values, np.asarray(pier_cost_mvt_avg),yerr=np.asarray(pier_cost_mvt_std),color= 'k',capsize=10, marker='x')
        ax.errorbar(alpha_values, np.asarray(kn_act_cost_mvt_avg),yerr=np.asarray(kn_act_cost_mvt_std),ls='dashed',color= '#F5B14C',capsize=10, marker='*',linewidth=lwdth)
        ax.xaxis.set_label_coords(.5, -.075)    
        ax.legend([ algnm, compnm, 'Stationary', septnm, knalgnm], prop={'size': lsze},labelspacing=0.2)
        plt.title(servpltle, fontsize=asze)

        #plt.ylabel(servpltle, fontsize=20)
        #plt.xlabel(r'Range of $\frac{\rho}{1+\rho}$', fontsize=20)
        #plt.xlabel(r'Increasing $\rho\;\rightarrow$', fontsize=20)
        plt.xlabel('Relative Service Cost Importance', fontsize=asze-8)
        plt.xticks(np.arange(0,6)*0.2,('0','0.2','0.4','0.6','0.8','1'))
        plt.xticks(fontsize=tkse)
        plt.yticks(fontsize=tkse)
        #plt.show()
        plt.savefig('pltfrdata/t=800_f_n=10_n=400__Kernel=rbf_f_rep=2_%d_len_scale=0.200000_%s_2D.png' %(rand_int,wplt))

       
else:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        X=np.arange(len(gam_val))
        ax.bar(X + 0.15,  np.asarray(act_cost_mvt_avg), yerr=np.asarray(act_cost_mvt_std), color = gpmd_col, width = 0.15)
        ax.bar(X + 0.00,  np.asarray(comp_cost_mvt_avg), yerr=np.asarray(comp_cost_mvt_std), color = cgp_col, width = 0.15)
        ax.bar(X + 0.30,  np.asarray(stat_cost_mvt_avg), yerr=np.asarray(stat_cost_mvt_std), color = '#F3A0F2', width = 0.15)
        ax.bar(X + 0.45,  np.asarray(servopt_cost_mvt_avg), yerr=np.asarray(servopt_cost_mvt_std), ls='dashed', facecolor='None',edgecolor = 'y', color='white',  hatch='//',  width = 0.15)
        ax.bar(X + 0.60,  np.asarray(kn_act_cost_mvt_avg), yerr=np.asarray(kn_act_cost_mvt_std), ls='dashed',facecolor='None',edgecolor = '#F5B14C', color='white', hatch='//', width = 0.15)
        
        
        plt.title('Total Energy Generated', fontsize=25)
        #plt.ylabel('Energy', fontsize=25)
        plt.ylim(bottom=1000)
        
        plt.title('Total Energy Generated', fontsize=25)
        #plt.ylabel('Energy', fontsize=25)
        plt.xlabel(r'$\rho$', fontsize=25)
        plt.xticks(X+0.30, gam_val)
        plt.yticks(np.arange(1000, 7000, 1000))
        plt.legend(labels=[ algnm ,compnm, septnm,'Stationary',knalgnm ],  prop={'size': 15})
        #plt.show()
        plt.savefig('newdirfont1216extrem/pltfrdata/t=%d_%d_init_t=%d_lat=%d_lon=%d_ls_t=%f_vr_1=0_bartotpow.png' %(T,rand_int,init_t,lat,lon,ls_1))




