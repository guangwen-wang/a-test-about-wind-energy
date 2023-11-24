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
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import cos, nonzero, random
from torch import serialization
import time
import os

from FRThelper import Tree

class RKHSModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(RKHSModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = RBFKernel()
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    def length_scale(self, value):
        self.covar_module.lengthscale = value 

likelihood = GaussianLikelihood()


class FRTOBJECT(Tree):
    def __init__(self,rootv,kn_root,p,kn_p,vld,ld,train_x_e,mind,n,n_e,eps,avg_mc,u7f,id_matrix,cost_ratio,beta,d_x,len_scale,kernel,map_2d,ini_x=None):
        self.root=rootv
        self.kn_root=kn_root
        self.a=ini_x
        self.p=p
        self.kn_p=kn_p
        self.vld=vld
        self.ld=ld
        self.train_x_e=train_x_e
        self.n=n
        self.n_e=n_e
        self.eps=eps
        self.avg_mc=avg_mc
        self.u7f=u7f
        self.id_matrix=id_matrix
        self.mind=mind
        self.cost_ratio=cost_ratio
        self.beta=beta
        self.len_scale=len_scale
        self.d_x=d_x
        self.kernel=kernel
        self.map_2d=map_2d
    def mirrordescent(frtob,f_st,p,xt_old,acroot):
        f_st=f_st
        eps=frtob.eps
        n=frtob.n
        vld=frtob.vld
        ld=frtob.ld
        acroot=acroot
        xt_old=xt_old
        id_matrix=frtob.id_matrix
        u7f=frtob.u7f
        p=p
        if torch.max(f_st)>eps:
            #print(f_st)
            m=torch.ceil(torch.max(f_st)/eps)
            #print(m)
            f_st=f_st/m
        else:
            m=1
        for u in range(0,n):
            nodeu=p[tuple(vld[u,ld])]#possible error in state
            #print(f_st[u])
            nodeu.cost=float(f_st[u])
        root=acroot
        for k_it in range(int(m)):#if cost is bigger than eps
            Tree.update_cprob_postorder(root)
            #print(m,"cst loop")  
        Tree.cprobtorprob_preorder(root)
        old_pt=[]
        nodeu=p[tuple(vld[xt_old,ld])]
        while(nodeu.parent!=None):
            old_pt.append(nodeu.id)
            nodeu=nodeu.parent
        old_pt=np.asarray(old_pt)
        #print(kn_old_pt,"oldpath")
        xt_int=Tree.ocsampler(root,np.flip(old_pt),np.flip(u7f))
        lp=[]
        lp_u=[]
        id=[]
        for u in range(n):
            nodeu=p[tuple(vld[u,ld])]
            lfid=nodeu.id
            if id_matrix[xt_int,lfid]==1:
                #print(nodeu.rprob)
                lp.append(nodeu.rprob)
                lp_u.append(u)
                id.append(nodeu.id)
                #print(id_matrix[xt_int,:])
                #print(kn_lp,kn_id,"leaves")

        if sum(lp)==0:#has to be changed
            print(lp,xt_int,id)
            print("weird",xt_int, id,old_pt)
            for i in range(len(lp)):
                lp[i]=1/len(lp)
        else:
            lp=lp/sum(lp)
        m=torch.distributions.categorical.Categorical(torch.tensor(lp))
        #print(lp)
        xt=lp_u[m.sample()]
        #print(p[tuple(vld[xt,ld])].id)
        return xt
    def FRTOT(frtob,T,train_y_e,gam):
        #print('welcome')
        torch.manual_seed((os.getpid()*gam*int(time.time()))%123456789)
        cost_ratio=frtob.cost_ratio
        acroot=frtob.root
        kn_acroot=frtob.kn_root
        a=frtob.a
        p=frtob.p
        kn_p=frtob.kn_p
        vld=frtob.vld
        ld=frtob.ld
        train_x_e=frtob.train_x_e
        n=frtob.n
        n_e=frtob.n_e
        eps=frtob.eps
        avg_mc=frtob.avg_mc
        u7f=frtob.u7f
        id_matrix=frtob.id_matrix
        beta=frtob.beta
        mind=frtob.mind
        d_x=frtob.d_x
        T=T
        train_y_e=train_y_e
        gam=gam
        len_scale=frtob.len_scale
        d_x=torch.from_numpy(d_x)
        kernel=frtob.kernel
        if a==None:
            a=int(np.floor(np.random.uniform(0,1)*n))
            for u in range(0,n):
                un=p[tuple(vld[u,ld])]
                un.rprob=0#initialize zero prob at all states except initial state
            p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
            rootv=acroot
            rootv.rprob=Tree.rprob_calc_nodes(rootv)
        else:
            for u in range(0,n):
                un=p[tuple(vld[u,ld])]
                un.rprob=0#initialize zero prob at all states except initial state
            p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
            rootv=acroot
            rootv.rprob=Tree.rprob_calc_nodes(rootv)
        #k=RBFKernel(lengthscale=0.2)
        #cov_em=k(train_x_e,train_x_e).evaluate()+0.001*torch.eye(n*n_e)#noise added to make sure matrix is positive definite
        
        #train_y_e=MultivariateNormal(torch.zeros(n*n_e),cov_em).sample()
        #build a GP function on the entire state and context space by sampling from multivariate gaussian with covariance as the kernel
        #if torch.min(train_y_e) < 0:
        #    train_y_e=train_y_e-torch.min(train_y_e)#increase min to 0
        #[comp_cost_mvt,act_cost_mvt,rand_cost_mvt,servopt_cost_mvt,comp_cost_serv,act_cost_serv,rand_cost_serv,servopt_cost_serv, stat_cost_mvt, stat_cost_serv]

        avg_sc=torch.sum(train_y_e)/(n*n_e)
        sgrid=train_x_e.reshape(n,n_e,3)
        x2d=sgrid[:,1,:]
        map_2d={}
        fgrid=train_y_e.reshape(n,n_e)*(1/mind)*gam*(avg_mc/avg_sc)

        #constants
        r_sq=0.01*(torch.max(train_y_e)-torch.min(train_y_e))*(1/mind)*gam*(avg_mc/avg_sc)
        noi_lm=torch.tensor(max(float(r_sq),pow(10,-3)))
        de_inf=0.1
        #b_sq=torch.matmul(torch.matmul(train_y_e.reshape(1,n*n_e)*(1/mind),k(train_x_e,train_x_e).evaluate()),train_y_e.reshape(n*n_e,1)*(1/mind))
        comp_cost_mvt,act_cost_mvt,rand_cost_mvt, servopt_cost_mvt, comp_cost_serv, act_cost_serv, rand_cost_serv, servopt_cost_serv, stat_cost_mvt, stat_cost_serv, pier_cost_mvt, pier_cost_serv, kn_act_cost_mvt, kn_act_cost_serv =  0,0,0,0,0,0,0,0,0,0,0,0,0,0        
        f_trep=2
        #print(sgrid)

        for f_rep in range(f_trep):

            #t=1
            for u in range(0,n):
                p[tuple(vld[u,ld])].rprob=0
                kn_p[tuple(vld[u,ld])].rprob=0#initialize zero prob at all states except initial state
            p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
            kn_p[tuple(vld[a,ld])].rprob=1
            rootv=acroot
            kn_root=kn_acroot
            rootv.rprob=Tree.rprob_calc_nodes(rootv)
            kn_root.rprob=Tree.rprob_calc_nodes(kn_root)
            et=int(torch.floor(torch.rand(1)*n_e))
            st=sgrid[:,et,:]
            f_st=torch.zeros(n,1)
            xt=FRTOBJECT.mirrordescent(frtob,f_st,p,a,acroot)
            #print(xt)
            train_x_gp=[st[a],st[xt]]
            yt=fgrid[xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
            train_y_gp=[yt]
            model = RKHSModel(st[xt].reshape(1,3), yt.reshape(1), likelihood)#building rkhs function
            model.covar_module.lengthscale=len_scale
            model.likelihood.noise=noi_lm
            model.eval()
            xt_old=xt

            #Known MTS mirror descent
            kn_f_st=fgrid[:,et]
            kn_xt=FRTOBJECT.mirrordescent(frtob,kn_f_st,kn_p,a,kn_acroot)
            #print(xt)
            kn_train_x_gp=[st[a],st[kn_xt]]
            kn_yt=fgrid[kn_xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
            kn_train_y_gp=[kn_yt]
            kn_xt_old=kn_xt






            #comparemodel
            likelihood_1=GaussianLikelihood()
            comp_xt=[st[a],st[torch.argmin(f_st)]]
            yt_cp=fgrid[torch.argmin(f_st),et]+torch.randn(1)*torch.sqrt(noi_lm)
            comp_yt=[yt_cp]
            model_cp=RKHSModel(st[torch.argmin(f_st)].reshape(1,3), yt_cp.reshape(1), likelihood_1)#building rkhs function
            model_cp.covar_module.lengthscale=len_scale
            model_cp.likelihood.noise=noi_lm
            model_cp.eval()

            #random model
            xt_rd=int(np.floor(np.random.uniform(0,1)*n))
            rand_xt=[st[a],st[xt_rd]]
            yt_rd=fgrid[xt_rd,et]+torch.randn(1)*torch.sqrt(noi_lm)
            rand_yt=[yt_rd]

            #piermodel
            likelihood_2=GaussianLikelihood()
            #print(d_x[a,:].squeeze())
            #print(d_x)
            #print(f_st.squeeze()+d_x[a,:].squeeze())
            xt_pier=torch.argmin(f_st.squeeze()+d_x[a,:].squeeze())
            pier_xt=[st[a],st[xt_pier]]
            yt_pier=fgrid[xt_pier,et]+torch.randn(1)*torch.sqrt(noi_lm)
            pier_yt=[yt_pier]
            model_pier=RKHSModel(st[xt_pier].reshape(1,3), yt_pier.reshape(1), likelihood_2)#building rkhs function
            model_pier.covar_module.lengthscale=len_scale
            model_pier.likelihood.noise=noi_lm
            model_pier.eval()

            #servopt model
            #likelihood_2=GaussianLikelihood()
            xt_so=torch.argmin(fgrid[:,et])
            servopt_xt=[st[a],st[xt_so]]
            yt_so=fgrid[xt_so,et]+torch.randn(1)*torch.sqrt(noi_lm)
            servopt_yt=[yt_so]

            #stationary model
            xt_stat=a
            stat_xt=[st[a],st[xt_stat]]
            yt_stat=fgrid[xt_stat,et]+torch.randn(1)*torch.sqrt(noi_lm)
            stat_yt=[yt_stat]

            #beta=5
            #print(stat_yt)

            for t in range(1,T):
                et=int(torch.floor(torch.rand(1)*n_e))#randomly generate context at each time t
                st=sgrid[:,et,:]#states
                #f_st=fgrid[:,et]#function value corresponding to the states above
                if beta == None:
                    beta=torch.sqrt(2*torch.log10(torch.tensor(((n*n_e)*t*t*torch.pi*torch.pi)/(6*de_inf)))/(torch.log10(torch.exp(torch.tensor(1.)))))
                f_st=torch.max(model(st.reshape(n,3)).mean-(beta)*model(st.reshape(n,3)).stddev,torch.tensor(0).float())
                #print(beta)
                xt=FRTOBJECT.mirrordescent(frtob,f_st,p,xt_old,acroot)
                #exploration coefficient
                if t<50:
                    exp_cf=0.0
                    if torch.rand(1)<exp_cf:
                        xt=torch.argmin(f_st)
                        print(xt)
                yt=fgrid[xt,et]+torch.randn(1)*torch.sqrt(noi_lm)

                train_x_gp.append(st[xt])
                train_y_gp.append(yt)
                #model = RKHSModel(torch.stack(train_x_gp).reshape(t+1,2), torch.stack(train_y_gp), likelihood)#building rkhs function
                #model.covar_module.lengthscale=0.2
                model=model.get_fantasy_model(st[xt].reshape(1,3),yt.reshape(1))
                model.eval()
                xt_old=xt

                #known mirror descent mts
                kn_f_st=fgrid[:,et]
                kn_xt=FRTOBJECT.mirrordescent(frtob,kn_f_st,kn_p,kn_xt_old,kn_acroot)
                kn_yt=fgrid[kn_xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
                kn_train_x_gp.append(st[kn_xt])
                kn_train_y_gp.append(kn_yt)
                kn_xt_old=kn_xt



                #print(model.parameters)
                #print(t,"time loop")
                #compare model
                f_st_cp=torch.max(model_cp(st.reshape(n,3)).mean-beta*model_cp(st.reshape(n,3)).stddev,torch.tensor(0).float())
                xt_cp=torch.argmin(f_st_cp)
                yt_cp=fgrid[xt_cp,et]+torch.randn(1)*torch.sqrt(noi_lm)
                comp_xt.append(st[xt_cp])
                comp_yt.append(yt_cp)
                model_cp=model_cp.get_fantasy_model(st[xt_cp].reshape(1,3),yt_cp.reshape(1))
                model_cp.eval()
                #print(t)

                #compare model
                f_st_pier=model_pier(st.reshape(n,3)).mean-beta*model_pier(st.reshape(n,3)).stddev
                xt_pier=torch.argmin(f_st_pier.squeeze()+d_x[xt_pier,:].squeeze())
                yt_pier=fgrid[xt_pier,et]+torch.randn(1)*torch.sqrt(noi_lm)
                pier_xt.append(st[xt_pier])
                pier_yt.append(yt_pier)
                model_pier=model_pier.get_fantasy_model(st[xt_pier].reshape(1,3),yt_pier.reshape(1))
                model_pier.eval()



                #random model
                xt_rd=int(np.floor(np.random.uniform(0,1)*n))
                yt_rd=fgrid[xt_rd,et]+torch.randn(1)*torch.sqrt(noi_lm)
                rand_xt.append(st[xt_rd])
                rand_yt.append(yt_rd)


                #service cost optimal model
                xt_so=torch.argmin(fgrid[:,et])
                servopt_xt.append(st[xt_so])
                yt_so=fgrid[xt_so,et]+torch.randn(1)*torch.sqrt(noi_lm)
                servopt_yt.append(yt_so)
                #model_so=model_so.get_fantasy_model(st[xt_so].reshape(1,2),yt_so.reshape(1))
                #model_so.eval()

                #stationary model
                xt_stat=a
                stat_xt.append(st[xt_stat])
                yt_stat=fgrid[xt_stat,et]+torch.randn(1)*torch.sqrt(noi_lm)
                stat_yt.append(yt_stat)

                #print(len(comp_xt),len(comp_yt))
            
            comp_cost_serv=comp_cost_serv+torch.sum(torch.stack(comp_yt)[:,0],dim=0).squeeze()

            act_cost_serv=act_cost_serv+torch.sum(torch.stack(train_y_gp)[:,0],dim=0).squeeze()

            rand_cost_serv=rand_cost_serv+torch.sum(torch.stack(rand_yt)[:,0],dim=0).squeeze()

            servopt_cost_serv=servopt_cost_serv+torch.sum(torch.stack(servopt_yt)[:,0],dim=0).squeeze()

            stat_cost_serv=stat_cost_serv+torch.sum(torch.stack(stat_yt)[:,0],dim=0).squeeze()

            pier_cost_serv=pier_cost_serv+torch.sum(torch.stack(pier_yt)[:,0],dim=0).squeeze()
            
            kn_act_cost_serv=kn_act_cost_serv+torch.sum(torch.stack(kn_train_y_gp)[:,0],dim=0).squeeze()


            comp_cost_mvt=comp_cost_mvt+torch.sum(torch.abs(torch.stack(comp_xt)[1:T+1,0]-torch.stack(comp_xt)[0:T,0]),dim=0)*(1/mind)

            act_cost_mvt=act_cost_mvt+torch.sum(torch.abs(torch.stack(train_x_gp)[1:T+1,0]-torch.stack(train_x_gp)[0:T,0]),dim=0)*(1/mind)

            rand_cost_mvt=rand_cost_mvt+torch.sum(torch.abs(torch.stack(rand_xt)[1:T+1,0]-torch.stack(rand_xt)[0:T,0]),dim=0)*(1/mind)

            servopt_cost_mvt=servopt_cost_mvt+torch.sum(torch.abs(torch.stack(servopt_xt)[1:T+1,0]-torch.stack(servopt_xt)[0:T,0]),dim=0)*(1/mind)

            stat_cost_mvt=stat_cost_mvt+torch.sum(torch.abs(torch.stack(stat_xt)[1:T+1,0]-torch.stack(stat_xt)[0:T,0]),dim=0)*(1/mind)

            pier_cost_mvt=pier_cost_mvt+torch.sum(torch.abs(torch.stack(pier_xt)[1:T+1,0]-torch.stack(pier_xt)[0:T,0]),dim=0)*(1/mind)
            
            kn_act_cost_mvt=kn_act_cost_mvt+torch.sum(torch.abs(torch.stack(kn_train_x_gp)[1:T+1,0]-torch.stack(kn_train_x_gp)[0:T,0]),dim=0)*(1/mind)


        ret_set=[comp_cost_mvt,act_cost_mvt,rand_cost_mvt,servopt_cost_mvt,stat_cost_mvt, comp_cost_serv,act_cost_serv,rand_cost_serv,servopt_cost_serv, stat_cost_serv, pier_cost_mvt, pier_cost_serv, kn_act_cost_serv, kn_act_cost_mvt]
        for rit in range(len(ret_set)):
            #print(ret_it)
            ret_set[rit]=ret_set[rit]/(f_trep*(1+gam))
            #print(ret_it,gam)
        #print(ret_set,"ret")

        return ret_set


    def FRTOT_fixgamma(frtob,T,gam,i):
        torch.manual_seed((os.getpid()*i*int(time.time()))%123456789)
        #print('good')
        cost_ratio=frtob.cost_ratio
        acroot=frtob.root
        kn_acroot=frtob.kn_root
        a=frtob.a
        p=frtob.p
        kn_p=frtob.kn_p
        vld=frtob.vld
        ld=frtob.ld
        train_x_e=frtob.train_x_e
        n=frtob.n
        n_e=frtob.n_e
        eps=frtob.eps
        avg_mc=frtob.avg_mc
        u7f=frtob.u7f
        id_matrix=frtob.id_matrix
        beta=frtob.beta
        mind=frtob.mind
        d_x=frtob.d_x
        len_scale=frtob.len_scale
        T=T
        kernel=frtob.kernel
        map_2d=frtob.map_2d
        #len_scale=len_scale
        d_x=torch.from_numpy(d_x)
        k=kernel
        cov_em=k(train_x_e,train_x_e).evaluate()+0.001*torch.eye(n*n_e)#noise added to make sure matrix is positive definite
        #print(cov_em)
        train_y_e=MultivariateNormal(torch.zeros(n*n_e),cov_em).sample()
        #print(train_y_e, train_y_e.shape)
        #build a GP function on the entire state and context space by sampling from multivariate gaussian with covariance as the kernel
        if torch.min(train_y_e) < 0:
            train_y_e=train_y_e-torch.min(train_y_e)#increase min to 0
        gam=gam
        if a==None:
            a=int(np.floor(np.random.uniform(0,1)*n))
            for u in range(0,n):
                un=p[tuple(vld[u,ld])]
                un.rprob=0#initialize zero prob at all states except initial state
            p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
            rootv=acroot
            rootv.rprob=Tree.rprob_calc_nodes(rootv)
        else:
            for u in range(0,n):
                un=p[tuple(vld[u,ld])]
                un.rprob=0#initialize zero prob at all states except initial state
            p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
            rootv=acroot
            rootv.rprob=Tree.rprob_calc_nodes(rootv)
        #k=RBFKernel(lengthscale=0.2)
        #cov_em=k(train_x_e,train_x_e).evaluate()+0.001*torch.eye(n*n_e)#noise added to make sure matrix is positive definite
        
        #train_y_e=MultivariateNormal(torch.zeros(n*n_e),cov_em).sample()
        #build a GP function on the entire state and context space by sampling from multivariate gaussian with covariance as the kernel
        #if torch.min(train_y_e) < 0:
        #    train_y_e=train_y_e-torch.min(train_y_e)#increase min to 0
        #[comp_cost_mvt,act_cost_mvt,rand_cost_mvt,servopt_cost_mvt,comp_cost_serv,act_cost_serv,rand_cost_serv,servopt_cost_serv, stat_cost_mvt, stat_cost_serv]


        #nodeu=p[tuple(vld[a,ld])]
        #kn_nodeu=kn_p[tuple(vld[a,ld])]
        #while nodeu.parent!=None:
            #print(nodeu.id,nodeu.rprob)
            #print(kn_nodeu.id,nodeu.rprob)
         #   nodeu=nodeu.parent
          #  kn_nodeu=kn_nodeu.parent

        avg_sc=torch.sum(train_y_e)/(n*n_e)
        #print(train_x_e)
        sgrid=train_x_e.reshape((n,n_e,3))###will it work???
        #print(sgrid,'sgrid')
        #print(sgrid[:,1,:],'one matrix')
        x2d=sgrid[:,1,:]
        #print(x2d[:,0:2])
        map_2d={}
        fgrid=train_y_e.reshape(n,n_e)*(1/mind)*gam*(avg_mc/avg_sc)
        #print(fgrid,train_y_e)
        #print(x2d,train_x_e)
        #print(fgrid.min(0))

        #constants
        r_sq=0.01*(torch.max(train_y_e)-torch.min(train_y_e))*(1/mind)*gam*(avg_mc/avg_sc)
        noi_lm=torch.tensor(max(float(r_sq),pow(10,-3)))
        de_inf=0.1
        #b_sq=torch.matmul(torch.matmul(train_y_e.reshape(1,n*n_e)*(1/mind),k(train_x_e,train_x_e).evaluate()),train_y_e.reshape(n*n_e,1)*(1/mind))
        comp_cost_mvt,act_cost_mvt,rand_cost_mvt, servopt_cost_mvt, comp_cost_serv, act_cost_serv, rand_cost_serv, servopt_cost_serv, stat_cost_mvt, stat_cost_serv, pier_cost_mvt, pier_cost_serv, kn_act_cost_mvt, kn_act_cost_serv =  torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T) ,torch.zeros(T) ,torch.zeros(T) ,torch.zeros(T) ,torch.zeros(T), torch.zeros(T)
        f_trep=3
       
        for f_rep in range(f_trep):

            #t=1
            for u in range(0,n):
                p[tuple(vld[u,ld])].rprob=0
                kn_p[tuple(vld[u,ld])].rprob=0#initialize zero prob at all states except initial state
            p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
            kn_p[tuple(vld[a,ld])].rprob=1
            rootv=acroot
            kn_root=kn_acroot
            rootv.rprob=Tree.rprob_calc_nodes(rootv)
            kn_root.rprob=Tree.rprob_calc_nodes(kn_root)
            et=int(torch.floor(torch.rand(1)*n_e))
            #print(et)
            st=sgrid[:,et,:]
            f_st=torch.zeros(n,1)
            xt=FRTOBJECT.mirrordescent(frtob,f_st,p,a,acroot)
            #print(xt)
            #a_2d=map_2d[a]
            #xt_2d=map_2d[xt]
            train_x_gp=[st[a],st[xt]]
            #print(train_x_gp)
            yt=fgrid[xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
            train_y_gp=[yt]
            #print(train_y_gp)
            model = RKHSModel(st[xt].reshape(1,3), yt.reshape(1), likelihood)#building rkhs function
            model.covar_module.lengthscale=len_scale
            model.likelihood.noise=noi_lm
            model.eval()
            xt_old=xt
            #print('over')

            kn_xt=FRTOBJECT.mirrordescent(frtob,f_st,kn_p,a,kn_acroot)
            kn_train_x_gp=[st[a],st[kn_xt]]
            kn_yt=fgrid[kn_xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
            kn_train_y_gp=[kn_yt]
            kn_xt_old=kn_xt
            #print(kn_train_y_gp)


            #comparemodel
            likelihood_1=GaussianLikelihood()
            comp_xt=[st[a],st[torch.argmin(f_st)]]
            yt_cp=fgrid[torch.argmin(f_st),et]+torch.randn(1)*torch.sqrt(noi_lm)
            comp_yt=[yt_cp]
            model_cp=RKHSModel(st[torch.argmin(f_st)].reshape(1,3), yt_cp.reshape(1), likelihood_1)#building rkhs function
            model_cp.covar_module.lengthscale=len_scale
            model_cp.likelihood.noise=noi_lm
            model_cp.eval()
            #print(comp_yt,'comp')


            #piermodel
            likelihood_2=GaussianLikelihood()
            #print(d_x[a,:].squeeze())
            #print(d_x)
            #print(f_st.squeeze()+d_x[a,:].squeeze())
            xt_pier=torch.argmin(f_st.squeeze()+d_x[a,:].squeeze())
            pier_xt=[st[a],st[xt_pier]]
            yt_pier=fgrid[xt_pier,et]+torch.randn(1)*torch.sqrt(noi_lm)
            pier_yt=[yt_pier]
            model_pier=RKHSModel(st[xt_pier].reshape(1,3), yt_pier.reshape(1), likelihood_2)#building rkhs function
            model_pier.covar_module.lengthscale=len_scale
            model_pier.likelihood.noise=noi_lm
            model_pier.eval()
            #print(pier_yt)


            #random model
            xt_rd=int(np.floor(np.random.uniform(0,1)*n))
            rand_xt=[st[a],st[xt_rd]]
            yt_rd=fgrid[xt_rd,et]+torch.randn(1)*torch.sqrt(noi_lm)
            rand_yt=[yt_rd]




            #servopt model
            #likelihood_2=GaussianLikelihood()
            xt_so=torch.argmin(fgrid[:,et])
            servopt_xt=[st[a],st[xt_so]]
            yt_so=fgrid[xt_so,et]+torch.randn(1)*torch.sqrt(noi_lm)
            servopt_yt=[yt_so]

            #stationary model
            xt_stat=a
            stat_xt=[st[a],st[xt_stat]]
            yt_stat=fgrid[xt_stat,et]+torch.randn(1)*torch.sqrt(noi_lm)
            stat_yt=[yt_stat]

            


            #beta=5

            for t in range(1,T):
                et=int(torch.floor(torch.rand(1)*n_e))#randomly generate context at each time t
                #print(et)
                st=sgrid[:,et,:]#states
                #f_st=fgrid[:,et]#function value corresponding to the states above
                if beta == None:
                    beta=torch.sqrt(2*torch.log10(torch.tensor(((n*n_e)*t*t*torch.pi*torch.pi)/(6*de_inf)))/(torch.log10(torch.exp(torch.tensor(1.)))))
                f_st=torch.max(model(st.reshape(n,3)).mean-(beta)*model(st.reshape(n,3)).stddev,torch.tensor(0).float())
                #print(beta)
                xt=FRTOBJECT.mirrordescent(frtob,f_st,p,xt_old,acroot)
                #exploration coefficient
                if t<50:
                    exp_cf=0.0
                    if torch.rand(1)<exp_cf:
                        xt=torch.argmin(f_st)
                        print(xt)
                yt=fgrid[xt,et]+torch.randn(1)*torch.sqrt(noi_lm)

                train_x_gp.append(st[xt])
                train_y_gp.append(yt)
                #model = RKHSModel(torch.stack(train_x_gp).reshape(t+1,2), torch.stack(train_y_gp), likelihood)#building rkhs function
                #model.covar_module.lengthscale=0.2
                model=model.get_fantasy_model(st[xt].reshape(1,3),yt.reshape(1))
                model.eval()
                xt_old=xt
                #print(train_x_gp,train_y_gp)

                #print('over')


                 #known mirror descent mts
                kn_f_st=fgrid[:,et]
                #print(beta)
                #print("kn1")
                kn_xt=FRTOBJECT.mirrordescent(frtob,kn_f_st,kn_p,kn_xt_old,kn_acroot)
                #print(t)
                #print("kn2")
                kn_yt=fgrid[kn_xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
                kn_train_x_gp.append(st[kn_xt])
                kn_train_y_gp.append(kn_yt)
                kn_xt_old=kn_xt

                #print('done')



                #print(model.paramete)
                #print(t,"time loop")
                #compare model
                f_st_cp=torch.max(model_cp(st.reshape(n,3)).mean-beta*model_cp(st.reshape(n,3)).stddev,torch.tensor(0).float())
                xt_cp=torch.argmin(f_st_cp)
                yt_cp=fgrid[xt_cp,et]+torch.randn(1)*torch.sqrt(noi_lm)
                comp_xt.append(st[xt_cp])
                comp_yt.append(yt_cp)
                model_cp=model_cp.get_fantasy_model(st[xt_cp].reshape(1,3),yt_cp.reshape(1))
                model_cp.eval()
                #print(t)
                #print(comp_yt,'comppy')

                #compare model
                f_st_pier=model_pier(st.reshape(n,3)).mean-beta*model_pier(st.reshape(n,3)).stddev
                xt_pier=torch.argmin(f_st_pier.squeeze()+d_x[xt_pier,:].squeeze())
                yt_pier=fgrid[xt_pier,et]+torch.randn(1)*torch.sqrt(noi_lm)
                pier_xt.append(st[xt_pier])
                pier_yt.append(yt_pier)
                model_pier=model_pier.get_fantasy_model(st[xt_pier].reshape(1,3),yt_pier.reshape(1))
                model_pier.eval()
                #print(pier_yt)

                #random model
                xt_rd=int(np.floor(np.random.uniform(0,1)*n))
                yt_rd=fgrid[xt_rd,et]+torch.randn(1)*torch.sqrt(noi_lm)
                rand_xt.append(st[xt_rd])
                rand_yt.append(yt_rd)


                #service cost optimal model
                xt_so=torch.argmin(fgrid[:,et])
                servopt_xt.append(st[xt_so])
                yt_so=fgrid[xt_so,et]+torch.randn(1)*torch.sqrt(noi_lm)
                servopt_yt.append(yt_so)
                #model_so=model_so.get_fantasy_model(st[xt_so].reshape(1,2),yt_so.reshape(1))
                #model_so.eval()

                #stationary model
                xt_stat=a
                stat_xt.append(st[xt_stat])
                yt_stat=fgrid[xt_stat,et]+torch.randn(1)*torch.sqrt(noi_lm)
                stat_yt.append(yt_stat)

            #print('done')
            comp_cost_serv=comp_cost_serv+torch.cumsum(torch.stack(comp_yt)[:,0],dim=0).squeeze()

            act_cost_serv=act_cost_serv+torch.cumsum(torch.stack(train_y_gp)[:,0],dim=0).squeeze()

            rand_cost_serv=rand_cost_serv+torch.cumsum(torch.stack(rand_yt)[:,0],dim=0).squeeze()

            servopt_cost_serv=servopt_cost_serv+torch.cumsum(torch.stack(servopt_yt)[:,0],dim=0).squeeze()

            stat_cost_serv=stat_cost_serv+torch.cumsum(torch.stack(stat_yt)[:,0],dim=0).squeeze()

            pier_cost_serv=pier_cost_serv+torch.cumsum(torch.stack(pier_yt)[:,0],dim=0).squeeze()

            kn_act_cost_serv=kn_act_cost_serv+torch.cumsum(torch.stack(kn_train_y_gp)[:,0],dim=0).squeeze()



            comp_cost_mvt=comp_cost_mvt+torch.cumsum(torch.sqrt(torch.pow(torch.stack(comp_xt)[1:T+1,0]-torch.stack(comp_xt)[0:T,0],2)+torch.pow(torch.stack(comp_xt)[1:T+1,1]-torch.stack(comp_xt)[0:T,1],2)),dim=0)*(1/mind)

            #print(comp_cost_mvt,"compmvtcost",comp_xt,mind)
            act_cost_mvt=act_cost_mvt+torch.cumsum(torch.sqrt(torch.pow(torch.stack(train_x_gp)[1:T+1,0]-torch.stack(train_x_gp)[0:T,0],2)+torch.pow(torch.stack(train_x_gp)[1:T+1,1]-torch.stack(train_x_gp)[0:T,1],2)),dim=0)*(1/mind)

            #print(act_cost_mvt,"actmvtcost",train_x_gp,mind)

            rand_cost_mvt=rand_cost_mvt+torch.cumsum(torch.sqrt(torch.pow(torch.stack(rand_xt)[1:T+1,0]-torch.stack(rand_xt)[0:T,0],2)+torch.pow(torch.stack(rand_xt)[1:T+1,1]-torch.stack(rand_xt)[0:T,1],2)),dim=0)*(1/mind)

            servopt_cost_mvt=servopt_cost_mvt+torch.cumsum(torch.sqrt(torch.pow(torch.stack(servopt_xt)[1:T+1,0]-torch.stack(servopt_xt)[0:T,0],2)+torch.pow(torch.stack(servopt_xt)[1:T+1,1]-torch.stack(servopt_xt)[0:T,1],2)),dim=0)*(1/mind)

            stat_cost_mvt=stat_cost_mvt+torch.cumsum(torch.sqrt(torch.pow(torch.stack(stat_xt)[1:T+1,0]-torch.stack(stat_xt)[0:T,0],2)+torch.pow(torch.stack(stat_xt)[1:T+1,1]-torch.stack(stat_xt)[0:T,1],2)),dim=0)*(1/mind)

            pier_cost_mvt=pier_cost_mvt+torch.cumsum(torch.sqrt(torch.pow(torch.stack(pier_xt)[1:T+1,0]-torch.stack(pier_xt)[0:T,0],2)+torch.pow(torch.stack(pier_xt)[1:T+1,1]-torch.stack(pier_xt)[0:T,1],2)),dim=0)*(1/mind)
 
            kn_act_cost_mvt=kn_act_cost_mvt+torch.cumsum(torch.sqrt(torch.pow(torch.stack(kn_train_x_gp)[1:T+1,0]-torch.stack(kn_train_x_gp)[0:T,0],2)+torch.pow(torch.stack(kn_train_x_gp)[1:T+1,1]-torch.stack(kn_train_x_gp)[0:T,1],2)),dim=0)*(1/mind)


        ret_set=[comp_cost_mvt,act_cost_mvt,rand_cost_mvt,servopt_cost_mvt,stat_cost_mvt, comp_cost_serv,act_cost_serv,rand_cost_serv,servopt_cost_serv, stat_cost_serv, pier_cost_mvt, pier_cost_serv, kn_act_cost_mvt, kn_act_cost_serv]
        for rit in range(len(ret_set)):
            #print(ret_it)
            ret_set[rit]=ret_set[rit]/(f_trep)
            #print(ret_it,gam)

        return ret_set




