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
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import cos, nonzero, random
from torch import serialization
import time
import os
import netCDF4 as nc


from FRThelper import Tree

class RKHSModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(RKHSModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()

        self.covar_module =  ScaleKernel(RBFKernel(),active_dims=torch.tensor([0,1]))#+ScaleKernel(RBFKernel()) #+ ScaleKernel(RBFKernel(active_dims=torch.tensor([2])))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    def length_scale(self, value):
        self.covar_module.lengthscale = value 

likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.01))


class FRTOBJECT(Tree):
    def __init__(self,rootv,kn_root,p,kn_p,vld,ld,train_x_e,train_x_e_1,mind,n,n_e,eps,avg_mc,u7f,id_matrix,cost_ratio,beta,d_x,vw,ini_x=None):
        self.root=rootv
        self.kn_root=kn_root
        self.a=ini_x
        self.p=p
        self.kn_p=kn_p
        self.vld=vld
        self.ld=ld
        self.train_x_e=train_x_e
        self.train_x_e_1=train_x_e_1
        self.n=n
        self.n_e=n_e
        self.eps=eps
        self.avg_mc=avg_mc
        self.u7f=u7f
        self.id_matrix=id_matrix
        self.mind=mind
        self.cost_ratio=cost_ratio
        self.beta=beta
        self.d_x=d_x
        self.vw=vw
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
    def build_rkhs_model(train_x,train_y,likelihood,t,ls_1):
        ls_0=3.67#ls_1#3.67#1.6#2#3.67#0.78#1.78#0.78#1.48#0.78#1.75#0.78#1.52
        ls_1=ls_1#0.32#0.0016#0.09#0.0016#0.13#0.0016#0.047#0.0016#0.0072
        #ls_2=0.01
        vr_0=ls_1#6.85#60#50#100#70#70#50#12.94#9.86#12.94#9.52#12.94#19.56#12.94#8.46
        vr_1=0#1.783#5#1#8#1.7#4#4#1.87#1.56#1.87#1.07#1.87#0.70#1.87#1.14
        #vr_2=7.18
        noi_lm=2.73#15#7#15#15#2.73#3.97#2.87#3.97#3.5#3.97#4.86#3.97#3.27
        #print(ls_1)
        model = RKHSModel(train_x.reshape(t+1,3).float(), train_y.reshape(t+1).float(), likelihood,)#building rkhs function
        #for param_name, param in model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.item()}','before')
        #print(model.covar_module)
        #model.covar_module.kernels[0].base_kernel.raw_lengthscale=torch.nn.Parameter(torch.tensor([ls_0]))
        #model.covar_module.kernels[0].base_kernel.lengthscale=ls_0
        model.covar_module.base_kernel.lengthscale=ls_0
        #for param_name, param in model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.item()}')
        #model.covar_module.kernels[1].base_kernel.raw_lengthscale=torch.nn.Parameter(torch.tensor([ls_1]))
        #model.covar_module.kernels[1].base_kernel.lengthscale=ls_1
        #for param_name, param in model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.item()}')
        #model.covar_module.kernels[1].base_kernel.lengthscale=ls_2
        #model.covar_module.kernels[0].raw_outputscale=torch.nn.Parameter(torch.tensor([vr_0]))
        #model.covar_module.kernels[0].outputscale=vr_0
        model.covar_module.outputscale=vr_0
        #for param_name, param in model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.item()}')
        #model.covar_module.kernels[1].raw_outputscale=torch.nn.Parameter(torch.tensor([vr_1]))
        #model.covar_module.kernels[1].outputscale=vr_1
        #for param_name, param in model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.item()}')
        #model.covar_module.kernels[1].outputscale=vr_2
        #model.covar_module.kernels[0].kernels[0].base_kernel.lengthscale=ls_0
        #model.covar_module.kernels[0].kernels[1].base_kernel.lengthscale=ls_1
        #model.covar_module.kernels[1].base_kernel.lengthscale=ls_2
        #model.covar_module.kernels[0].kernels[0].outputscale=vr_0
        #model.covar_module.kernels[0].kernels[1].outputscale=vr_1
        #model.covar_module.kernels[1].outputscale=vr_2
        model.likelihood.noise=noi_lm
        #for param_name, param in model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.item()}')
        return model
    def func_train_model(model,likelihood,train_x,train_y,training_iter):
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        #print(train_x)
        train_x=torch.stack(train_x[1:])
        train_y=torch.stack(train_y)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            #print(train_x)
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y.float())
            loss.sum().backward()
            print('Iter %d/%d - Loss: %.3f noise: %.3f' % (i + 1, training_iter, loss.item(), model.likelihood.noise.item()    ))
            optimizer.step()
            if training_iter%2==0:
                print(loss)
                for param_name, param in model.named_parameters():
                    print(f'Parameter name: {param_name:42} value = {param.item()}')
        return model
    def FRTOT(frtob,T,train_y_e,gam):
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
        d_x=torch.from_numpy(d_x)
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
        sgrid=train_x_e.reshape(n,n_e,2)
        fgrid=train_y_e.reshape(n,n_e)*(1/mind)*gam*(avg_mc/avg_sc)
        #print(fgrid.min(0))
        #in_comp=torch.tensor(np.minimum(12*np.ones((n_e,n)),vw))

        #constants
        r_sq=0.01*(torch.max(train_y_e)-torch.min(train_y_e))*(1/mind)*gam*(avg_mc/avg_sc)
        noi_lm=torch.tensor(max(float(r_sq),pow(10,-3)))
        de_inf=0.1
        #b_sq=torch.matmul(torch.matmul(train_y_e.reshape(1,n*n_e)*(1/mind),k(train_x_e,train_x_e).evaluate()),train_y_e.reshape(n*n_e,1)*(1/mind))
        comp_cost_mvt,act_cost_mvt,rand_cost_mvt, servopt_cost_mvt, comp_cost_serv, act_cost_serv, rand_cost_serv, servopt_cost_serv, stat_cost_mvt, stat_cost_serv, pier_cost_mvt, pier_cost_serv, kn_act_cost_mvt, kn_act_cost_serv =  0,0,0,0,0,0,0,0,0,0,0,0,0,0        
        f_trep=3
        training_iter=50
        print(sgrid)

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
            et=0
            st=sgrid[:,et]
            f_st=torch.zeros(n,1)
            xt=FRTOBJECT.mirrordescent(frtob,f_st,p,a,acroot)
            #print(xt)
            print(st)
            train_x_gp=[st[a],st[xt]]
            yt=fgrid[xt,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            train_y_gp=[yt]
            model = RKHSModel(st[xt].reshape(1,2), yt.reshape(1), likelihood)#building rkhs function
            #model.covar_module.lengthscale=len_scale
            #model.likelihood.noise=noi_lm
            print(model.parameters)
            model.train()
            likelihood.train()
            model=FRTOBJECT.func_train_model(model,likelihood,train_x_gp,train_y_gp,training_iter)
            model.eval()
            xt_old=xt

            #Known MTS mirror descent
            kn_xt=FRTOBJECT.mirrordescent(frtob,f_st,kn_p,a,kn_acroot)
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
            model_cp=RKHSModel(st[torch.argmin(f_st)].reshape(1,2), yt_cp.reshape(1), likelihood_1)#building rkhs function
            #model_cp.covar_module.lengthscale=len_scale
            #model_cp.likelihood.noise=noi_lm
            model_cp.train()
            likelihood_1.train()
            model_cp=FRTOBJECT.func_train_model(model_cp,likelihood_1,comp_xt,comp_yt,training_iter)
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
            yt_pier=fgrid[xt_pier,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            pier_yt=[yt_pier]
            model_pier=RKHSModel(st[xt_pier].reshape(1,2), yt_pier.reshape(1), likelihood_2)#building rkhs function
            #model_pier.covar_module.lengthscale=len_scale
            #model_pier.likelihood.noise=noi_lm
            model_pier.train()
            likelihood_2.train()
            model_pier=FRTOBJECT.func_train_model(model_pier,likelihood_2,pier_xt,pier_yt,training_iter)
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

            for t in range(1,T):
                et=t
                st=sgrid[:,et]#states
                #f_st=fgrid[:,et]#function value corresponding to the states above
                if beta == None:
                    beta=torch.sqrt(2*torch.log10(torch.tensor(((n*n_e)*t*t*torch.pi*torch.pi)/(6*de_inf)))/(torch.log10(torch.exp(torch.tensor(1.)))))
                f_st=torch.max(model(st.reshape(n,2)).mean-(beta)*model(st.reshape(n,2)).stddev,torch.tensor(0).float())
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
                model=model.get_fantasy_model(st[xt].reshape(1,2),yt.reshape(1))
                model.train()
                likelihood.train()
                model=FRTOBJECT.func_train_model(model,likelihood,train_x_gp,train_y_gp,training_iter)
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
                f_st_cp=torch.max(model_cp(st.reshape(n,2)).mean-beta*model_cp(st.reshape(n,2)).stddev,torch.tensor(0).float())
                xt_cp=torch.argmin(f_st_cp)
                yt_cp=fgrid[xt_cp,et]+torch.randn(1)*torch.sqrt(noi_lm)
                comp_xt.append(st[xt_cp])
                comp_yt.append(yt_cp)
                model_cp=model_cp.get_fantasy_model(st[xt_cp].reshape(1,2),yt_cp.reshape(1))
                model_cp.train()
                likelihood_1.train()
                model_cp=FRTOBJECT.func_train_model(model_cp,likelihood_1,comp_xt,comp_yt,training_iter)
                model_cp.eval()
                #print(t)

                #compare model
                f_st_pier=model_pier(st.reshape(n,2)).mean-beta*model_pier(st.reshape(n,2)).stddev
                xt_pier=torch.argmin(f_st_pier.squeeze()+d_x[xt_pier,:].squeeze())
                yt_pier=fgrid[xt_pier,et]+torch.randn(1)*torch.sqrt(noi_lm)
                pier_xt.append(st[xt_pier])
                pier_yt.append(yt_pier)
                model_pier=model_pier.get_fantasy_model(st[xt_pier].reshape(1,2),yt_pier.reshape(1))
                model_pier.train()
                likelihood_2.train()
                model_pier=FRTOBJECT.func_train_model(model_pier,likelihood_2,pier_xt,pier_yt,training_iter)
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

        return ret_set


    def FRTOT_fixgamma(frtob,T,gam,i,st_ti,ls_1):
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
        train_x_e_1=frtob.train_x_e_1
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
        vw=frtob.vw
        ls_1=ls_1
        #len_scale=len_scale
        d_x=torch.from_numpy(d_x)
        #k=kernel(lengthscale=len_scale)
        #cov_em=k(train_x_e,train_x_e).evaluate()#+0.001*torch.eye(n*n_e)#noise added to make sure matrix is positive definite
        #print(cov_em)
        #train_y_e=MultivariateNormal(torch.zeros(n*n_e),cov_em).sample()
        vw=vw[:,st_ti:st_ti+T]
        train_y_e=torch.tensor(vw)
        ###to be written

        
        
        
        
        
        
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

        #avg_sc=torch.pow(torch.sum(train_y_e)/(n*n_e),2)
        rt=gam
        nd=int(np.floor(T/24))
        sgrid=train_x_e.reshape((n,n_e,nd,3))
        sgrid_nm=train_x_e_1.reshape((n,n_e,nd,3))
        v_rtd=torch.tensor(12)
        in_comp=np.minimum(train_y_e.reshape(n,T),12*np.ones((n,T)))
        v_r2=pow(12,2)
        #print(avg_sc)
        fgrid=(torch.pow(in_comp,3)*0.0579-torch.pow(train_y_e.reshape(n,T),2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)#to compensate for the scaling in distance matrix in FRT algorithm
        #fgrid=train_y_e
        fmax=(torch.pow(v_rtd,3)*0.0579-torch.pow(v_rtd,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
        #print(fgrid)
        #print(d_x)
        #print(torch.sum(fgrid)/(n*n_e))
        #print(torch.sum(d_x)/(n*n))
        #print(fgrid.min(0)
        noi_lm=torch.tensor(0.6316)
        #constants
        #r_sq=0.01*(torch.max(fgrid)-torch.min(fgrid))
        #noi_lm=torch.tensor(max(float(r_sq),pow(10,-3)))
        de_inf=0.1
        #b_sq=torch.matmul(torch.matmul(train_y_e.reshape(1,n*n_e)*(1/mind),k(train_x_e,train_x_e).evaluate()),train_y_e.reshape(n*n_e,1)*(1/mind))
        comp_cost_mvt,act_cost_mvt,rand_cost_mvt, servopt_cost_mvt, comp_cost_serv, act_cost_serv, rand_cost_serv, servopt_cost_serv, stat_cost_mvt, stat_cost_serv, pier_cost_mvt, pier_cost_serv, kn_act_cost_mvt, kn_act_cost_serv =  torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T) ,torch.zeros(T) ,torch.zeros(T) ,torch.zeros(T) ,torch.zeros(T), torch.zeros(T)
        comp_power_serv, act_power_serv, servopt_power_serv, stat_power_serv, pier_power_serv, kn_act_power_serv =  torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T), torch.zeros(T)
        
        f_trep=3
        training_iter=50
        xmn=568.166
        xsd=461.945
        emn=11.9
        esd=7.28
        dmn=2.03
        dsd=1.39
        fmn=0
        fsd=1
        fgrid_nm=(fgrid-fmn)/fsd
        vw=torch.tensor(vw)
        a=i
        #print('3')
        for f_rep in range(f_trep):

            #t=1
            #a=int(np.floor(np.random.uniform(0,1)*n))
            #print(a,'initial state')
            for u in range(0,n):
                p[tuple(vld[u,ld])].rprob=0
                kn_p[tuple(vld[u,ld])].rprob=0#initialize zero prob at all states except initial state
            p[tuple(vld[a,ld])].rprob=1#1 prob at initial state and x_a[a] is the initial point
            kn_p[tuple(vld[a,ld])].rprob=1
            rootv=acroot
            kn_root=kn_acroot
            rootv.rprob=Tree.rprob_calc_nodes(rootv)
            kn_root.rprob=Tree.rprob_calc_nodes(kn_root)
            et=0
            tet=int(et%24)
            det=int(np.floor(et/24))
            st=sgrid[:,tet,det]
            st_nm=sgrid_nm[:,tet,det]
            f_st=torch.zeros(n,1)
            xt=FRTOBJECT.mirrordescent(frtob,f_st,p,a,acroot)
            #print(xt)
            train_x_gp=[st[a],st[xt]]
            train_x_gp_nm=[st_nm[a],st_nm[xt]]
            #yt=fgrid[xt,et]#+torch.randn(1)*torch.sqrt(noi_lm)#should noise be normalized or unnormalized
            yt=vw[xt,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            yt_nm=(yt-fmn)/fsd#add noise after normalization
            in_comp=np.minimum(yt,12)
            #print(avg_sc)
            ft=(torch.pow(in_comp,3)*0.0579-torch.pow(yt,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind) 
            train_y_gp_power=[ft]
            train_y_gp=[yt]
            train_y_gp_cost=[torch.max(fgrid[:,et])-ft]
            
            train_y_gp_nm=[yt_nm]
            model = FRTOBJECT.build_rkhs_model(st_nm[xt], yt_nm, likelihood,0,ls_1)#building rkhs function
            model.eval()
            #model.train()
            #likelihood.train()
            #model=FRTOBJECT.func_train_model(model,likelihood,train_x_gp,train_y_gp,training_iter)
          
            #how to set parameters
            
            
            xt_old=xt
            #print('over')
            #kn_f_st=fgrid[:,et]-fmin
            kn_f_st=fgrid[:,et]
            #kn_f_st=fmax-kn_f_st
            kn_f_st=torch.max(fgrid[:,et])-kn_f_st
            kn_xt=FRTOBJECT.mirrordescent(frtob,kn_f_st,kn_p,a,kn_acroot)
            kn_train_x_gp=[st[a],st[kn_xt]]
            kn_yt=fgrid[kn_xt,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            #kn_yt=vw[kn_xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
            kn_train_y_gp=[kn_yt]
            kn_train_y_gp_power=[kn_yt]
            kn_train_y_cost=[torch.max(fgrid[:,et])-kn_yt]
            kn_xt_old=kn_xt


            #comparemodel
            likelihood_1=GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.01))
            xt_cp=torch.argmin(f_st)
            comp_xt=[st[a],st[xt_cp]]
            comp_xt_nm=[st_nm[a],st_nm[xt_cp]]
            yt_cp=vw[xt_cp,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            #yt_cp=vw[xt_cp,et]+torch.randn(1)*torch.sqrt(noi_lm)
            yt_cp_nm=(yt_cp-fmn)/fsd
            comp_yt=[yt_cp]
            in_comp=np.minimum(yt_cp,12)
            ft_comp=(torch.pow(in_comp,3)*0.0579-torch.pow(yt_cp,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
            comp_yt_power=[ft_comp]
            comp_yt_cost=[torch.max(fgrid[:,et])-ft_comp]
            comp_yt_nm=[yt_cp_nm]
            model_cp=FRTOBJECT.build_rkhs_model(st_nm[xt_cp], yt_cp_nm, likelihood_1,0,ls_1)#building rkhs function
            #model_cp.train()
            #likelihood_1.train()
            #model_cp=FRTOBJECT.func_train_model(model_cp,likelihood_1,comp_xt,comp_yt,training_iter)
            model_cp.eval()

            #find power learned wind

            #piermodel
            likelihood_2=GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.01))
            xt_pier=torch.argmin(f_st.squeeze()+d_x[a,:].squeeze())
            pier_xt=[st[a],st[xt_pier]]
            pier_xt_nm=[st_nm[a],st_nm[xt_pier]]
            yt_pier=vw[xt_pier,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            #yt_pier=vw[xt_pier,et]+torch.randn(1)*torch.sqrt(noi_lm)
            yt_pier_nm=(yt_pier-fmn)/fsd
            pier_yt=[yt_pier]
            in_comp=np.minimum(yt_pier,12)
            ft_pier=(torch.pow(in_comp,3)*0.0579-torch.pow(yt_pier,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
            pier_yt_power=[ft_pier]
            pier_yt_cost=[torch.max(fgrid[:,et])-ft_pier]
            pier_yt_nm=[yt_pier_nm]
            model_pier=FRTOBJECT.build_rkhs_model(st_nm[xt_pier], yt_pier_nm, likelihood_2,0,ls_1)#building rkhs function
            #model_pier.train()
            #likelihood_2.train()
            #model_pier=FRTOBJECT.func_train_model(model_pier,likelihood_2,pier_xt,pier_yt,training_iter)
            model_pier.eval()


            #random model
            xt_rd=int(np.floor(np.random.uniform(0,1)*n))
            rand_xt=[st[a],st[xt_rd]]
            yt_rd=fgrid[xt_rd,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            rand_yt=[yt_rd]



            #servopt model
            #likelihood_2=GaussianLikelihood()
            #xt_so=torch.argmin(fgrid[:,et])
            xt_so=torch.argmin(torch.max(fgrid[:,et])-fgrid[:,et])
            servopt_xt=[st[a],st[xt_so]]
            yt_so=fgrid[xt_so,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            #yt_so=vw[servopt_xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
            servopt_yt=[yt_so]
            servopt_yt_power=[yt_so]
            servopt_yt_cost=[torch.max(fgrid[:,et])-yt_so]

            #stationary model
            xt_stat=a
            stat_xt=[st[a],st[xt_stat]]
            yt_stat=fgrid[xt_stat,et]#+torch.randn(1)*torch.sqrt(noi_lm)
            #yt_pier=vw[stat_xt,et]+torch.randn(1)*torch.sqrt(noi_lm)
            stat_yt=[yt_stat]
            stat_yt_power=[yt_stat]
            stat_yt_cost=[torch.max(fgrid[:,et])-yt_stat]

            #beta=5

            for t in range(1,T):
                #change power output to cost
                #print(t)
                et=t#randomly generate context at each time t
                tet=int(et%24)
                det=int(np.floor(et/24))
                st=sgrid[:,tet,det]
                st_nm=sgrid_nm[:,tet,det]
                #st=sgrid[:,et]#states
                #f_st=fgrid[:,et]#function value corresponding to the states above
                if beta == None:
                    beta=torch.sqrt(2*torch.log10(torch.tensor(((n*n_e)*t*t*torch.pi*torch.pi)/(6*de_inf)))/(torch.log10(torch.exp(torch.tensor(1.)))))
                #f_st=torch.max(model(st.reshape(n,3)).mean-(beta)*model(st.reshape(n,3)).stddev,torch.tensor(0).float())
                #f_st_nm=torch.max(model(st_nm.reshape(n,3)).mean-(beta)*model(st_nm.reshape(n,3)).stddev,torch.tensor(0).float())
                f_st_nm=model(st_nm.reshape(n,3)).mean+(beta)*model(st_nm.reshape(n,3)).stddev
                #print(model(st_nm.reshape(n,3)).stddev)
                f_st=f_st_nm*fsd+fmn
                f_st_nm_2=model(st_nm.reshape(n,3)).mean-(beta)*model(st_nm.reshape(n,3)).stddev
                f_st_2=f_st_nm_2*fsd+fmn
                b=(f_st-12*torch.ones(n))*(f_st_2-12*torch.ones(n))
                b[b<0]=0
                ind=(b==0).nonzero()
                #print(f_st,'without subtraction')
                #in_comp=np.minimum(f_st.reshape(n,1),12*np.ones((n,1)))
                #print(avg_sc)
                #f_st=((torch.pow(12*torch.ones(n,1),3)-torch.pow(in_comp,3))*0.0579+torch.pow(f_st.reshape(n,1),2)*0.09)*(1/0.15)*30*0.01*(1/mind)
                #f_st=torch.max(fmax-f_st,torch.tensor(0).float())
                #f_st=torch.max(torch.max(fgrid[:,et])-f_st,torch.tensor(0).float())
                in_comp=torch.min(f_st,12*torch.ones(n))
                f_st=(torch.pow(in_comp,3)*0.0579-torch.pow(f_st,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
                
                in_comp_2=torch.min(f_st_2,12*torch.ones(n))
                f_st_2=(torch.pow(in_comp_2,3)*0.0579-torch.pow(f_st_2,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
                f_st=torch.max(f_st,f_st_2)
                f_st[ind]=fmax
                f_st=torch.max(f_st)-f_st
                #why not maximum in this vector??
                #convert f_st to cost
                #f_st=fmax-fgrid[:,et]
                xt=FRTOBJECT.mirrordescent(frtob,f_st,p,xt_old,acroot)
                #exploration coefficient
                if t<50:
                    exp_cf=0.0
                    if torch.rand(1)<exp_cf:
                        xt=torch.argmin(f_st)
                        print(xt)
                yt=vw[xt,et]#+torch.randn(1)*torch.sqrt(noi_lm)
                yt_nm=(yt-fmn)/fsd
                in_comp=np.minimum(yt,12)
                #print(avg_sc)
                ft=(torch.pow(in_comp,3)*0.0579-torch.pow(yt,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
                train_x_gp.append(st[xt])
                train_y_gp.append(yt)
                train_y_gp_power.append(ft)
                train_y_gp_cost.append(torch.max(fgrid[:,et])-ft)
                train_x_gp_nm.append(st_nm[xt])
                train_y_gp_nm.append(yt_nm)
               
                #    model.train()
                #    likelihood.train()
                #    model=FRTOBJECT.func_train_model(model,likelihood,train_x_gp,train_y_gp,training_iter)
                #print(ls_1,'ls_1')
                model = FRTOBJECT.build_rkhs_model(torch.stack(train_x_gp_nm[1:]), torch.stack(train_y_gp), likelihood,t,ls_1)#building rkhs function
                #print(model.covar_module)
                model.eval()
                xt_old=xt

                #print('over')


                 #known mirror descent mts
                #subtract by the maximum at each time step
                #kn_f_st=fmax-fgrid[:,et]
                kn_f_st=torch.max(fgrid[:,et])-fgrid[:,et]
                #print(beta)
                #print("kn1")
                #print(fgrid[:,et])
                kn_xt=FRTOBJECT.mirrordescent(frtob,kn_f_st,kn_p,kn_xt_old,kn_acroot)
                #print(t)
                #print("kn2")
                kn_yt=fgrid[kn_xt,et]#+torch.randn(1)*torch.sqrt(noi_lm)
                kn_train_x_gp.append(st[kn_xt])
                kn_train_y_gp.append(kn_yt)
                kn_train_y_gp_power.append(kn_yt)
                kn_train_y_cost.append(torch.max(fgrid[:,et])-kn_yt)
                kn_xt_old=kn_xt

                #print('done')



                #print(model.paramete)
                #print(t,"time loop")
                #compare model
                #f_st_cp=torch.max(model_cp(st.reshape(n,3)).mean-beta*model_cp(st.reshape(n,3)).stddev,torch.tensor(0).float())
                f_st_cp_nm=model_cp(st_nm.reshape(n,3)).mean+beta*model_cp(st_nm.reshape(n,3)).stddev
                f_st_cp=f_st_cp_nm*fsd+fmn
                #print(f_st_cp)
                f_st_cp_nm_2=model_cp(st_nm.reshape(n,3)).mean-beta*model_cp(st_nm.reshape(n,3)).stddev
                f_st_cp_2=f_st_cp_nm_2*fsd+fmn
                #f_st_cp=torch.max(fmax-f_st_cp,torch.tensor(0).float())
                b_cp=(f_st_cp-12*torch.ones(n))*(f_st_cp_2-12*torch.ones(n))
                b_cp[b_cp<0]=0
                ind_cp=(b_cp==0).nonzero()
                in_comp=torch.min(f_st_cp,12*torch.ones(n))
                f_st_cp=(torch.pow(in_comp,3)*0.0579-torch.pow(f_st_cp,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
                #f_st_cp=torch.max(fmax-f_st_cp,torch.tensor(0).float())
                in_comp_2=torch.min(f_st_cp_2,12*torch.ones(n))
                f_st_cp_2=(torch.pow(in_comp_2,3)*0.0579-torch.pow(f_st_cp_2,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
                f_st_cp=torch.max(f_st_cp,f_st_cp_2)
                f_st_cp[ind_cp]=fmax
                f_st_cp=torch.max(fgrid[:,et])-f_st_cp
                #print(f_st_cp)
                xt_cp=torch.argmin(f_st_cp)
                #print(xt_cp,'3')
                yt_cp=vw[xt_cp,et]#+torch.randn(1)*torch.sqrt(noi_lm)
                comp_xt.append(st[xt_cp])
                comp_yt.append(yt_cp)
                in_comp=np.minimum(yt_cp,12)
                #print(avg_sc)
                ft_cp=(torch.pow(in_comp,3)*0.0579-torch.pow(yt_cp,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
                comp_yt_cost.append(torch.max(fgrid[:,et])-ft_cp)
                comp_yt_power.append(ft_cp)
                yt_cp_nm=(yt_cp-fmn)/fsd
                comp_xt_nm.append(st_nm[xt_cp])
                comp_yt_nm.append(yt_cp_nm)

                #print(st[xt_cp],yt_cp)
                #model_cp=model_cp.get_fantasy_model(st[xt_cp].reshape(1,3),yt_cp.reshape(1))
                model_cp = FRTOBJECT.build_rkhs_model(torch.stack(comp_xt_nm[1:]), torch.stack(comp_yt_nm), likelihood_1,t,ls_1)#building rkhs function
               
                
                #    model_cp.train()
                #    likelihood_1.train()
                #    model_cp=FRTOBJECT.func_train_model(model_cp,likelihood_1,comp_xt,comp_yt,training_iter) 
                model_cp.eval()
                #print(t)

                #compare model
                #f_st_pier=model_pier(st.reshape(n,3)).mean-beta*model_pier(st.reshape(n,3)).stddev
                f_st_pier_nm=model_pier(st_nm.reshape(n,3)).mean+beta*model_pier(st_nm.reshape(n,3)).stddev
                f_st_pier=f_st_pier_nm*fsd+fmn
                f_st_pier_nm_2=model_pier(st_nm.reshape(n,3)).mean-beta*model_pier(st_nm.reshape(n,3)).stddev
                f_st_pier_2=f_st_pier_nm_2*fsd+fmn
                b_pier=(f_st_pier-12*torch.ones(n))*(f_st_pier_2-12*torch.ones(n))
                b_pier[b_pier<0]=0
                ind_pier=(b_pier==0).nonzero()
                #print(f_st_pier)
                #print(fmax)
                in_comp=torch.min(f_st_pier,12*torch.ones(n))
                f_st_pier=(torch.pow(in_comp,3)*0.0579-torch.pow(f_st_pier,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)              
                #print(f_st_pier)
                #print(fmax)
                in_comp_2=torch.min(f_st_pier_2,12*torch.ones(n))
                f_st_pier_2=(torch.pow(in_comp_2,3)*0.0579-torch.pow(f_st_pier_2,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)              
                f_st_pier=torch.max(f_st_pier,f_st_pier_2)
                f_st_pier[ind_pier]=fmax
                f_st_pier=torch.max(fgrid[:,et])-f_st_pier
                #f_st_pier=fmax-fgrid[:,et]
                xt_pier=torch.argmin(f_st_pier.squeeze()+d_x[xt_pier,:].squeeze())
                yt_pier=vw[xt_pier,et]#+torch.randn(1)*torch.sqrt(noi_lm)
                pier_xt.append(st[xt_pier])
                pier_yt.append(yt_pier)
                in_comp=np.minimum(yt_pier,12)
                #print(avg_sc)
                ft_pier=(torch.pow(in_comp,3)*0.0579-torch.pow(yt_pier,2)*0.09)*(1/0.15)*60*rt*(1/v_r2)*(1/mind)
                pier_yt_cost.append(torch.max(fgrid[:,et])-ft_pier)
                pier_yt_power.append(ft_pier)
                yt_pier_nm=(yt_pier-fmn)/fsd
                pier_xt_nm.append(st_nm[xt_pier])
                pier_yt_nm.append(yt_pier_nm)
                #model_pier=model_pier.get_fantasy_model(st[xt_pier].reshape(1,3),yt_pier.reshape(1))
                model_pier = FRTOBJECT.build_rkhs_model(torch.stack(pier_xt_nm[1:]), torch.stack(pier_yt_nm), likelihood_2,t,ls_1)#building rkhs function
                #print(model_pier.covar_module)
               
                #    model_pier.train()
                #    likelihood_2.train()
                #    model_pier=FRTOBJECT.func_train_model(model_pier,likelihood_2,pier_xt,pier_yt,training_iter)
                model_pier.eval()

                #random model
                xt_rd=int(np.floor(np.random.uniform(0,1)*n))
                yt_rd=fgrid[xt_rd,et]#+torch.randn(1)*torch.sqrt(noi_lm)
                rand_xt.append(st[xt_rd])
                rand_yt.append(yt_rd)


                #service cost optimal model
                xt_so=torch.argmin(torch.max(fgrid[:,et])-fgrid[:,et])
                servopt_xt.append(st[xt_so])
                yt_so=fgrid[xt_so,et]#+torch.randn(1)*torch.sqrt(noi_lm)
                servopt_yt.append(yt_so)
                servopt_yt_cost.append(torch.max(fgrid[:,et])-yt_so)
                servopt_yt_power.append(yt_so)
                #model_so=model_so.get_fantasy_model(st[xt_so].reshape(1,2),yt_so.reshape(1))
                #model_so.eval()

                #stationary model
                xt_stat=a
                stat_xt.append(st[xt_stat])
                yt_stat=fgrid[xt_stat,et]#+torch.randn(1)*torch.sqrt(noi_lm)
                stat_yt.append(yt_stat)
                stat_yt_cost.append(torch.max(fgrid[:,et])-yt_stat)
                stat_yt_power.append(yt_stat)

                #print(f_st,"actual")
                #print(f_st_cp,"compare")
                #print(f_st_pier,"pier")
                #print(torch.max(fgrid[:,et])-fgrid[:,et],"right")#why did i print et-1
                
            comp_cost_serv=comp_cost_serv+torch.cumsum(torch.stack(comp_yt_cost),dim=0).squeeze()

            #comp_cost_serv=comp_cost_serv+torch.cumsum(torch.stack(comp_yt)[:,0],dim=0).squeeze()

            #act_cost_serv=act_cost_serv+torch.cumsum(torch.stack(train_y_gp)[:,0],dim=0).squeeze()

            act_cost_serv=act_cost_serv+torch.cumsum(torch.stack(train_y_gp_cost),dim=0).squeeze()

            #rand_cost_serv=rand_cost_serv+torch.cumsum(torch.stack(rand_yt)[:,0],dim=0).squeeze()

            rand_cost_serv=rand_cost_serv+torch.cumsum(fmax-torch.stack(rand_yt),dim=0).squeeze()

            #servopt_cost_serv=servopt_cost_serv+torch.cumsum(torch.stack(servopt_yt)[:,0],dim=0).squeeze()

            servopt_cost_serv=servopt_cost_serv+torch.cumsum(torch.stack(servopt_yt_cost),dim=0).squeeze()

            #stat_cost_serv=stat_cost_serv+torch.cumsum(torch.stack(stat_yt)[:,0],dim=0).squeeze()

            stat_cost_serv=stat_cost_serv+torch.cumsum(torch.stack(stat_yt_cost),dim=0).squeeze()

            pier_cost_serv=pier_cost_serv+torch.cumsum(torch.stack(pier_yt_cost),dim=0).squeeze()

            #pier_cost_serv=pier_cost_serv+torch.cumsum(torch.stack(pier_yt)[:,0],dim=0).squeeze()

            #kn_act_cost_serv=kn_act_cost_serv+torch.cumsum(torch.stack(kn_train_y_gp)[:,0],dim=0).squeeze()

            kn_act_cost_serv=kn_act_cost_serv+torch.cumsum(torch.stack(kn_train_y_cost),dim=0).squeeze()





            comp_power_serv=comp_power_serv+torch.cumsum(torch.stack(comp_yt_power),dim=0).squeeze()

            #comp_cost_serv=comp_cost_serv+torch.cumsum(torch.stack(comp_yt)[:,0],dim=0).squeeze()

            #act_cost_serv=act_cost_serv+torch.cumsum(torch.stack(train_y_gp)[:,0],dim=0).squeeze()

            act_power_serv=act_power_serv+torch.cumsum(torch.stack(train_y_gp_power),dim=0).squeeze()

            #rand_cost_serv=rand_cost_serv+torch.cumsum(torch.stack(rand_yt)[:,0],dim=0).squeeze()

            #rand_cost_serv=rand_cost_serv+torch.cumsum(fmax-torch.stack(rand_yt),dim=0).squeeze()

            #servopt_cost_serv=servopt_cost_sertorch.cumsum(torch.stack(servopt_yt)[:,0],dim=0).squeeze()

            servopt_power_serv=servopt_power_serv+torch.cumsum(torch.stack(servopt_yt_power),dim=0).squeeze()

            #stat_cost_serv=stat_cost_serv+torch.cumsum(torch.stack(stat_yt)[:,0],dim=0).squeeze()

            stat_power_serv=stat_power_serv+torch.cumsum(torch.stack(stat_yt_power),dim=0).squeeze()

            pier_power_serv=pier_power_serv+torch.cumsum(torch.stack(pier_yt_power),dim=0).squeeze()

            #pier_cost_serv=pier_cost_serv+torch.cumsum(torch.stack(pier_yt)[:,0],dim=0).squeeze()

            #kn_act_cost_serv=kn_act_cost_serv+torch.cumsum(torch.stack(kn_train_y_gp)[:,0],dim=0).squeeze()

            kn_act_power_serv=kn_act_power_serv+torch.cumsum(torch.stack(kn_train_y_gp_power),dim=0).squeeze()





            comp_cost_mvt=comp_cost_mvt+torch.cumsum(torch.abs(torch.stack(comp_xt)[1:T+1,0]-torch.stack(comp_xt)[0:T,0]),dim=0)*(1/mind)

            act_cost_mvt=act_cost_mvt+torch.cumsum(torch.abs(torch.stack(train_x_gp)[1:T+1,0]-torch.stack(train_x_gp)[0:T,0]),dim=0)*(1/mind)

            rand_cost_mvt=rand_cost_mvt+torch.cumsum(torch.abs(torch.stack(rand_xt)[1:T+1,0]-torch.stack(rand_xt)[0:T,0]),dim=0)*(1/mind)

            servopt_cost_mvt=servopt_cost_mvt+torch.cumsum(torch.abs(torch.stack(servopt_xt)[1:T+1,0]-torch.stack(servopt_xt)[0:T,0]),dim=0)*(1/mind)

            stat_cost_mvt=stat_cost_mvt+torch.cumsum(torch.abs(torch.stack(stat_xt)[1:T+1,0]-torch.stack(stat_xt)[0:T,0]),dim=0)*(1/mind)

            pier_cost_mvt=pier_cost_mvt+torch.cumsum(torch.abs(torch.stack(pier_xt)[1:T+1,0]-torch.stack(pier_xt)[0:T,0]),dim=0)*(1/mind)

            kn_act_cost_mvt=kn_act_cost_mvt+torch.cumsum(torch.abs(torch.stack(kn_train_x_gp)[1:T+1,0]-torch.stack(kn_train_x_gp)[0:T,0]),dim=0)*(1/mind)


        ret_set=[comp_cost_mvt,act_cost_mvt,rand_cost_mvt,servopt_cost_mvt,stat_cost_mvt, comp_cost_serv,act_cost_serv,rand_cost_serv,servopt_cost_serv, stat_cost_serv, pier_cost_mvt, pier_cost_serv, kn_act_cost_mvt, kn_act_cost_serv, comp_power_serv,act_power_serv,servopt_power_serv, stat_power_serv, pier_power_serv, kn_act_power_serv]
        for rit in range(len(ret_set)):
            #print(ret_it)
            ret_set[rit]=ret_set[rit]/(f_trep*(1+gam))
            #print(ret_it,gam)

        return ret_set




