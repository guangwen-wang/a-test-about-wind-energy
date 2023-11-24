from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import cos, nonzero, random
from torch import serialization


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

import time
import os


def find_zero(f, x0, x1, eps=1e-6, max_iter=100):
    x = (x0 + x1) / 2
    s = (x1 - x0) / 4
    for i in range(max_iter):
        y = f(x)
        if abs(y) < eps:
            return x
        x -= s * np.sign(y)
        s /= 2
    return x

# Bregman projection onto simplex using the weighted, shifted entropy
# alphas = dual variables for the constraints y_i >= 0
# and the projection is y

# x = point of the simplex
# v = vector of costs
# delta = vector of shifts
# w = vector of weights

# entropy functional is Phi(x) = \sum_i w_i (x_i + delta_i)

def bp(x, v, delta, w, max_iter=100, err=1e-6):

    n = len(x)
    alpha = np.zeros(n)

    target = 1 + np.sum(delta)
    x_shift = x + delta

    for i in range(max_iter):

        z = (alpha-v)/w

        f = lambda L : np.sum(x_shift*np.exp(L/w + z))-target
        L = find_zero(f, 0, np.max(w+v), err, max_iter)

        y = x_shift*np.exp((L+alpha-v)/w) - delta

        # take a gradient descent step toward the correct alphas
        alpha_prime = np.maximum(0, v - L + w*np.log((y+delta)/x_shift))#Prof. Lee contained only delta instead of y+delta
        if (np.linalg.norm(alpha-alpha_prime, ord=2) < err):
            break
        alpha = alpha_prime

    y = np.maximum(y,0)
    y = y/np.sum(y)
    return y




















class Tree:
    def __init__(self, cargo, level,par=None, weight=None, leftmc=None, cost=0, rightmc=None, rightsi=None,leftsi=None,id=None,leaves=0,rprob=0,cprob=0,rprob_old=0,cprob_old=0,tta=None, eta=None, delt=None,weight_children=None):
        self.cargo = cargo#store the cluster center of the node(if possible)
        self.weight = weight#weight of the edge above it
        self.level = level#level of the cluster center from root
        self.parent = par#parent
        self.leftmostchild  = leftmc
        self.rightmostchild = rightmc
        self.rightsibling = rightsi
        self.leftsibling = leftsi
        self.id=id#unique id 
        self.leaves=leaves#no of leaves in the subtree rooted at this node
        self.rprob=rprob#probability of the state being in the leaves of the subtree rooted at this node(corresponding to x)
        self.cprob=cprob#conditional probability(corresponding to q)
        self.cost=cost#cumulative cost defined based on the cost on the children
        self.rprob_old=rprob_old
        self.cprob_old=cprob_old
        self.eta=eta
        self.tta=tta
        self.delt=delt
        self.weight_children=weight_children
    def __str__(self):
        return str(self.cargo)
    
    def ocsampler(root,ls_pt,wts):
      croot=root
      child=root.leftmostchild
      children_list=[child]
      alpha=[child.rprob_old]
      beta=[child.rprob]
      ids=[child.id]
      #cbeta=[child.cprob]
      while(child.rightsibling!=None):
        child=child.rightsibling
        alpha.append(child.rprob_old)
        beta.append(child.rprob)
        ids.append(child.id)
        #cbeta.append(child.cprob)
        children_list.append(child)
      #print(alpha)
      #print(beta)
      alpha=np.asarray(alpha)
      beta=np.asarray(beta)
      #print(alpha,beta,root.rprob,cbeta)
      alphabef=alpha
      alpha=alpha/np.sum(alpha)
      betabef=beta
      beta=beta/np.sum(beta)
      d_l=(np.ones((len(alpha),len(alpha)))-np.eye(len(alpha)))*(2*wts[0])
      #print(ls_pt)
      #print(d_l)
      cxt=int(np.argwhere(ids==ls_pt[0]))
      sam_dist=ot.emd(alpha,beta,d_l)
      sam_dist=sam_dist[cxt,:].squeeze()
      if np.sum(sam_dist) == 0:
        print(np.sum(ot.emd(alpha,beta,d_l)))
        print(alpha,beta,betabef,alphabef)
        print(ids)
        print(ot.emd(alpha,beta,d_l))
        print(cxt,ids[cxt],ls_pt[0])
        print(root.rprob,root.rprob_old)
        return ids[cxt]
        #print(np.sum(ot.emd(alpha,beta,d_l)))
        #print("something else")
      #print(sam_dist)
      sam_dist=sam_dist/(np.sum(sam_dist))
      #print(sam_dist)
      if len(alpha)==1:
        x_l=0
      else:
        x_l=int(np.random.choice(sam_dist.size,1,p=sam_dist))
      #print(x_l,cxt,"sampled by oc")
      if x_l!=cxt :
        #create a matrix internal vertices*leaves marking one if it is a leaf
        return ids[x_l]
      else:
        if children_list[x_l].leftmostchild == None:
          return ids[x_l]
        #print(ids[x_l],ls_pt[0],"path",alphabef,betabef)
        return Tree.ocsampler(children_list[x_l],ls_pt[1:],wts[1:])#node corresponding to cxt, next weight, next ls_pt.id 
    def update_constants_postorder(tree):#the mirror descent procedure to update conditional probabilities
      if tree == None: return
      if tree.leftmostchild == None: return
      Tree.update_constants_postorder(tree.leftmostchild)#inorder traversal
      lc=[]#no of leaves under each children
      w=[]
      node=tree.leftmostchild
      while(node!=None):
        lc.append(node.leaves)
        w.append(node.weight)
        node=node.rightsibling
      #print(tree.leaves)
      tree.tta=np.asarray(lc)/tree.leaves#constants
      tree.eta=1+np.log2(1/tree.tta)#constants
      tree.delt=tree.tta/tree.eta#constants from the paper
      tree.weight_children=np.asarray(w)/tree.eta
      Tree.update_constants_postorder(tree.rightsibling)#traversal towards the right sibling
      return 



    def update_cprob_postorder(tree):#the mirror descent procedure to update conditional probabilities
      if tree == None: return
      if tree.leftmostchild == None: return
      Tree.update_cprob_postorder(tree.leftmostchild)#inorder traversal
      a=[]#list of cprobs of the children 
      cv=[]#costs of the children
      #lc=[]#no of leaves under each children
      #w=[]#weights of the edges between the node and its children
      node=tree.leftmostchild
      while(node!=None):
        a.append(node.cprob)
        cv.append(node.cost)
        #lc.append(node.leaves)
        #w.append(node.weight)
        node=node.rightsibling
      #print(tree.leaves)
      delt=tree.delt
      w_v=tree.weight_children
      a=np.asarray(a)
      new_prob=bp(a,np.asarray(cv),delt,w_v)#mirror descent
      new_prob=new_prob/sum(new_prob)
      node=tree.leftmostchild
      for i in range(len(new_prob)):
        #node.cprob_old=node.cprob
        node.cprob=new_prob[i]#store the probs
        node=node.rightsibling
      if sum(new_prob)-1 >= pow(10,-4):
        print('wtf',new_prob)
      tree.cost=np.sum(new_prob*np.asarray(cv))#weighted cost
      Tree.update_cprob_postorder(tree.rightsibling)#traversal towards the right sibling
      return 
    def cprobtorprob_preorder(tree):#calculating actual probability from conditional probability(q->x)
      if tree == None: return 
      if tree.parent!=None:
        nodeu=tree.parent
        tree.rprob_old=tree.rprob
        tree.rprob=(tree.cprob*nodeu.rprob)#start from the root
        #print(tree.cprob,nodeu.rprob)
      Tree.cprobtorprob_preorder(tree.leftmostchild)#preorder
      Tree.cprobtorprob_preorder(tree.rightsibling)
      return tree
    def clonetree_preorder(tree,treedict,parent=None):
        if tree == None: return
        newnode=Tree(cargo=tree.cargo,weight=tree.weight,rprob=tree.rprob,level=tree.level,par=parent,id=tree.id, leaves=tree.leaves, eta=tree.eta, tta=tree.tta, delt=tree.delt, weight_children=tree.weight_children)
        treedict[tree]=newnode
        #if newnode.rprob==1:
           # print("1")
        newnode.leftmostchild=Tree.clonetree_preorder(tree.leftmostchild,treedict,newnode)
        newnode.rightsibling=Tree.clonetree_preorder(tree.rightsibling,treedict,parent) 
        return newnode
    
    
    def rprobtocprob_postorder(tree):#calculating conditional probability from actual probability(x->q)not unique
      if tree == None: return 
      Tree.rprobtocprob_postorder(tree.leftmostchild)#postorder
      if tree.parent!=None:
        if tree.parent.rprob!=0:
          tree.cprob=(tree.rprob/tree.parent.rprob)
        else:#probability of the parent being 0, we give equal probability towards all the siblings
          nodeu=tree.parent.leftmostchild
          t=1
          while(nodeu.rightsibling!=None):
            t=t+1
            nodeu=nodeu.rightsibling
          tree.cprob=(1/t)
      else:
        tree.cprob=tree.rprob
      Tree.rprobtocprob_postorder(tree.rightsibling)
      return 
      
    def define_id_postorder(tree,id):#unique id definition postorder
      id=id
      if tree == None: return id
      tree.id=Tree.define_id_postorder(tree.leftmostchild,id)
      cid=Tree.define_id_postorder(tree.rightsibling,tree.id+1)
      return cid
    def define_weight_postorder(tree,weight):#level wise weights postorder initialized with the minimum weight
      weight=weight
      if tree == None: return 1
      tree.weight=weight*Tree.define_weight_postorder(tree.leftmostchild,weight)#recursively multiplied
      cweight=Tree.define_weight_postorder(tree.rightsibling,weight)
      return tree.weight
    def cal_wasserstein_dist(tree):#calculate wasserstein distance(nodes initialized with difference of the 2 probs)
      if tree == None: return 0
      wdist=np.absolute(tree.rprob)*tree.weight+Tree.cal_wasserstein_dist(tree.leftmostchild)
      return wdist+ Tree.cal_wasserstein_dist(tree.rightsibling)
    def print_leaves_postorder(tree):
      if tree == None: return
      Tree.print_leaves_postorder(tree.leftmostchild)
      print(tree.leaves,tree.id)
      Tree.print_leaves_postorder(tree.rightsibling)
      return
    def count_leaves(tree):#count leaves under a node--post order
      if tree == None: return 0
      if tree.leftmostchild == None:
        return 1+Tree.count_leaves(tree.rightsibling) 
      tree.leaves=Tree.count_leaves(tree.leftmostchild)
      #t=t+1
      return tree.leaves+Tree.count_leaves(tree.rightsibling)
    def rprob_calc_nodes(tree):#calculation of the actual probs based on the children's probs(leaf probs pre initialized)
      if tree == None: return 0
      if tree.leftmostchild == None:
        return tree.rprob+ Tree.rprob_calc_nodes(tree.rightsibling)
      tree.rprob = Tree.rprob_calc_nodes(tree.leftmostchild)
      return tree.rprob+Tree.rprob_calc_nodes(tree.rightsibling)
    def prune(croot):#prune nodes having more than half the leaves of the parent(due to change of tau from 2 to 7)
      if croot == None: return
      if croot.leftmostchild == None: return 
      Tree.prune(croot.leftmostchild)#post order
      proot=croot.parent
      if proot == None: return
      if croot.leaves>(proot.leaves/2):#condition check
        grandchild=croot.leftmostchild
        grandchild.parent=proot
        while grandchild.rightsibling!=None:#reassign the parent of the grandchildren as proot
          grandchild=grandchild.rightsibling
          grandchild.parent=proot
        if croot.rightsibling!=None:
          grandchild.rightsibling=croot.rightsibling
        else:
          proot.rightmostchild=grandchild#have to think of more cases#have to declare of the parents
        if croot.id!=proot.leftmostchild.id:
          eroot=proot.leftmostchild
          while eroot.rightsibling.id!=croot.id:#find the left sibling of croot
            eroot=eroot.rightsibling
          eroot.rightsibling=croot.leftmostchild
        else:
          proot.leftmostchild=croot.leftmostchild
      Tree.prune(croot.rightsibling)
      return 
    def main(x_a,n,cseed):
        np.random.seed((os.getpid()*cseed*int(time.time()))%123456789)
        d_x=distance_matrix(x_a.reshape(n,1),x_a.reshape(n,1))#find distance between all pairs of states
        td_x=d_x+100000*np.identity(n)#add a large value to all the diagonal values 
        mind=np.min(td_x)#find the minimum distance between 2 states different from each other
        d_x=(1/mind)*d_x#scale the distance matrix with inverse of the minimum distance so that the new minimum distance becomes 1 
        diam=np.max(d_x)#maximum distance
        Delta=np.power(2,np.ceil(np.log2(diam)))#find the nearest power of 2 to the maximum distance
        
        pi=np.random.permutation(np.arange(0,n))#random permutation
        per={pi[x]:x for x in range(0,n)}#store the permutation in a dictionary
        r0=1+np.random.uniform(0,1)#random number between 1 and 2
        ld=int(np.log2(Delta)+1)#number of levels based on the maximum distance
        r=r0*2 ** np.arange(0,ld+1)
        r=r/2#edge weights for each level---last value would be the Delta
        vld=np.ones((n,ld+1,ld+1))*(-1)#matrix to store the cluster centers at each level
        for i in range(0,ld+1):
            for v in range(0,n):
                vld[v,i,i]=per[np.min(pi[np.where(d_x[v]<r[ld-i])])]#find all the indices whose states are closer than r[ld-i], find their corresponding permutations and take the minimum among those as the cluster center
                if i<ld:
                    vld[v,i+1]=vld[v,i]#store the cluster centres iteratively

        u, indices = np.unique(vld[:,0], axis=0, return_index=True)#the unique cluster centers in each level
        root=Tree(u[0],0,id=0)#root node with id-0,cluster center u[0] and level-0
        y=1
        p={tuple(vld[indices[j],0]) : root for j in range(0,len(u))}#dictionary with cluster center vs tree node
        pld=(-1)*np.ones((n,ld+1))
        for i in range(1,ld+1):
            u, indices, rind = np.unique(vld[:,i], axis=0, return_index=True, return_inverse=True)#find the cluster centers in each level
            #print(vld[:,ld-i])
            #print(u[rind])
            a=[]#store the nodes in a list for assigning as parent in the next level
            for j in range(0,len(u)):
                if p[tuple(vld[indices[j],i-1])].leftmostchild==None:#find the previous level cluster center assgined to this point 
                    p[tuple(vld[indices[j],i-1])].leftmostchild=Tree(u[j],i,par=p[tuple(vld[indices[j],i-1])],id=y)
                    p[tuple(vld[indices[j],i-1])].rightmostchild=p[tuple(vld[indices[j],i-1])].leftmostchild
                    a.append(p[tuple(vld[indices[j],i-1])].leftmostchild) 
                else:
                    p[tuple(vld[indices[j],i-1])].rightmostchild.rightsibling=Tree(u[j],i,par=p[tuple(vld[indices[j],i-1])],id=y)
                    p[tuple(vld[indices[j],i-1])].rightmostchild=p[tuple(vld[indices[j],i-1])].rightmostchild.rightsibling
                    a.append(p[tuple(vld[indices[j],i-1])].rightmostchild)
                pld[np.argwhere(rind==j),i]=p[tuple(vld[indices[j],i-1])].id#store the id of the parent of the current cluster center in all the locations of the current cluster center
                y=y+1#id increase
            #print(np.argwhere(rind==j),j,i-1)
            #print(a[j].cargo,a[j].parent.cargo)

            p={tuple(vld[indices[j],i]) : a[j] for j in range(0,len(u))}#dictionary for nodes in the current level with key as their location
        # 
        r=r[1:]#last value is edge of none
        r7=np.power(7,np.ceil(np.log(r)/np.log(7)))
        r7f=np.flip(r7)#conversion of the edges to base 7 and flipped
        # 
        #modification of the tree based on Prof. Lee's lemma
        #prune levels based on the no of unique edge levels
        u,indices,rind=np.unique(r7f,return_index=True,return_inverse=True)#determine the new no of levels based on the no of unique edge levels
        rootv=p[tuple(vld[0,ld])]#leaf
        while(rootv.parent!=None):
            rootv=rootv.parent#finding root
        #print(rootv.level)
        acroot=rootv#store the root for later use
        cn=[]
        cn.append(rootv)
        #print(rind)
        for i in range(1,len(rind)):
            an=[]
            if rind[i-1]==rind[i]:#rind[0]--edges between root and first level of children---check the edge weight is same as before
                for croot in cn:#prune the level
                    #print("k")
                    #print("yes",i,croot.level)
                    childl=croot.leftmostchild#will always exist as we dont store at the last level
                    #print(childl.level)
                    childr=croot.rightmostchild
                    croot.leftmostchild=childl.leftmostchild
                    croot.rightmostchild=childr.rightmostchild
                    while (childl!=None):
                        grandchild=childl.leftmostchild
                        #print(grandchild.level)
                        grandchild.parent=croot#change the parent
                        while (grandchild.rightsibling!=None):
                            grandchild=grandchild.rightsibling
                            grandchild.parent=croot
                        if (childl.rightsibling!=None):
                            grandchild.rightsibling=childl.rightsibling.leftmostchild#change the right sibling at the grand children level while changing from one child to another
                        childl=childl.rightsibling
                    an.append(croot)#stay at the level of nodes as the next layer is pruned
            else:#if the edges are different, move to the next level and store the nodes
                for croot in cn:
                    child=croot.leftmostchild
                    while (child!=None):
                        an.append(child)
                        child=child.rightsibling
            cn=an
        #prune based on leaves count of a node and its parent
        for u in range(0,n):
            un=p[tuple(vld[u,ld])]
            un.leaves=1#initialize leaves count on leaves
        rootv=acroot#stored root
        rootv.leaves=Tree.count_leaves(rootv)#define the leaves count over all the nodes
        rootv.leaves=Tree.prune(rootv)#prune
        #add nodes to equalize all root-leaf path length
        maxl=1#find max length
        dfl=np.zeros((n))
        for u in range(0,n):
            rootv=p[tuple(vld[u,ld])]
            cl=1
            while (rootv.parent!=None):
                rootv=rootv.parent
                cl=cl+1
            dfl[u]=cl#store the length of each root-leaf path
            #print(rootv.id)
            if maxl<cl:
                maxl=cl
        for u in range(0,n):
            if dfl[u]<maxl:#for all paths less than the maximum
                cl=dfl[u]
                nodeu=p[tuple(vld[u,ld])]
                while(cl!=maxl):#add maxl-dfl[u] nodes between leaf and its parent 
                    nodeu.leftmostchild=Tree(nodeu.cargo,cl+1,par=nodeu)
                    nodeu.rightmostchild=nodeu.leftmostchild
                    nodeu=nodeu.leftmostchild
                    cl=cl+1
                p[tuple(vld[u,ld])]=nodeu#the new leaf node
        rootv=acroot 
        rootv.id=Tree.define_id_postorder(rootv,0)#define ids  

        id_matrix=np.zeros((rootv.id+1,rootv.id+1))#the ids of the leaves corresponding to an internal node
        for u in range(0,n):
            nodeu=p[tuple(vld[u,ld])]
            lf_id=nodeu.id
            id_matrix[lf_id,lf_id]=1
            while (nodeu.parent!=None):
                nodeu=nodeu.parent
                id_matrix[nodeu.id,lf_id]=1
        #id_lf={p[tuple(vld[u,ld])].id : u for u in range(n)}

        u7f=np.unique(r7f)#returns in ascending order
        u7f=u7f[len(u7f)-maxl+1:len(u7f)]#ignore the initial values which are outside the root-leaf length
        cu7f=np.cumsum(u7f)#cumulative sum

        #distance calculation
        pid=(-1)*np.ones((n,maxl))#matrix to store the ids corresponding to each root-leaf path
        for u in range(0,n):
            rootv=p[tuple(vld[u,ld])]
            pid[u,maxl-1]=rootv.id
            cl=maxl-2
            while (rootv.parent!=None):
                rootv=rootv.parent
                pid[u,cl]=rootv.id
                cl=cl-1
        for u in range(0,n):#initialize leaves count in the new leaves
            un=p[tuple(vld[u,ld])]
            un.leaves=1
        rootv=acroot
        rootv.leaves=Tree.count_leaves(rootv)
        #unique edges
        #u7f=np.unique(r7f)#returns in ascending order
        #u7f=u7f[len(u7f)-maxl+1:len(u7f)]#ignore the initial values which are outside the root-leaf length
        #cu7f=np.cumsum(u7f)#cumulative sum
        #distance matrix
        d_g=np.zeros((n,n))
        for v in range(0,n):
            d_g[v,:]=2*cu7f[maxl-2-((pid-pid[v,:])!=0).argmax(axis=1)]#from pid matrix find the level at which the root leaf paths of v and every other leaf converges
            d_g[v,v]=0
        dfds=np.sum(np.abs(d_x-d_g))
        eps=np.amin(d_g+np.amax(d_g)*np.identity(n))
        return [eps,acroot,p,vld,ld,u7f,id_matrix,dfds]






