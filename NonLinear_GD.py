#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:53:38 2018

@author: niharika-shimona
"""
import sys
import numpy as np
import scipy
import os
from numpy import linalg as LA
from sklearn import preprocessing

# torch
import torch
import pickle
import torch.multiprocessing as mp
from torch import optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from scipy import optimize
import scipy.io as sio
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt


#plt.ioff()

# dev
#from dev.utils import data_tools


def func(params, *args):
                
    D_n= args[0].numpy()
    lamb_n= args[1].numpy()
    y_n= args[2].numpy()
    B_upd= args[3].numpy()
    gamma= args[4]
    lambda_2 =args[5]
    model_upd =args[6]
    
    C_n= params
   
    m = np.shape(C_n)[0]
    C_n_copy = np.reshape(C_n,(m,1))
    C_n_T =torch.transpose(torch.from_numpy(C_n_copy),0,1)
    C_n_copy = np.array(C_n_copy)
    
    y_pred = model_upd(C_n_T.float()).detach().numpy()
    cons = D_n- np.matmul(B_upd,np.diagflat(C_n_copy))         
    
    error_C= gamma*LA.norm(y_n-y_pred)**2 + lambda_2 * LA.norm(C_n_copy,2)**2 
    + 0.5*LA.norm(D_n-np.matmul(B_upd,(np.diagflat(C_n_copy))),2)**2 + 0.5*np.trace(np.matmul(lamb_n.T,cons))
   
    return error_C
        
def get_input_optimizer(input_C):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS(input_C,lr=0.9)
    return optimizer


class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H,bias=False)
    self.linear2 = torch.nn.Linear(H, D_out,bias=False)

  def forward(self, x):
    """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary (differentiable) operations on Tensors.
    """
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


def err_compute_NL(corr,B,C,model,Y,D,lamb,gamma,lambda_1,lambda_2,lambda_3):
    
    "Computes the error at the current interation given the values of the iterates at the instant"

    fit_err = 0
    const_err =0 
    aug_lag_err =0
    err_W_reg =0

    for n in xrange(Y.size()[0]):
    
        "select out constraint and correlation terms"  
    
        Corr_n = torch.reshape(corr[n,:,:],(corr.size()[1],corr.size()[2]))
        lamb_n = torch.reshape(lamb[n,:,:],(lamb.size()[1],lamb.size()[2]))
        D_n = torch.reshape(D[n,:,:],(D.size()[1],D.size()[2]))
        
        B_T = torch.transpose(B,0,1)
        "compute correlation fit"
        X = Corr_n - torch.mm(D_n,B_T) 
        fit_err = fit_err + torch.norm(X,2)**2
    
        "compute lagrangian + reg. error"
        cons = D_n-torch.mm(B,torch.diagflat(C[:,n]))
        const_err = const_err+ torch.trace(torch.mm(torch.transpose(lamb_n,0,1),cons))
        aug_lag_err = aug_lag_err +0.5*torch.norm(cons,2)**2
    
        C_T =torch.transpose(C,0,1)
#        est_Y = C_T.mm(w1).clamp(min=0).mm(w2)
        est_Y = model(C_T)
        
#        err_W_reg = lambda_3 * (torch.norm(model.linear1.weight,2)**2 + torch.norm(model.linear2.weight,2)**2)
        err_W_reg= torch.zeros(1)
    "total error"
    err = fit_err + const_err + aug_lag_err + lambda_1*torch.norm(B,1) + gamma*torch.norm(est_Y-Y,2)**2 + lambda_2*torch.norm(C,2)**2+ lambda_3*err_W_reg
        
    return err.detach().numpy()

def alt_min_runner(corr,B_init,C_init,model_init,D_init,lamb_init,Y,gamma,lambda_1,lambda_2,lambda_3,lr1):
    
    "Outer Iteration module for alt. min."
    
    B_old = B_init
    C_old =C_init
    model_old=model_init
    D_old= D_init
    lamb_old =lamb_init
    num_iter_max =500
    thresh=1e-06
    err_out=[]

    
    for iter in xrange(num_iter_max):
        
        err_out.append(err_compute_NL(corr,B_old,C_old,model_old,Y,D_old,lamb_old,gamma,lambda_1,lambda_2,lambda_3))
        print ' At iteration' + `iter`+ '|| Error:' + `err_out[iter]` + '\n'
        
        
#        plt.hold(True)
        
        
        if (iter<20):
        
            [B,C,model,D,lamb] = alt_min_NL(corr,B_old,C_old,model_old,D_old,lamb_old,Y,gamma,lambda_1,lambda_2,lambda_3,lr1); 
        
        else:

            " scale the learning for the constraints "
            lr2 = 0.5*lr1
            [B,C,model,D,lamb] = alt_min_NL(corr,B_old,C_old,model_old,D_old,lamb_old,Y,gamma,lambda_1,lambda_2,lambda_3,lr2); 
        
        if(err_out[iter]> 10e10):
               
            print '\n This diverged, you lost,your life is a mess \n '
            B=B_old
            C=C_old
            D=D_old
            lamb =lamb_old
            model=model_old
            break
        
        " updates for alt. min. variables "
        B_old = B
        C_old = C
        D_old =D
        lamb_old =lamb
        model_old=model
        
        "exit conditions"
        if((iter>0) and( (abs((err_out[iter]-err_out[iter-1])) < thresh) or (err_out[iter]-err_out[iter-1]>10))):
           
            if(err_out[iter]>err_out[iter-1]):
               
                print '\n Exiting due to increase in function value, at iter' +`iter` + '\n'
          
            break
    fig,ax = plt.subplots()
    ax.plot(list(range(iter)),err_out[0:iter:1],'r')
    plt.title('Loss',fontsize=16)
    plt.ylabel('Error' ,fontsize=12)
    plt.xlabel('num of iterations',fontsize=12)
    plt.show()
    return B,C,model,D,lamb
    

def alt_min_NL(corr,B,C,model,D,lamb,Y,gamma,lambda_1,lambda_2,lambda_3,lr1):
   "Given the current values of the iterates, performs a single step of alternating minimisation"


   "B update"
   num_iter_max =100
    
   print 'Optimise B \n'
    
   "learning rate"
   t =0.0001;

   err_inner = []

   for iter in xrange(num_iter_max):
  
       DG = torch.zeros(B.size())
  
       for j in xrange(corr.size()[0]):
           
           Corr_j = torch.reshape(corr[j,:,:],(corr.size()[1],corr.size()[2]))
           D_j = torch.reshape(D[j,:,:],(D.size()[1],D.size()[2]))
           lamb_j = torch.reshape(lamb[j,:,:],(lamb.size()[1],lamb.size()[2]))
           D_j_T = torch.transpose(D_j,0,1)
           
           T1 = 2*(torch.mm((torch.mm(B,D_j_T)-Corr_j),D_j))
           T2 = torch.mm(D_j,torch.diagflat(C[:,j]))
           T3 = torch.mm(torch.mm(B,torch.diagflat(C[:,j])),torch.diagflat(C[:,j]))
           T4 = torch.mm(lamb_j,torch.diagflat(C[:,j]))
           DG = DG + T1 - T2 + T3 - T4 
           
       if(iter ==0):
        
          DG_init = DG;

       "Iterative shrinkage thresholding"
       X_mat = B - t*DG/lambda_1
       Mult = (torch.max(torch.abs(X_mat)-t*torch.ones(X_mat.size()),torch.zeros(X_mat.size())))
       B = torch.sign(X_mat)*Mult

 
       err_inner.append(err_compute_NL(corr,B,C,model,Y,D,lamb,gamma,lambda_1,lambda_2,lambda_3))
       print 'At B iteration ' + `iter` + '|| Error:' + `err_inner[iter]` + '\n'   
#%   plot(1:iter,err_inner,'b');
#%   hold on;
#%   drawnow;
#  
       if ((iter>1) and (((torch.norm(DG,2)/torch.norm(DG_init,2))< 10e-06) or (err_inner[iter]>err_inner[iter-1]))):
      
          break
   B_upd = B

   print ' At final B iteration || Error: ' + `err_compute_NL(corr,B_upd,C,model,Y,D,lamb,gamma,lambda_1,lambda_2,lambda_3)`  
     
# C update

   print 'Optimise C and weights \n'
# update coefficients
#% quadratic prog solver: x = quadprog(H,f,A,b)
   
   [C_upd,model_upd] = Coefficient_Updates_NL(corr,B_upd,C,Y,D,lamb,model,gamma,lambda_2,lambda_3)

   "Constraint variable updates "
   
   [D_upd,lamb_upd] =Proximal_Updates_NL(corr,lamb,B_upd,C_upd,lr1)
  
   print ' Step D and lambda || Error: ' + `err_compute_NL(corr,B_upd,C_upd,model_upd,Y,D_upd,lamb_upd,gamma,lambda_1,lambda_2,lambda_3)`+'\n'

   return B_upd,C_upd,model_upd,D_upd,lamb_upd

def Proximal_Updates_NL(corr,lamb,B_upd,C_upd,lr):
    
    num_iter_max = 100
    
    lamb_upd = torch.zeros(lamb.size())
    D_upd = torch.zeros((corr.size()[0],B_upd.size()[0],C_upd.size()[0]))
    
    
    for k in xrange(lamb.size()[0]):
      
        Corr_k = torch.reshape(corr[k,:,:],(corr.size()[1],corr.size()[2]))
        lamb_k = torch.reshape(lamb[k,:,:],(lamb.size()[1],lamb.size()[2]))
     
        for c in xrange(num_iter_max):
               
            T1= (torch.mm(B_upd,torch.diagflat(C_upd[:,k]))+ 2*torch.mm(Corr_k,B_upd) - lamb_k)
            T2=  torch.inverse(torch.eye(B_upd.size()[1])+2*(torch.mm(torch.transpose(B_upd,0,1),B_upd)))
            D_k = torch.mm(T1,T2)
       #gradient ascent for the kth lagrangian
            lamb_k = lamb_k + (0.5**(c-1))*lr*(D_k - torch.mm(B_upd,torch.diagflat(C_upd[:,k])))
        
       #initial gradient norm
            if (c ==0):
                grad_norm_init = torch.norm(D_k - B_upd.mm(torch.diagflat(C_upd[:,k])),2)
                
            conv = torch.norm(D_k - torch.mm(B_upd,torch.diagflat(C_upd[:,k])),2)
            
            if (conv/grad_norm_init<10e-06):
                break;
     
        lamb_upd[k,:,:]= lamb_k
        D_upd[k,:,:] =D_k
    
    return D_upd,lamb_upd

    
def Coefficient_Updates_NL(corr,B_upd,C,Y,D,lamb,model,gamma,lambda_2,lambda_3):
    
    "Neural Network Updates for behavioural data mapping"  
#    w1.requires_grad_(True)
#    w2.requires_grad_(True)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4,momentum=0.9,weight_decay=lambda_3)
    
    
    # W update
    for t in range(500):
        # Forward pass: compute predicted y using operations on Tensors. Since w1 and
        # w2 have requires_grad=True, operations involving these Tensors will cause
        # PyTorch to build a computational graph, allowing automatic computation of
        # gradients. Since we are no longer implementing the backward pass by hand we
        # don't need to keep references to intermediate values.
        C_T= torch.transpose(C,0,1)
        y_pred = model(C_T)
        
#        y_pred = C_T.mm(w1).clamp(min=0).mm(w2)
#        w1 = model.linear1.weight
#        w2 = model.linear2.weight
#        w1_T = torch.transpose(w1,0,1)
#        w2_T = torch.transpose(w2,0,1)
#        loss = gamma* (y_pred - Y).pow(2).sum() + lambda_3 * torch.trace((w1_T.mm(w1) + w2_T.mm(w2)))
#        loss = gamma*loss_fn(y_pred, Y) + lambda_3 * torch.trace(w1_T.mm(w1)) + lambda_3*torch.trace(w2_T.mm(w2))
        loss = gamma*loss_fn(y_pred, Y) 
#        print(t, loss.item())

  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Tensors with requires_grad=True.
  # After this call w1.grad and w2.grad will be Tensors holding the gradient
  # of the loss with respect to w1 and w2 respectively.
#       loss.backward()
        
  # Update weights using gradient descent. For this step we just want to mutate
  # the values of w1 and w2 in-place; we don't want to build up a computational
  # graph for the update steps, so we use the torch.no_grad() context manager
  # to prevent PyTorch from building a computational graph for the updates
#        with torch.no_grad():
#          w1 -= learning_rate * w1.grad
#          w2 -= learning_rate * w2.grad
  
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    # Manually zero the gradients after running the backward pass
#          w1.grad.zero_()
#          w2.grad.zero_()
#    
#    w1.detach()
#    w2.detach()
#          
#    w1_upd =w1
#    w2_upd=w2      
    del optimizer
    model_upd= model 
    
    print ' Step W || Error: ' + `err_compute_NL(corr,B_upd,C,model_upd,Y,D,lamb,gamma,lambda_1,lambda_2,lambda_3)`+'\n'

    C_upd = Optimise_Coefficients(C,model_upd,D,B_upd,Y,lamb,gamma,lambda_2)
    
    print ' Step C || Error: ' + `err_compute_NL(corr,B_upd,C_upd,model_upd,Y,D,lamb,gamma,lambda_1,lambda_2,lambda_3)`+'\n'

    return C_upd,model_upd

def Optimise_Coefficients(C,model_upd,D,B_upd,Y,lamb,gamma,lambda_2):      
    # C update      
#    learning_rate =1e-3
    
    for n in xrange(C.size()[1]):
        
#        C_n = Variable(torch.reshape(C[:,n].clone(),(C.size()[0],1)),requires_grad = True)
        C_n_init = torch.reshape(C[:,n],(C.size()[0],1)).clone().numpy()
        lamb_n =lamb[n,:,:]
        D_n= D[n,:,:]
        y_n= Y[n]
        mybounds = [(0,None)]*np.shape(C_n_init)[0]
        
        C_n_upd = scipy.optimize.fmin_l_bfgs_b(func,x0=C_n_init,args=(D_n,lamb_n,y_n,B_upd,gamma,lambda_2,model_upd), bounds=mybounds, approx_grad=True)

#        optimizer_C =get_input_optimizer([C_n])
#        
#        for t in range(200):
#            
##            y_n_pred = torch.transpose(C_n,0,1).mm(w1).clamp(min=0).mm(w2)
#            
#            
#            optimizer_C.zero_grad()
#            
#            def closure():
#                optimizer_C.zero_grad()
#                y_n_pred = model_upd(torch.transpose(C_n,0,1)) 
#                cons = D_n-torch.mm(B_upd,torch.diagflat(C_n[:]))            
#           
#                loss_C = gamma*torch.norm(y_n-y_n_pred)**2 + lambda_2 * torch.norm(C_n,2)**2 + 0.5*torch.norm(D_n-B_upd.mm(torch.diagflat(C_n)),2)**2 
#                + 0.5*torch.trace(torch.transpose(lamb_n,0,1).mm(cons))
#           
#                loss_C.backward()
#                return loss_C
#            
#            
#            
#            optimizer_C.step(closure)
#            grad_C_n = C_n.grad
#            
#            if(torch.norm(grad_C_n)<1e-04):
#                
#                break
#            
##            with torch.no_grad():
##                C_n -= learning_rate * grad_C_n
#            
##            C_n.grad.zero()
        
        
        if (n==0):
            m = np.shape(C_n_upd[0])[0]
            C_n_upd = np.reshape(np.asarray(C_n_upd[0]),(m,1))
            
            C_upd =torch.from_numpy(C_n_upd)
        else:
            m = np.shape(C_n_upd[0])[0]
            C_n_upd = np.reshape(np.asarray(C_n_upd[0]),(m,1))
            C_upd= torch.cat((C_upd,torch.from_numpy(C_n_upd)),1)
      
        print 'Patient ' + `n` + ' Optimised \n'
    
    return C_upd.float()


if __name__ == '__main__':         

    data = sio.loadmat('/home/niharika-shimona/Documents/Projects/Autism_Network/Sparse-Connectivity-Patterns-fMRI/Simulated_Data/Simulated_Data_set_polybias_1.mat')

 
gamma=float(sys.argv[1])
lambda_1=float(sys.argv[2])
lambda_2=float(sys.argv[3])
lambda_3=float(sys.argv[4])
H1 = int(sys.argv[5])
lr1=1e-04

dir_name = '/home/niharika-shimona/Documents/Projects/Autism_Network/Sparse-Connectivity-Patterns-fMRI/Neural_Network_Prototype/'+ 'workspace_out_shl_sparsity_' + `lambda_1` + '_reg_C_' + `lambda_2` + '_reg_W_'+ `lambda_3` + '_trad_' + `gamma` + '_H1_'+ `H1` +'/'
if not os.path.exists(dir_name):
 	os.makedirs(dir_name)

log_filename = dir_name +'logfile.txt'

log = open(log_filename, 'w')
sys.stdout = log

B = torch.from_numpy(data['B']).float()
C = torch.from_numpy(data['C']).float()
W = torch.from_numpy(data['W']).float()
corr = torch.from_numpy(data['corr']).float()
Y = torch.from_numpy(data['Y']).float()

device = torch.device('cpu')
#initialisation
#B_init = torch.randn(B.size()).float()
B_init =B + 0.0001* torch.randn(B.size()).float()
C_init = torch.randn(C.size()).float()
D_init = torch.zeros((C.size()[1],B.size()[0],B.size()[1])).float()
lamb_init = torch.zeros(C.size()[1],B.size()[0],B.size()[1]).float()

for n in xrange(C.size()[1]):
    D_init[n][:][:] = B_init.mm(torch.diagflat(C_init[:,n])) 


D_init = D_init.float()

N, D_in, H1, H2, D_out = 58, C.size()[0],H1,4, 1
#w1_init = torch.randn(D_in, H, device=device)
#w2_init = torch.randn(H, D_out, device=device)
#
#w1_init =w1_init.float()   
#w2_init =w2_init.float()
print 'Optimising with Parameters: '+ '\n' + ' sparsity: ' + `lambda_1` + '| reg_C: ' + `lambda_2` + '| reg_W: '+ `lambda_3` + '| trad: ' + `gamma`

#model_init = torch.nn.Sequential(
#          torch.nn.Linear(D_in, H1, bias=False),
#         torch.nn.Sigmoid(),
#          torch.nn.Linear(H1, H2, bias=False),
#          torch.nn.Sigmoid(),
#          torch.nn.Linear(H2, D_out, bias=False),
#        ).to(device)

model_init = torch.nn.Sequential(torch.nn.Linear(D_in, H1, bias= False),torch.nn.Sigmoid(),torch.nn.Linear(H1,D_out, bias= False)).to(device)

[B_gd,C_gd,model_gd,D_gd,lamb_gd] = alt_min_runner(corr,B_init,C_init,model_init,D_init,lamb_init,Y,gamma,lambda_1,lambda_2,lambda_3,lr1)
B_norm = preprocessing.normalize(B,'l2',axis=0)
B_gd_norm = preprocessing.normalize(B_gd,'l2',axis=0)

print np.matmul(B_norm.T,B_gd_norm) 

"Save Outputs"
dict_save ={'B_gd':B_gd,'C_gd':'C_gd','model_gd':model_gd,'D_gd':D_gd, 'lamb_gd':lamb_gd, 
            'B':B, 'C':C, 'W':W, 
            'corr':corr,
            'Y':Y}

filename = dir_name + 'Output.p'
pickle.dump(dict_save, open(filename, "wb"))

font = {'family' : 'normal',
         'size'   : 12}
matplotlib.rc('font', **font)

fig,ax =plt.subplots()
ax= plt.imshow(B_gd_norm, cmap=plt.cm.jet,aspect='auto')
plt.title('Recovered Networks')
plt.show()
figname = dir_name +'B_gd.png'
fig.savefig(figname)   # save the figure to fil
plt.close(fig)

fig1,ax1 =plt.subplots()
ax1= plt.imshow(B_norm, cmap=plt.cm.jet,aspect='auto')
plt.title('Generating Networks')
plt.show()
figname1 = dir_name +'B.png'
fig1.savefig(figname1)   # save the figure to fil
plt.close(fig1)

fig2,ax2 =plt.subplots()
Y_pred = model_gd(torch.transpose(C_gd,0,1)).detach().numpy()
ax2.scatter(Y.detach().numpy(),Y_pred,color = 'r')
ax2.plot([Y_pred.min()-2,Y_pred.max()+2], [Y_pred.min()-2, Y_pred.max()+2], 'k--', lw=4)
ax2.set_xlabel('Measured',fontsize =12)
ax2.set_ylabel('Predicted',fontsize =12)
plt.title('Performance')
plt.show()
figname2 = dir_name +'Performance.png'
fig2.savefig(figname2)   # save the figure to fil
plt.close(fig2)
