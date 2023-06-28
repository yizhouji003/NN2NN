import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
import math
import torch.linalg as linalg
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from numpy import cos, sin, arccos, arctan2, sqrt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


class Srelu(nn.Module):#定义Srelu
    def __init__(self) :
        super().__init__()

    def forward(self, z):
        z0_ = z.clone()
        d = torch.tensor(.01)
        z0_[(z0_>= d)] = z[(z0_>= d)]
        z0_[(z0_ <= 0.0)] = 0.0
        z0_[torch.logical_and(z0_ < d, z0_ > 0.0)] = z[torch.logical_and(z < d, z > 0.0)]**2/(2*d)
        return z0_


class ICNN(nn.Module):  # represents the controller gain
    def __init__(self):
        super(ICNN, self).__init__()
        self.w_y1=nn.Parameter(torch.randn(2,16))
        self.w_y2=nn.Parameter(torch.randn(2,1))
        self.w_z0=nn.Parameter(torch.randn(2,16))
        self.w_z1=nn.Parameter(torch.randn(16,16))
        self.w_z2=nn.Parameter(torch.randn(16,1))
        self.srleu=Srelu()
        
    def forward(self, z):
        
        z0 = z.clone()
        z1 = z0 @ self.w_z0 
        z1s =self.srleu(z1)
        z2 = z1s @ torch.relu(self.w_z1)   +z0 @ self.w_y1 
        z2s = self.srleu(z2)
        z3 =  z2s @ torch.relu(self.w_z2)  + z0 @ self.w_y2
        z3s = self.srleu(z3)
        
        return z3s

    """def relu_grad(self, z):
         z_ = z.clone()
         # z_.squeeze_()
         z_[(z_ > 0.0)] = 1.0
         z_[(z_ <= 0.0)] = 0.0
         return z_"""

    def relu_grad(self, z):
        d = torch.tensor(.01)
        z_ = z.clone()
        z_[(z_ >= d)] = 1.0
        z_[(z_ <= 0.0)] = 0.0
        z_[torch.logical_and(z_ < d, z_ > 0.0)] = z[torch.logical_and(z < d, z > 0.0)]/d
        return z_

    def grad_x(self, z):
        with torch.no_grad():
            dz1_dx = self.w_z0.T
            z0 = z.clone()
            z1 = z0 @ self.w_z0 
            z1s = self.srleu(z1) 
            dz1r_dz1 = self.relu_grad(z1)
            dz1r_dz1_ = dz1r_dz1.unsqueeze(2).repeat(1,1,dz1_dx.shape[1])
            dz1_dx_ = dz1_dx.expand_as(dz1r_dz1_)
            dz1r_dx =dz1_dx_ * dz1r_dz1_


            z2 = z1s @ torch.relu(self.w_z1)   +z0 @ self.w_y1  
            dz2r_dz = self.relu_grad(z2)
            z2s = self.srleu(z2)
            dzz1_dx =   torch.relu(self.w_z1.T) @ dz1r_dx
            dzy1_dx = self.w_y1.T
            dzy1_dx_plus_dzz1_dx = dzy1_dx + dzz1_dx
            dz2r_dz_ = dz2r_dz.unsqueeze(2).repeat(1, 1, dzy1_dx_plus_dzz1_dx.shape[2])
            dz2r_dx = dz2r_dz_ * dzy1_dx_plus_dzz1_dx

            z3 = z2s @ torch.relu(self.w_z2)   +z0 @ self.w_y2  
            dz3r_dz = self.relu_grad(z3)
            
            dzz2_dx = torch.relu(self.w_z2.T) @ dz2r_dx 
            dzy2_dx = self.w_y2.T
            dzy2_dx_plus_dzz2_dx = dzy2_dx + dzz2_dx
            dz3r_dz_ = dz3r_dz.unsqueeze(2).repeat(1, 1, dzy2_dx_plus_dzz2_dx.shape[2])
            dz3r_dx = dz3r_dz_ * dzy2_dx_plus_dzz2_dx 

           
            return dz3r_dx.squeeze(1)


class Damping(nn.Module):  # represents the controller gain
    def __init__(self):
        super(Damping, self).__init__()
        N = 2
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        damp_min=torch.tensor([0.001,0.001])
        self.damp_min = damp_min
        
        self.w_d1=nn.Parameter(torch.randn(2,16))
        self.w_d2=nn.Parameter(torch.randn(16,16))
        self.w_d3=nn.Parameter(torch.randn(16,2))
        self.w_o1=nn.Parameter(torch.randn(2,16))
        self.w_o2=nn.Parameter(torch.randn(16,16))
        self.w_o3=nn.Parameter(torch.randn(16,1))
        self.b_d1=nn.Parameter(torch.randn(16))
        self.b_d2=nn.Parameter(torch.randn(16))
        self.b_d3=nn.Parameter(torch.randn(2))
        self.b_o1=nn.Parameter(torch.randn(16))
        self.b_o2=nn.Parameter(torch.randn(16))
        self.b_o3=nn.Parameter(torch.randn(1))

    def forward(self, input):
        x = input

        d1 = x @ self.w_d1   + self.b_d1
        d1t = torch.tanh(d1)
        d2 =  d1t @ self.w_d2 + self.b_d2
        d2t = torch.tanh(d2)
        d3 = d2t @ self.w_d3  + self.b_d3
        d3r = (torch.relu(d3)+self.damp_min) * x
    


        n = self.diag_output_dim
        diag_idx = np.diag_indices(n)
        off_diag_idx = np.tril_indices(n, k=-1)
        D = torch.zeros(x.shape[0], n)

        o1 =  x @ self.w_o1 + self.b_o1
        o1t = torch.tanh(o1)
        o2 = o1t @ self.w_o2  + self.b_o2
        o2t = torch.tanh(o2)
        o3 = o2t @ self.w_o3  + self.b_o3



        for i in range(x.shape[0]):
            L = torch.zeros(n, n)
            diag_elements = d3r[i]
            off_diag_elements = o3[i]
            L[off_diag_idx] = off_diag_elements
            L[diag_idx] = diag_elements
            D_temp = L@L.t()
            D[i] = D_temp @ x[i]
        return D



class DES_control_learner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        coord_dim=2
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self.init_quad_pot = 1.0,
        self.min_quad_pot = 1e-3,
        self.max_quad_pot = 1e1,
        self.icnn_min_lr = 1e-1,

        init_quad_pot_param = torch.ones(coord_dim)*torch.Tensor([1.0])
        self._quad_pot_param = init_quad_pot_param
        self._min_quad_pot_param = torch.Tensor([1e-3])
        self._max_quad_pot_param = torch.Tensor([1e1])


        self.icnn_module=ICNN()
        self.damping_module=Damping()
        self.horizon = 5
        self.state = torch.zeros(
            (self.horizon+1,4), requires_grad=True).to(self.device)
        self.control = torch.zeros(
            (self.horizon,2), requires_grad=True).to(self.device)
        self.stage_cost = torch.zeros(
             self.horizon+1, requires_grad=True).to(self.device)
        dt=0.01
        self.dt=dt
        self.m1=1.0
        self.m2=1.0
        self.l1=1.0
        self.l2=1.0
        self.g = 9.8
        
        

    def get_action(self, *inputs):
        state = torch.zeros((1,4), requires_grad=False).to(self.device)
        state[0] = inputs[0]
        x = state[:,:2] 
        x_dot = state[:,2:]

        quad_pot = self._quad_pot_param.clamp(
            min=(self._min_quad_pot_param.item()),
            max=(self._max_quad_pot_param.item()))

        self.quad_pot=quad_pot  
        self.u_pot = torch.reshape(-self.icnn_module.grad_x(x),[1,2])
        
        self.u_quad = -torch.diag(quad_pot) @ state[:,:2].t()
        self.u_quad.t_()
        self.u_damp = -self.damping_module(x_dot)
        self.x=x   
        self.x_dot=x_dot
        return self.u_pot, self.u_quad, self.u_damp

    def Gravity(self, x):
        
        b_2=x
        M=torch.tensor([[(self.m1+self.m2)*self.l1**2+self.m2*self.l2**2+2*self.m2*self.l1*self.l2*torch.cos(b_2),self.m2*self.l2**2+self.m2*self.l2*self.l1*torch.cos(b_2)],
                                    [self.m2*self.l1*self.l2*torch.cos(b_2)+self.m2*self.l2**2,self.m2*self.l2**2]])
        return M
    
    def gravity_compensate(self,x):
        b1=x[0]
        b2=x[1]
        gc=torch.tensor([2*self.g*torch.cos(b1)+self.g*torch.cos(b1+b2),self.g*torch.cos(b1+b2)])
        return gc
    
    def f(self,x,u):
        b_1=x[0]
        b_2=x[1]
        v_1=x[2]
        v_2=x[3]
        u_1=u[0]
        u_2=u[1]
        M=self.Gravity(b_2)
        A=torch.tensor([[-self.m2*self.l1*self.l2*(2*v_1*v_2+v_2**2)*torch.sin(b_2)],[self.m2*self.l1*self.l2*(v_1**2)*torch.sin(b_2)]])
        B=torch.tensor([[(self.m1+self.m2)*self.g*self.l1*torch.cos(b_1)+self.m2*self.g*self.l2*torch.cos(b_1+b_2)],[self.m2*self.g*self.l2*torch.cos(b_1+b_2)]])
        ux=torch.tensor([[u_1],[u_2]])
        dot=torch.inverse(M) @ (ux-A-B)
        d1=v_1
        d2=v_2
        d3=dot[0]
        d4=dot[1]

        a1=b_1+self.dt*d1
        a2=b_2+self.dt*d2
        a3=v_1+self.dt*d3
        a4=v_2+self.dt*d4
        state=torch.tensor([a1,a2,a3,a4])
        return state



    def forward(self, x):
        return 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params':self.icnn_module.parameters()},
            {'params':self.damping_module.parameters()}], lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = torch.zeros(1).to(self.device)
        # train_batch represents initial states
        num_traj = train_batch.size()[0]

        for i in range(num_traj):
            state = (self.state*torch.zeros(self.horizon+1,4)).to(self.device)
            control = (self.control*torch.zeros(self.horizon,2)).to(self.device)
            control_wg = (self.control*torch.zeros(self.horizon,2)).to(self.device)
            stage_cost = (self.stage_cost*torch.zeros(self.horizon+1)).to(self.device)
            state[0] = train_batch[i]
            

            for t in range(self.horizon):
                self.get_action(state[t])
                gc=self.gravity_compensate(state[t])
                control[t] = gc + self.u_quad+self.u_pot+self.u_damp 
                control_wg[t]=self.u_quad+self.u_pot+self.u_damp 
                state[t+1] = self.f(state[t],control[t])
                stage_cost[t] = state[t,:].clone() @ state[t,:].clone().t() + control_wg[t].clone() @ control_wg[t].clone().t()
            self.x=state[self.horizon-1:self.horizon,:2].clone()
            
            stage_cost[self.horizon] =state[self.horizon,:].clone() @ state[self.horizon,:].clone().t()
            loss = loss + torch.sum(stage_cost)
        print(loss)
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        state = (self.state*torch.zeros(self.horizon+1,4)).to(self.device)
        control = (self.control*torch.zeros(self.horizon,2)).to(self.device)
        control_wg = (self.control*torch.zeros(self.horizon,2)).to(self.device)
        stage_cost = (self.stage_cost*torch.zeros(self.horizon+1)).to(self.device)
        state[0] = test_batch
        
        for t in range(self.horizon):
            
            self.get_action(state[t])
            gc=self.gravity_compensate(state[t])
            control[t] = gc + self.u_quad+self.u_pot+self.u_damp
            control_wg[t] = self.u_quad+self.u_pot+self.u_damp 
            stage_cost[t] = state[t,:].clone() @ state[t,:].clone().t() + control_wg[t].clone() @ control_wg[t].clone().t()
            state[t+1] = self.f(state[t],control[t])
        self.x=state[self.horizon-1:self.horizon,:2]
        stage_cost[self.horizon] = state[self.horizon,:].clone() @ state[self.horizon,:].clone().t()        
        loss =torch.sum(stage_cost)
            
        self.log('test_loss', loss)
        return loss
    
    def plottrajetory(self):
        T=20000
        state = torch.zeros(T+1,4).to(self.device)
        control = torch.zeros(T,2).to(self.device)
        state[0] = torch.tensor([1,1,0,0])
        time = torch.zeros(T,1).to(self.device)
        for t in range(T):
            self.get_action(state[t],x_0=state[0,:2])
            gc=self.gravity_compensate(state[t])
            control[t] = gc + self.u_quad+self.u_pot+self.u_damp
            time[t]=torch.tensor(self.dt*t)
            state[t+1] = self.f(state[t],control[t])

        plt.figure()
        fig,ax= plt.subplots(1,1)
        ax.set_xlabel(r'time ')
        ax.set_ylabel(r'robot link angles')

        # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
        #         'tab:olive', 'tab:cyan', '#90ee90', '#c20078']

        ax.grid(linestyle='-')

        ax.plot(time[:T],state[:T,0].detach(), color='tab:blue',linewidth=1.5, label=r'$\beta_1$')
        ax.plot(time[:T],state[:T,1].detach(), color='tab:orange',linewidth=1.5, label=r'$\beta_2$')
        plt.legend()
        plt.show()
        return state

if __name__ == '__main__':
        training_data = torch.tensor([
        [ 2.9631, -1.2119,0,0],
        [ 2.6983, -1.4708,0,0],
        [ 1.5784,  0.0645,0,0],
        [ 1.2377, -0.0705,0,0],
        [-2.9282, -2.7014,0,0],
        [-1.1126, -1.7541,0,0],
        [ 0.3949, -0.2294,0,0],
        [-2.6270, -1.9890,0,0],
        [ 2.1363, -0.9139,0,0],
        [-2.4482, -2.0395,0,0],
        [ 1.9694,  0.1532,0,0],
        [-0.6018, -1.3801,0,0],
        [-1.8443, -0.3834,0,0],
        [ 0.9002, -1.5965,0,0],
        [-2.0063,  0.6002,0,0],
        [ 0.2906,  2.6184,0,0],
        [ 0.0635,  1.0226,0,0],
        [ 1.9227, -2.6288,0,0],
        [ 1.9413, -0.8881,0,0],
        [-1.6996,  0.8763,0,0]
        ])
                                    
        
        train_dataloader = DataLoader(training_data, batch_size=20)

        model = DES_control_learner()
        # training

        trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
                            callbacks=[], max_epochs=300)


        # model = DES_control_learner.load_from_checkpoint(
        #     checkpoint_path="model_convert_data.ckpt",strict=False)
        

        trainer.fit(model, train_dataloader)
        trainer.save_checkpoint("DES_model_2link_20.ckpt")

        train_model=DES_control_learner.load_from_checkpoint(
            checkpoint_path="DES_model_2link_20.ckpt")
        torch.save(train_model.state_dict(),'DES_model_2link_20.pth')
        
        # testing_data = torch.tensor([[-3,1,0,0]])
        # test_dataloader = DataLoader(testing_data, batch_size=1)
        # trainer.test(train_model, test_dataloader)

        # model.plottrajetory()

