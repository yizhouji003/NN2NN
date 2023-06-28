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
# from normal_energy_shaping import normal_control_learner
from normal_DES import DES_control_learner


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

class Controller_w(nn.Module):  # represents the controller weight
    def __init__(self):
        super(Controller_w,self).__init__()
        self.f01 = nn.Linear(4, 64)
        self.f02 = nn.Linear(64, 128)
        self.f03 = nn.Linear(128, 128)
        self.f04 = nn.Linear(128, 128)
        self.f05 = nn.Linear(128, 128)
        self.f06 = nn.Linear(128, 128)
        self.f07 = nn.Linear(128, 32)
        

        self.f11 = nn.Linear(4, 64)
        self.f12 = nn.Linear(64, 64)
        self.f13 = nn.Linear(64, 64)
        self.f14 = nn.Linear(64, 64)
        self.f15 = nn.Linear(64, 64)
        self.f16 = nn.Linear(64, 64)
        self.f17 = nn.Linear(64, 2)



        self.f21 = nn.Linear(4, 64)
        self.f22 = nn.Linear(64, 128)
        self.f23 = nn.Linear(128,128)
        self.f24 = nn.Linear(128,128)
        self.f25 = nn.Linear(128,128)
        self.f26 = nn.Linear(128, 128)
        self.f27 = nn.Linear(128, 32)



        self.f31 = nn.Linear(4, 64)
        self.f32 = nn.Linear(64, 128)
        self.f33 = nn.Linear(128,128)
        self.f34 = nn.Linear(128,128)
        self.f35 = nn.Linear(128,128)
        self.f36 = nn.Linear(128, 128)
        self.f37 = nn.Linear(128, 16*16)

        
        self.f41 = nn.Linear(4, 64)
        self.f42 = nn.Linear(64, 64)
        self.f43 = nn.Linear(64,64)
        self.f44 = nn.Linear(64,64)
        self.f45 = nn.Linear(64,64)
        self.f46 = nn.Linear(64, 64)
        self.f47 = nn.Linear(64, 16)
        



        self.y01 = nn.Linear(4, 64)
        self.y02 = nn.Linear(64, 128)
        self.y03 = nn.Linear(128,128)
        self.y04 = nn.Linear(128,128)
        self.y05 = nn.Linear(128,128)
        self.y06 = nn.Linear(128, 128)
        self.y07= nn.Linear(128, 32)
        


        self.y11 = nn.Linear(4, 64)
        self.y12 = nn.Linear(64, 128)
        self.y13 = nn.Linear(128,128)
        self.y14 = nn.Linear(128,128)
        self.y15 = nn.Linear(128,128)
        self.y16 = nn.Linear(128, 128)
        self.y17 = nn.Linear(128, 16*16)


        self.y21 = nn.Linear(4, 64)
        self.y22 = nn.Linear(64, 128)
        self.y23 = nn.Linear(128, 128)
        self.y24 = nn.Linear(128,128)
        self.y25 = nn.Linear(128,128)
        self.y26 = nn.Linear(128, 128)
        self.y27 = nn.Linear(128, 32)


        self.y31 = nn.Linear(4, 64)
        self.y32 = nn.Linear(64, 128)
        self.y33 = nn.Linear(128, 128)
        self.y34 = nn.Linear(128,128)
        self.y35 = nn.Linear(128,128)
        self.y36 = nn.Linear(128, 128)
        self.y37 = nn.Linear(128, 32)

        
        self.y41 = nn.Linear(4, 64)
        self.y42 = nn.Linear(64, 128)
        self.y43 = nn.Linear(128, 128)
        self.y44 = nn.Linear(128,128)
        self.y45 = nn.Linear(128,128)
        self.y46 = nn.Linear(128, 128)
        self.y47 = nn.Linear(128, 16*16)



        self.y51 = nn.Linear(4, 64)
        self.y52 = nn.Linear(64, 64)
        self.y53 = nn.Linear(64,64)
        self.y54 = nn.Linear(64,64)
        self.y55 = nn.Linear(64,64)
        self.y56 = nn.Linear(64, 64)
        self.y57 = nn.Linear(64, 16)



    def forward(self, z):
        x=z.clone()
        x01 = self.f01(x)
        x011 = torch.tanh(x01)
        x012 = self.f02(x011)
        x013 = torch.tanh(x012)
        x014 = self.f03(x013)
        x015 = torch.tanh(x014)
        x016 = self.f04(x015)
        x017 = torch.tanh(x016)
        x018 = self.f05(x017)
        x019 = torch.tanh(x018)
        x02=torch.tanh(self.f06(x019))
        x0 = torch.tanh(self.f07(x02))
        w_y1 = torch.reshape(x0,[2,16])

        x11 = self.f11(x)
        x111 = torch.tanh(x11)
        x112 = self.f12(x111)
        x113 = torch.tanh(x112)
        x114 = self.f13(x113)
        x115 = torch.tanh(x114)
        x116 = self.f14(x115)
        x117 = torch.tanh(x116)
        x118 = self.f15(x117)
        x119 = torch.tanh(x118)
        x12=torch.tanh(self.f16(x119))
        x1 = torch.tanh(self.f17(x12))
        w_y2 = torch.reshape(x1,[2,1])

        x21 = self.f21(x)
        x211 = torch.tanh(x21)
        x212 = self.f22(x211)
        x213 = torch.tanh(x212)
        x214 = self.f23(x213)
        x215 = torch.tanh(x214)
        x216 = self.f24(x215)
        x217 = torch.tanh(x216)
        x218 = self.f25(x217)
        x219 = torch.tanh(x218)
        x22=torch.tanh(self.f26(x219))
        x2 = torch.tanh(self.f27(x22))
        w_z0 = torch.reshape(x2,[2,16])

        x31 = self.f31(x)
        x311 = torch.tanh(x31)
        x312 = self.f32(x311)
        x313 = torch.tanh(x312)
        x314 = self.f33(x313)
        x315 = torch.tanh(x314)
        x316 = self.f34(x315)
        x317 = torch.tanh(x316)
        x318 = self.f35(x317)
        x319 = torch.tanh(x318)
        x32=torch.tanh(self.f36(x319))
        x3 = torch.tanh(self.f37(x32))
        # x3 = torch.relu(torch.tanh(self.f37(x32)))
        w_z1 = torch.reshape(x3,[16,16])

        x41 = self.f41(x)
        x411 = torch.tanh(x41)
        x412 = self.f42(x411)
        x413 = torch.tanh(x412)
        x414 = self.f43(x413)
        x415 = torch.tanh(x414)
        x416 = self.f44(x415)
        x417 = torch.tanh(x416)
        x418 = self.f45(x417)
        x419 = torch.tanh(x418)
        x42=torch.tanh(self.f46(x419))
        x4 = torch.tanh(self.f47(x42))
        # x4 = torch.relu(torch.tanh(self.f47(x42)))
        w_z2 = torch.reshape(x4,[16,1])
        
        z0_1 = self.y01(x)
        z0_11 = torch.tanh(z0_1)
        z0_12 = self.y02(z0_11)
        z0_13 = torch.tanh(z0_12)
        z0_14 = self.y03(z0_13)
        z0_15 = torch.tanh(z0_14)
        z0_16 = self.y04(z0_15)
        z0_17 = torch.tanh(z0_16)
        z0_18 = self.y05(z0_17)
        z0_19 = torch.tanh(z0_18)
        z0_2 = torch.tanh(self.y06(z0_19))
        z0 = torch.tanh(self.y07(z0_2))
        w_d1 = torch.reshape(z0,[2,16])
        
        z1_1 = self.y11(x)
        z1_11 = torch.tanh(z1_1)
        z1_12 = self.y12(z1_11)
        z1_13 = torch.tanh(z1_12)
        z1_14 = self.y13(z1_13)
        z1_15 = torch.tanh(z1_14)
        z1_16 = self.y14(z1_15)
        z1_17 = torch.tanh(z1_16)
        z1_18 = self.y15(z1_17)
        z1_19 = torch.tanh(z1_18)
        z1_2 = torch.tanh(self.y16(z1_19))
        z1 = torch.tanh(self.y17(z1_2))
        w_d2 = torch.reshape(z1,[16,16])

        z2_1 = self.y21(x)
        z2_11 = torch.tanh(z2_1)
        z2_12 = self.y22(z2_11)
        z2_13 = torch.tanh(z2_12)
        z2_14 = self.y23(z2_13)
        z2_15 = torch.tanh(z2_14)
        z2_16 = self.y24(z2_15)
        z2_17 = torch.tanh(z2_16)
        z2_18 = self.y25(z2_17)
        z2_19 = torch.tanh(z2_18)
        z2_2 = torch.tanh(self.y26(z2_19))
        z2 = torch.tanh(self.y27(z2_2))
        w_d3= torch.reshape(z2,[16,2])

        z310 = self.y31(x)
        z311 = torch.tanh(z310)
        z312 = self.y32(z311)
        z313 = torch.tanh(z312)
        z314 = self.y33(z313)
        z315 = torch.tanh(z314)
        z316 = self.y34(z315)
        z317 = torch.tanh(z316)
        z318 = self.y35(z317)
        z319 = torch.tanh(z318)
        z32 = torch.tanh(self.y36(z319))
        z3 = torch.tanh(self.y37(z32))
        w_o1 = torch.reshape(z3,[2,16])

        z410 = self.y41(x)
        z411 = torch.tanh(z410)
        z412 = self.y42(z411)
        z413 = torch.tanh(z412)
        z414 = self.y43(z413)
        z415 = torch.tanh(z414)
        z416 = self.y44(z415)
        z417 = torch.tanh(z416)
        z418 = self.y45(z417)
        z419 = torch.tanh(z418)
        z42 = torch.tanh(self.y46(z419))
        z4 = torch.tanh(self.y47(z42))
        w_o2 = torch.reshape(z4,[16,16])

        z510 = self.y51(x)
        z511 = torch.tanh(z510)
        z512 = self.y52(z511)
        z513 = torch.tanh(z512)
        z514 = self.y53(z513)
        z515 = torch.tanh(z514)
        z516 = self.y54(z515)
        z517 = torch.tanh(z516)
        z518 = self.y55(z517)
        z519 = torch.tanh(z518)
        z52 = torch.tanh(self.y56(z519))
        z5 = torch.tanh(self.y57(z52))
        w_o3 = torch.reshape(z5,[16,1])
       
        

        return w_y1,w_y2,w_z0,w_z1,w_z2,w_d1,w_d2,w_d3,w_o1,w_o2,w_o3

class Controller_b(nn.Module):  # represents the controller bias
    def __init__(self):
        super(Controller_b,self).__init__()

        self.y01 = nn.Linear(4, 32)
        self.y02 = nn.Linear(32, 32)
        # self.y03 = nn.Linear(32,32)
        # self.y04 = nn.Linear(32,64)
        # self.y05 = nn.Linear(64,64)
        # self.y06 = nn.Linear(64, 64)
        self.y07 = nn.Linear(32, 16)



        self.y11 = nn.Linear(4, 32)
        self.y12 = nn.Linear(32, 32)
        # self.y13 = nn.Linear(32,32)
        # self.y14 = nn.Linear(32,64)
        # self.y15 = nn.Linear(64,64)
        # self.y16 = nn.Linear(64, 64)
        self.y17 = nn.Linear(32, 16)



        self.y21 = nn.Linear(4, 32)
        self.y22 = nn.Linear(32, 32)
        # self.y23 = nn.Linear(32,32)
        # self.y24 = nn.Linear(32,32)
        # self.y25 = nn.Linear(32,32)
        # self.y26 = nn.Linear(32, 32)
        self.y27 = nn.Linear(32, 2)


        self.y31 = nn.Linear(4, 32)
        self.y32 = nn.Linear(32, 32)
        # self.y33 = nn.Linear(32,32)
        # self.y34 = nn.Linear(32,32)
        # self.y35 = nn.Linear(32,32)
        # self.y36 = nn.Linear(32, 32)
        self.y37 = nn.Linear(32, 16)


        self.y41 = nn.Linear(4, 32)
        self.y42 = nn.Linear(32, 32)
        # self.y43 = nn.Linear(32,32)
        # self.y44 = nn.Linear(32,32)
        # self.y45 = nn.Linear(32,32)
        # self.y46 = nn.Linear(32, 32)
        self.y47 = nn.Linear(32, 16)


        self.y51 = nn.Linear(4, 32)
        self.y52 = nn.Linear(32, 64)
        # self.y53 = nn.Linear(64,64)
        # self.y54 = nn.Linear(64,64)
        # self.y55 = nn.Linear(64,32)
        # self.y56 = nn.Linear(32, 32)
        self.y57 = nn.Linear(64, 1)



    def forward(self, z):
        x = z.clone()

        z0_1 = self.y01(x)
        z0_11 = torch.tanh(z0_1)
        z0_12 = self.y02(z0_11)
        z0_13 = torch.tanh(z0_12)
        # z0_14 = self.y03(z0_13)
        # z0_15 = torch.tanh(z0_14)
        # z0_16 = self.y04(z0_15)
        # z0_17 = torch.tanh(z0_16)
        # z0_18 = self.y05(z0_17)
        # z0_19 = torch.tanh(z0_18)
        # z0_20 = self.y06(z0_19)
        # z0_2 = torch.tanh(z0_20)
        z0 = torch.tanh(self.y07(z0_13))

        z1_1 = self.y11(x)
        z1_11 = torch.tanh(z1_1)
        z1_12 = self.y12(z1_11)
        z1_13 = torch.tanh(z1_12)
        # z1_14 = self.y13(z1_13)
        # z1_15 = torch.tanh(z1_14)
        # z1_16 = self.y14(z1_15)
        # z1_17 = torch.tanh(z1_16)
        # z1_18 = self.y15(z1_17)
        # z1_19 = torch.tanh(z1_18)
        # z1_20 = self.y16(z1_19)
        # z1_2 = torch.tanh(z1_20)
        z1 = torch.tanh(self.y17(z1_13))

        z2_1 = self.y21(x)
        z2_11 = torch.tanh(z2_1)
        z2_12 = self.y22(z2_11)
        z2_13 = torch.tanh(z2_12)
        # z2_14 = self.y23(z2_13)
        # z2_15 = torch.tanh(z2_14)
        # z2_16 = self.y24(z2_15)
        # z2_17 = torch.tanh(z2_16)
        # z2_18 = self.y25(z2_17)
        # z2_19 = torch.tanh(z2_18)
        # z2_20 = self.y26(z2_19)
        # z2_2 = torch.tanh(z2_20)
        z2 = torch.tanh(self.y27(z2_13))

        z310 = self.y31(x)
        z311 = torch.tanh(z310)
        z312 = self.y32(z311)
        z313 = torch.tanh(z312)
        # z314 = self.y33(z313)
        # z315 = torch.tanh(z314)
        # z316 = self.y34(z315)
        # z317 = torch.tanh(z316)
        # z318 = self.y35(z317)
        # z319 = torch.tanh(z318)
        # z320 = self.y36(z319)
        # z32 = torch.tanh(z320)
        z3 = torch.tanh(self.y37(z313))

        z410 = self.y41(x)
        z411 = torch.tanh(z410)
        z412 = self.y42(z411)
        z413 = torch.tanh(z412)
        # z414 = self.y43(z413)
        # z415 = torch.tanh(z414)
        # z416 = self.y44(z415)
        # z417 = torch.tanh(z416)
        # z418 = self.y45(z417)
        # z419 = torch.tanh(z418)
        # z420 = self.y46(z419)
        # z42 = torch.tanh(z420)
        z4 = torch.tanh(self.y47(z413))

        z510 = self.y51(x)
        z511 = torch.tanh(z510)
        z512 = self.y52(z511)
        z513 = torch.tanh(z512)
        # z514 = self.y53(z513)
        # z515 = torch.tanh(z514)
        # z516 = self.y54(z515)
        # z517 = torch.tanh(z516)
        # z518 = self.y55(z517)
        # z519 = torch.tanh(z518)
        # z520 = self.y56(z519)
        # z52 = torch.tanh(z520)
        z5 = torch.tanh(self.y57(z513))



        return z0,z1,z2,z3,z4,z5
    
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
    def __init__(self,Controller_w
                 ):
        super(ICNN, self).__init__()
        self._control_w=Controller_w
        self.srleu=Srelu()
        
    def forward(self, z, x_0):
        w_y1,w_y2,w_z0,w_z1,w_z2,w_d1,w_d2,w_d3,w_o1,w_o2,w_o3=self._control_w(x_0)
        z0 = z.clone()
        z1 = z0 @ w_z0 
        z1s =self.srleu(z1)
        z2 = z1s @ torch.relu(w_z1)   +z0 @ w_y1 
        z2s = self.srleu(z2)
        z3 =  z2s @ torch.relu(w_z2)  + z0 @ w_y2
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

    def grad_x(self, z,x_0):
        with torch.no_grad():
            w_y1,w_y2,w_z0,w_z1,w_z2,w_d1,w_d2,w_d3,w_o1,w_o2,w_o3=self._control_w(x_0)
            dz1_dx = w_z0.T
            z0 = z.clone()
            z1 = z0 @ w_z0 
            z1s = self.srleu(z1) 
            dz1r_dz1 = self.relu_grad(z1)
            dz1r_dz1_ = dz1r_dz1.unsqueeze(2).repeat(1,1,dz1_dx.shape[1])
            dz1_dx_ = dz1_dx.expand_as(dz1r_dz1_)
            dz1r_dx =dz1_dx_ * dz1r_dz1_


            z2 = z1s @ torch.relu(w_z1)   +z0 @ w_y1  
            dz2r_dz = self.relu_grad(z2)
            z2s = self.srleu(z2)
            dzz1_dx =   torch.relu(w_z1.T) @ dz1r_dx
            dzy1_dx = w_y1.T
            dzy1_dx_plus_dzz1_dx = dzy1_dx + dzz1_dx
            dz2r_dz_ = dz2r_dz.unsqueeze(2).repeat(1, 1, dzy1_dx_plus_dzz1_dx.shape[2])
            dz2r_dx = dz2r_dz_ * dzy1_dx_plus_dzz1_dx

            z3 = z2s @ torch.relu(w_z2)   +z0 @ w_y2  
            dz3r_dz = self.relu_grad(z3)
            z3s = self.srleu(z3) 
            dzz2_dx = torch.relu(w_z2.T) @ dz2r_dx 
            dzy2_dx = w_y2.T
            dzy2_dx_plus_dzz2_dx = dzy2_dx + dzz2_dx
            dz3r_dz_ = dz3r_dz.unsqueeze(2).repeat(1, 1, dzy2_dx_plus_dzz2_dx.shape[2])
            dz3r_dx = dz3r_dz_ * dzy2_dx_plus_dzz2_dx 

           
            return dz3r_dx.squeeze(1)


class Damping(nn.Module):  # represents the controller gain
    def __init__(self,Controller_w,Controller_b):
        super(Damping, self).__init__()
        self._control_w=Controller_w
        self._control_b=Controller_b
        N = 2
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        damp_min=torch.tensor([0.001,0.001])
        self.damp_min = damp_min#


    def forward(self, input, x_0):
        # todo this method is not ready for batch input data
        # x = input.view(1,-1)
        x = input
        x0=x.clone()
        z=x.clone()
        w_y1,w_y2,w_z0,w_z1,w_z2,w_d1,w_d2,w_d3,w_o1,w_o2,w_o3=self._control_w(x_0)
        b_d1,b_d2,b_d3,b_o1,b_o2,b_o3=self._control_b(x_0)




        d1 = x @ w_d1   + b_d1
        d1t = torch.tanh(d1)
        d2 =  d1t @ w_d2 + b_d2
        d2t = torch.tanh(d2)
        d3 = d2t @ w_d3  + b_d3
        d3r = (torch.relu(d3)+self.damp_min) * x
    


        n = self.diag_output_dim
        diag_idx = np.diag_indices(n)
        off_diag_idx = np.tril_indices(n, k=-1)
        D = torch.zeros(x.shape[0], n)

        o1 =  x @ w_o1 + b_o1
        o1t = torch.tanh(o1)
        o2 = o1t @ w_o2  + b_o2
        o2t = torch.tanh(o2)
        o3 = o2t @ w_o3  + b_o3



        for i in range(x.shape[0]):
            L = torch.zeros(n, n)
            diag_elements = d3r[i]
            off_diag_elements = o3[i]
            L[off_diag_idx] = off_diag_elements
            L[diag_idx] = diag_elements
            D_temp = L@L.t()
            D[i] = D_temp @ x[i]
        return D



class control_learner(pl.LightningModule):
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

        self.controller_w=Controller_w()
        self.controller_b=Controller_b()
        self.icnn_module=ICNN(self.controller_w)
        self.damping_module=Damping(self.controller_w,self.controller_b)
        

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
        
        

    def get_action(self, *inputs, x_0):
        state = torch.zeros((1,4), requires_grad=False).to(self.device)
        state[0] = inputs[0]
        x = state[:,:2] 
        x_dot = state[:,2:]

        quad_pot = self._quad_pot_param.clamp(
            min=(self._min_quad_pot_param.item()),
            max=(self._max_quad_pot_param.item()))

        self.quad_pot=quad_pot  
        self.u_pot = torch.reshape(-self.icnn_module.grad_x(x,x_0),[1,2])
        
        self.u_quad = -torch.diag(quad_pot) @ state[:,:2].t()
        self.u_quad.t_()
        self.u_damp = -self.damping_module(x_dot,x_0)
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
        optimizer = torch.optim.Adam([{'params':self.controller_w.parameters()},
            {'params':self.controller_b.parameters()}], lr=4e-4)
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
                self.get_action(state[t],x_0=state[0,:])
                gc=self.gravity_compensate(state[t])
                control[t] =gc+ self.u_quad+self.u_pot+self.u_damp 
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
            self.get_action(state[t],x_0=state[0,:])
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
        T=50000
        state = torch.zeros(T+1,4).to(self.device)
        control = torch.zeros(T,2).to(self.device)
        state[0] = torch.tensor([-1,1,0,0])
        time = torch.zeros(T,1).to(self.device)
        for t in range(T):


            self.get_action(state[t],x_0=state[0,:])
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

        model = control_learner()
        # training

        trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
                            callbacks=[], max_epochs=300)
        

        # trainer.fit(model, train_dataloader)
        # trainer.save_checkpoint("model_2link_24.ckpt")

        train_model=control_learner.load_from_checkpoint(
            checkpoint_path="model_2link_24.ckpt")
        

        train_model.plottrajetory()

        # test_model = normal_control_learner.load_from_checkpoint(
        #     checkpoint_path="model_2link_normal20.ckpt"
        # )

        DES_model = DES_control_learner.load_from_checkpoint(
            checkpoint_path="DES_model_2link_20.ckpt"
        )

        fig = plt.figure()

        ax1 = fig.add_subplot(projection = '3d')
 
        # Make data.
        X = np.arange(-3.2, 3.2, 0.1)
        Y = np.arange(-3.2, 3.2, 0.1)
        X, Y = np.meshgrid(X, Y)
        z=np.empty((64,64))
        for i in range (64):
            for j in range (64):
                testing_data = torch.tensor([[X[i,j],Y[i,j],0,0]])
                print('testing_data',testing_data)
                test_dataloader = DataLoader(testing_data, batch_size=1)

                # loss1 = trainer.test(train_model, test_dataloader)
                # z[i,j]=loss1[0]['test_loss']        #NN2NN

                # loss2= trainer.test(test_model,test_dataloader)
                # z[i,j]=loss2[0]['test_loss']        #normal

                loss1 = trainer.test(train_model, test_dataloader)
                loss2 = trainer.test(DES_model,test_dataloader)
                z[i,j]=loss2[0]['test_loss']-loss1[0]['test_loss']    #normal-NN2NN
                
                


        
        surf = ax1.plot_surface(X, Y, z, 
        cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        
        
        ax1.zaxis.set_major_locator(LinearLocator(10))
        ax1.set_xlabel(r'$\beta_1$')
        ax1.set_ylabel(r'$\beta_2$')
        ax1.set_zlabel(r'$J_{DES}-J_{NNtoNN}$' , labelpad=10)
        ax1.set_zticks([0,300,600,900,1200]) 
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        
        plt.show()
