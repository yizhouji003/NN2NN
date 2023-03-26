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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


class InitialState(Dataset):
    def __init__(self, x0_min, x0_max, num_traj):
        self.x0 = torch.linspace(x0_min, x0_max, num_traj)
        

    def __len__(self):
        return len(self.x0)
    def __getitem__(self, idx):
        return self.x0[idx]


class controller(nn.Module):  # represents the controller gain
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 1)

        # self.linear=nn.Linear(1,1)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = torch.tanh(x1)
        x3 = self.fc2(x2)
        x4 = torch.tanh(x3)
        x5 = self.fc3(x4)
        x6 = torch.tanh(x5)
        x7 = self.fc4(x6)
        x8 = torch.tanh(x7)
        x9 = self.fc5(x8)
        x10 = torch.tanh(x9)
        x11 = self.fc6(x10)

        # x11=self.linear(x)

        return x11


class control_learner(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.controller = controller()
        self.horizon = 1
        self.state = torch.zeros(
            self.horizon+1, requires_grad=True).to(self.device)
        self.control = torch.zeros(
            self.horizon, requires_grad=True).to(self.device)
        self.stage_cost = torch.zeros(
            self.horizon+1, requires_grad=True).to(self.device)

    def forward(self, x):
        return 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=5e-3)
        # return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)]
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = torch.zeros(1).to(self.device)
        # train_batch represents initial states
        num_traj = train_batch.size()[0]
        for i in range(num_traj):
            state = (self.state*torch.zeros(1)).to(self.device)
            control = (self.control*torch.zeros(1)).to(self.device)
            stage_cost = (self.stage_cost*torch.zeros(1)).to(self.device)
            state[0] = train_batch[i]
            for t in range(self.horizon):
                control[t] = self.controller(
                    torch.reshape(state[0].clone(), (1,)))*state[t].clone()
                # state[t] = self.controller(torch.reshape(torch.rand(1), (1,)))
                state[t+1] = torch.sin(state[t].clone()) + control[t].clone()
                stage_cost[t] = state[t].clone()**2 + \
                    control[t].clone()**2

            stage_cost[self.horizon] = state[self.horizon]**2

            loss = loss + torch.sum(stage_cost)

        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        state = (self.state*torch.zeros(1)).to(self.device)
        control = (self.control*torch.zeros(1)).to(self.device)
        stage_cost = (self.stage_cost*torch.zeros(1)).to(self.device)
        # train_batch represents initial states
        state[0] = test_batch
        for t in range(self.horizon):
            control[t] = self.controller(
                torch.reshape(state[0].clone(), (1,)))*state[t].clone()
            # state[t] = self.controller(torch.reshape(torch.rand(1), (1,)))
            state[t+1] = torch.sin(state[t].clone()) + control[t].clone()
            stage_cost[t] = state[t].clone()**2 + \
                control[t].clone()**2

        stage_cost[self.horizon] = state[self.horizon]**2

        loss = torch.sum(stage_cost)
        self.log('test_loss', loss)
        return loss


if __name__ == '__main__':

    num_traj = 20
    training_data = InitialState(-5, 5, num_traj)

    train_dataloader = DataLoader(training_data, batch_size=num_traj)

    model = control_learner()
    # training

    trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
                         callbacks=[], max_epochs=2000)

    trainer.fit(model, train_dataloader)
    trainer.save_checkpoint("model_nonlinear_toy.ckpt")

    new_model = control_learner.load_from_checkpoint(
        checkpoint_path="model_nonlinear_toy.ckpt")

    testing_data = InitialState(1.8, 2, 1)
    test_dataloader = DataLoader(testing_data, batch_size=1)
    trainer.test(new_model, test_dataloader)

    # trainer.save_checkpoint("example_pendulum.ckpt")

    # new_model = neural_energy_casimir.load_from_checkpoint(
    #     checkpoint_path="example_pendulum.ckpt")

    # print(new_model.controller_equilibrium)
