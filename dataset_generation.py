import numpy as np
import math
import torch
import requests
import pickle
from network import Network


def calculate_electric_field(Q1, Q2, r1, r2, r, epsilon_0=8.854e-12):
    r1_vector = r - r1
    r2_vector = r - r2

    r1_magnitude = np.linalg.norm(r1_vector)
    r2_magnitude = np.linalg.norm(r2_vector)

    E1 = (1 / (4 * np.pi * epsilon_0)) * (Q1 / r1_magnitude ** 3) * r1_vector
    E2 = (1 / (4 * np.pi * epsilon_0)) * (Q2 / r2_magnitude ** 3) * r2_vector

    E_total = E1 + E2
    return E_total


def generate_arrays(Q1, Q2, r1, r2, x_range, y_range, z_range, x_step, y_step, z_step):

    x_array = []
    y_array = []
    z_array = []
    Ex_array = []
    Ey_array = []
    Ez_array = []

    # Sampling points are generated according to a step size within the specified range.
    for x in np.arange(x_range[0], x_range[1] + x_step, x_step):
        for y in np.arange(y_range[0], y_range[1] + y_step, y_step):
            for z in np.arange(z_range[0], z_range[1] + z_step, z_step):
                r = np.array([x, y, z])

                # Calculate the electric field.
                E = calculate_electric_field(Q1, Q2, r1, r2, r)

                # Store the coordinates and electric field components in separate arrays.
                x_array.append(x)
                y_array.append(y)
                z_array.append(z)
                Ex_array.append(E[0])
                Ey_array.append(E[1])
                Ez_array.append(E[2])

    x_array = np.array(x_array)
    y_array = np.array(y_array)
    z_array = np.array(z_array)
    Ex_array = np.array(Ex_array)
    Ey_array = np.array(Ey_array)
    Ez_array = np.array(Ez_array)

    x_tensor = torch.from_numpy(x_array).float()
    y_tensor = torch.from_numpy(y_array).float()
    z_tensor = torch.from_numpy(z_array).float()
    Ex_tensor = torch.from_numpy(Ex_array)
    Ey_tensor = torch.from_numpy(Ey_array)
    Ez_tensor = torch.from_numpy(Ez_array)

    return x_tensor, y_tensor, z_tensor, Ex_tensor, Ey_tensor, Ez_tensor


# Define the boundary conditions.
x_range = [10, 110]  # x-direction
y_range = [10, 110]  # y-direction
z_range = [10, 110]  # z-direction

# 采样步长
x_step = 0.5  # x-direction step
y_step = 0.5  # y-direction step
z_step = 0.5  # z-direction step

# Point charge.
Q1 = 1  # Coulomb.
Q2 = -1  # Coulomb.

# Point charge position.
r1 = np.array([0, 0, 100])  # m
r2 = np.array([0, 0, 0])  # m

class EFILN:

    def __init__(self):
        # select GPU or CPU
        # device = torch.device(
        #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = "cuda:0"
        self.if_noise = False #if add noise
        self.xloss = []
        self.yloss = []
        self.zloss = []

        # Define the neural network.
        self.model = Network(
            input_size=3,  # Number of neurons in the input layer.
            hidden_size=16,  # Number of neurons in the hidden layer.
            output_size=3,  # Number of neurons in the output layer.
            depth=8,  # Number of hidden layers.
            act=torch.nn.Tanh  # Activation function of the input layer and hidden layers.
        ).to(self.device)

        self.x_array, self.y_array, self.z_array, self.Ex_array, self.Ey_array, self.Ez_array = generate_arrays(Q1, Q2, r1, r2, x_range, y_range, z_range, x_step, y_step, z_step)

        self.E_inside = torch.stack([self.Ex_array, self.Ey_array, self.Ez_array], dim=-1).reshape(-1, 3)
        self.E_inside = self.E_inside.type(torch.float32)
        if self.if_noise == True:
            noise = torch.randn_like(self.E_inside) * (self.E_inside * 0.1)# add noise
            self.E_inside += noise
        # min values and max values
        min_vals = self.E_inside.min(dim=0, keepdim=True)[0]
        max_vals = self.E_inside.max(dim=0, keepdim=True)[0]
        with open('min_vals.pkl', 'wb') as file:
            # Store the data into a file using pickle.dump.
            pickle.dump(min_vals, file)
        with open('max_vals.pkl', 'wb') as file:
            # Store the data into a file using pickle.dump.
            pickle.dump(max_vals, file)
        x_min_vals = self.x_array.min(dim=0, keepdim=True)[0]
        x_max_vals = self.x_array.max(dim=0, keepdim=True)[0]
        y_min_vals = self.y_array.min(dim=0, keepdim=True)[0]
        y_max_vals = self.y_array.max(dim=0, keepdim=True)[0]
        z_min_vals = self.z_array.min(dim=0, keepdim=True)[0]
        z_max_vals = self.z_array.max(dim=0, keepdim=True)[0]
        # normalization
        self.E_inside = (self.E_inside - min_vals) / (max_vals - min_vals)
        self.x_array = (self.x_array - x_min_vals) / (x_max_vals - x_min_vals)
        self.y_array = (self.y_array - x_min_vals) / (y_max_vals - y_min_vals)
        self.z_array = (self.z_array - z_min_vals) / (z_max_vals - z_min_vals)
        self.x_array = self.x_array.unsqueeze(1)
        self.y_array = self.y_array.unsqueeze(1)
        self.z_array = self.z_array.unsqueeze(1)
        self.x_array = self.x_array.to(self.device)
        self.y_array = self.y_array.to(self.device)
        self.z_array = self.z_array.to(self.device)
        self.E_inside = self.E_inside.to(self.device)
        # self.Ex_array = self.Ex_array.type(torch.float32)
        # self.Ey_array = self.Ey_array.type(torch.float32)
        # self.Ez_array = self.Ez_array.type(torch.float32)
        self.E_inside.requires_grad = True
        # set mseloss
        self.criterion = torch.nn.MSELoss()

        # Define the iteration index to record how many times the loss has been called.
        self.iter = 1

        # set l-bfgs optimizer
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=10.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=0.001 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        #set adam optimizer
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.0001)


    # loss function
    def loss_func(self):

        self.adam.zero_grad()
        self.lbfgs.zero_grad()


        X_pred, Y_pred, Z_pred = self.model(self.E_inside)  # Use the current model to compute the predicted value of u.
        X_loss = self.criterion(
            X_pred, self.x_array)  # MSE
        Y_loss = self.criterion(
            Y_pred, self.y_array)  # MSE
        Z_loss = self.criterion(
            Z_pred, self.z_array)  # MSE

        loss = X_loss + Y_loss + Z_loss

        loss.backward()

        if self.iter % 100 == 0:
            # self.xloss.append(X_loss)
            # self.yloss.append(Y_loss)
            # self.zloss.append(Z_loss)
            self.xloss.append(X_loss.detach().cpu().numpy())
            self.yloss.append(Y_loss.detach().cpu().numpy())
            self.zloss.append(Z_loss.detach().cpu().numpy())
            print(self.iter, loss.item())
        self.iter = self.iter + 1
        return loss

    # 训练
    def train(self):
        self.model.train()

        print("Adam")
        for i in range(12000):
            self.adam.step(self.loss_func)
        print("L-BFGS")
        self.lbfgs.step(self.loss_func)
        with open('xloss.pkl', 'wb') as file:
            pickle.dump(self.xloss, file)
        with open('yloss.pkl', 'wb') as file:
            pickle.dump(self.yloss, file)
        with open('zloss.pkl', 'wb') as file:
            pickle.dump(self.zloss, file)


efiln = EFILN()

efiln.train()


torch.save(efiln.model, 'model.pth')

