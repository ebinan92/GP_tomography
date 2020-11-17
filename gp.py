# %%
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import compare_ssim as ssim

plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
plt.rcParams['font.size'] = 9  # フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

torch.manual_seed(1)
D_DIM = 42 #detector dimensition
MAT_DIM = 64 #reconstructed image dimenstion
NOISE_RATE = 0.125
KERNEL = "SE" # switch between nonstationary(NS) and stationary(SE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RMSELoss(torch.nn.Module):
    # RMSELoss
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class MGLoss(torch.nn.Module):
    # Evidence 
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, model):
        mu = model.d.unsqueeze(1)
        temp = mu.T.mm(model.d_cov_inv).mm(model.R)
        a = temp.mm(model.post_cov.T).mm(temp.T)
        b = torch.logdet(model.post_cov)
        c = torch.logdet(model.f_cov)
        loss = -(b-c + a)
        return loss

class NNGPT(torch.nn.Module):
    # NNGPT 
    def __init__(self, R, d, circle):
        super(NNGPT, self).__init__()
        self.le = torch.nn.Parameter(torch.tensor(1.))
        self.lc = torch.nn.Parameter(torch.tensor(1.))
        self.lm = torch.nn.Parameter(torch.tensor(1.))
        self.l_l = torch.nn.Parameter(torch.tensor(1.))
        self.sigma_l = torch.nn.Parameter(torch.tensor(1.))
        self.sigma_f = torch.nn.Parameter(torch.tensor(1.))
        self.sigma_d = torch.nn.Parameter(torch.tensor(200.).cuda())
        self.R = torch.tensor(R).to(device, dtype=torch.float32)
        self.d_raw = torch.tensor(d).to(device, dtype=torch.float32)
        noise = torch.empty(d.shape[0]).normal_(
            mean=0.0, std=torch.mean(torch.abs(torch.tensor(d)))*NOISE_RATE).to(device, dtype=torch.float32)
        self.d = torch.tensor(d).to(device, dtype=torch.float32) + noise
        self.d_cov = self.sigma_d * \
            torch.eye(self.d.shape[0], self.d.shape[0]).to(
                device, dtype=torch.float32)
        self.d_cov_inv = torch.inverse(self.d_cov).to(
            device, dtype=torch.float32)
        all_region = [(i, j) for i in range(MAT_DIM)
                      for j in range(MAT_DIM)]
        outer_region = list(
            zip(*np.where(((np.abs(circle) < 0.15) & (np.abs(circle) > 0.13)))))
        inner_region = list(
            zip(*np.where(np.abs(circle) > 0.99)))
        middle_region = list(
            zip(*np.where(((np.abs(circle) < 0.45) & (np.abs(circle) > 0.42)))))

        unknown_region = list(
            set(all_region) - (set(outer_region) | set(inner_region) | set(middle_region)))
        norm = 1.2
        self.o_len = len(outer_region)
        self.i_len = len(inner_region)
        self.m_len = len(middle_region)
        self.x = torch.tensor(
            list(map(lambda x: ((x[0]-int(MAT_DIM/2)) * norm, (x[1]-int(MAT_DIM/2)) * norm), all_region))).to(device, dtype=torch.float32)
        self.known_region = torch.tensor(
            list(map(lambda x: ((x[0]-int(MAT_DIM/2)) * norm, (x[1]-int(MAT_DIM/2)) * norm), outer_region + inner_region + middle_region))).to(device, dtype=torch.float32)
        self.unknown_region = torch.tensor(
            list(map(lambda x: ((x[0]-int(MAT_DIM/2)) * norm, (x[1]-int(MAT_DIM/2)) * norm), unknown_region))).to(device, dtype=torch.float32)
        self.known_region_list = list(
            list(zip(*(outer_region + inner_region + middle_region))))
        self.unknown_region_list = list(list(zip(*(unknown_region))))
        self.ls = torch.empty(len(outer_region)+len(inner_region) +
                              len(middle_region)).to(device, dtype=torch.float32)
        del outer_region, inner_region, unknown_region, all_region, middle_region
        gc.collect()

    def forward(self):
        x = self.known_region
        x2 = self.unknown_region
        if KERNEL  is "NS":
            X = torch.inverse(self.rbf_kernel(x, x))
            X2 = self.rbf_kernel(x, x2)
            self.ls[:self.o_len] = self.le
            self.ls[self.o_len:int(self.o_len + self.i_len)] = self.lc
            self.ls[int(self.o_len + self.i_len):] = self.lm
            a = torch.mean(self.ls).data
            ls2 = a + torch.mm(X2.T, X).mv(self.ls - torch.mean(self.ls))
            self.l_all = self.mk_l_matrix(ls2)
            f_cov = self.nonstationarykernel(self.l_all)
        else:
            f_cov = self.rbf_kernel(self.x, self.x)
        self.f_cov = f_cov
        self.f_cov_inv = torch.inverse(f_cov)
        self.post_cov = torch.inverse(self.R.T.mm(
            self.d_cov_inv).mm(self.R) + self.f_cov_inv)
        self.f_mu = self.post_cov.mv(self.R.T.mm(self.d_cov_inv).mv(self.d))
        self.post_cov_inv = torch.inverse(self.post_cov)
        self.d_mu = self.R.mv(self.f_mu)

    def rbf_kernel(self, x1, x2):
        R1 = x1[:, 0].unsqueeze(1).repeat(1, x2.shape[0])
        Z1 = x1[:, 1].unsqueeze(1).repeat(1, x2.shape[0])
        R2 = x2[:, 0].unsqueeze(0).repeat(x1.shape[0], 1)
        Z2 = x2[:, 1].unsqueeze(0).repeat(x1.shape[0], 1)
        d = (R1 - R2).pow(2) + (Z1 - Z2).pow(2)
        del R1, Z1, R2, Z2
        gc.collect()
        return self.sigma_l.pow(2) * torch.exp(-d/(2.0 * self.l_l.pow(2)))

    def mk_l_matrix(self, ls2):
        scale_length_list = torch.empty(
            (D_DIM, D_DIM)).to(device, dtype=torch.float32)
        scale_length_list[self.known_region_list] = self.ls
        scale_length_list[self.unknown_region_list] = ls2
        return scale_length_list.flatten()

    def nonstationarykernel(self, l_all):
        R1 = self.x[:, 0].unsqueeze(1).repeat(1, self.x.shape[0])
        Z1 = self.x[:, 1].unsqueeze(1).repeat(1, self.x.shape[0])
        R2 = self.x[:, 0].unsqueeze(0).repeat(self.x.shape[0], 1)
        Z2 = self.x[:, 1].unsqueeze(0).repeat(self.x.shape[0], 1)
        d = (R1 - R2).pow(2) + (Z1 - Z2).pow(2)
        del R1, Z1, R2, Z2
        gc.collect()
        l1 = l_all.unsqueeze(1).repeat(1, l_all.shape[0])
        l2 = l_all.unsqueeze(0).repeat(l_all.shape[0], 1)
        covar = 2. / (l1.pow(2) + l2.pow(2))
        return self.sigma_f.pow(2) * torch.abs(l1) * torch.abs(l2) * covar * torch.exp(-d*covar)


# %% Input data
input_name = f"screen_{D_DIM}" # Detector output
val1 = np.load(input_name + ".npy")
val1 = np.where(np.isnan(val1), 0, val1)
mat = np.load(f"mat_{D_DIM}_{MAT_DIM}.npy") # Sight line on poloidal cross section
img = np.load(f"img_{MAT_DIM}.npy")[:, ::-1] # Target for reconstruction
circle = np.load(f"circle_{MAT_DIM}.npy") # for length scale used by NSGPT
model = NNGPT(R=mat, d=val1, circle=circle)
plt.imshow(img,"bwr")
plt.colorbar()

# %% Model setup
train_y_img = torch.tensor(img.flatten())
model = model.to(device, dtype=torch.float32)
train_y = model.d.detach().to(device, dtype=torch.float32)
train_y_img = train_y_img.to(device, dtype=torch.float32)

optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

loss_function2 = RMSELoss()
loss_function = MGLoss()
# %% Train
model.train()
training_iter = 13
prev_loss = 1000000000000
prev_loss_img = 1000000000000
for i in range(training_iter):
    optimizer.zero_grad()
    model()
    if torch.isnan(model.d_mu).any():
        print("BREAK!!")
        break
    loss = loss_function(model)
    loss_img = loss_function2(model.f_mu, train_y_img)
    print(f"loss_img:{loss_img.item():.3f},evidence:{loss.item():.0f}")
    if prev_loss > loss:
        print("updated")
        prev_model = model
        prev_loss = loss
        prev_loss_img = loss_img
    loss.backward(retain_graph=True)
    optimizer.step()
for n, p in prev_model.named_parameters():
    print(n, p)

# %% Evaluate reconstuction
out_f = prev_model.f_mu.detach().cpu().numpy().reshape(*img.shape)
rmse_loss = prev_loss_img.detach().cpu().numpy()
ssim_loss = ssim(out_f, img,
                 data_range=img.max() - img.min())
if KERNEL  == "NS":
    plt.title(
        f"Non-stationary Gaussian Process method \n RMSE:{rmse_loss:.2f}, SSIM: {ssim_loss:.2f}")
else:
    plt.title(
        f"Stationary Gaussian Process method \n RMSE:{rmse_loss:.2f}, SSIM: {ssim_loss:.2f}")
y = np.linspace(-0.45, 0.45, MAT_DIM)
x = np.linspace(1.35, 2.25, MAT_DIM)
X, Y = np.meshgrid(x, y)
plt.ylabel("Height (m)")
plt.xlabel("Major radius (m)")
plt.pcolor(X, Y, out_f, cmap="bwr")
plt.axes().set_aspect('equal')
plt.colorbar(orientation="horizontal", shrink=0.45)
plt.savefig(f"{KERNEL}.png")
# %% Evaluation of cross section 
plt.title(
    f"Sectional view at height 0m ({KERNEL})")
plt.plot(x, out_f[int(MAT_DIM/2), :], label="Estimated")
plt.plot(x, img[int(MAT_DIM/2), :], label="True")
plt.ylabel("Emission")
plt.xlabel("Major radius (m)")
plt.legend()
# %%
s = prev_model.post_cov.detach().cpu().numpy()
cov = np.sqrt(np.diag(s).reshape(*img.shape))
plt.imshow(cov)
plt.colorbar()
plt.savefig("SE_sigma.png")


# %%
