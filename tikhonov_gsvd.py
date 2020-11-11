# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import compare_ssim as ssim

plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
plt.rcParams['font.size'] = 10  # フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

torch.manual_seed(1)
MAT_DIM = 42
D_DIM = 64
NOISE_RATE = 0.125
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCVloss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y, omega_sum):
        mse = self.mse(yhat, y) + self.eps
        loss = mse / (1 - 1/y.shape[0] * omega_sum)
        return loss


class IWAMA(torch.nn.Module):
    def __init__(self, R, g, circle):
        super(IWAMA, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        lap_base = np.array([-4] + [0]*(D_DIM*D_DIM - 1))
        circle = circle.flatten()
        mask = np.where(((np.abs(circle) < 0.4) & (np.abs(circle) > 0.265)))
        c = []
        for i in range(D_DIM*D_DIM):
            temp = np.roll(lap_base, i)
            if i % D_DIM != 0:
                temp[i-1] = 1
            if i % (D_DIM - 1) != 0:
                temp[i+1] = 1
            if i >= D_DIM:
                temp[i-D_DIM] = 1
            if D_DIM*D_DIM > i + D_DIM:
                temp[i+D_DIM] = 1
            temp[mask] = 0
            c.append(temp/D_DIM)
        c_inv = np.linalg.pinv(np.array(c))
        # c_inv = torch.tensor(np.identity(D_DIM*D_DIM))
        u, s, vh = np.linalg.svd(np.dot(R, c_inv), full_matrices=False)
        noise = torch.empty(g.shape[0]).normal_(
            mean=0.0, std=torch.mean(torch.tensor(g))*NOISE_RATE)
        self.u = torch.tensor(u).to(device)
        self.s = torch.tensor(s).to(device)
        self.vh = torch.tensor(vh).to(device)
        self.g = (torch.tensor(g) + noise).to(device)
        self.c_inv = torch.tensor(c_inv).to(device)
        self.R = torch.tensor(R).to(device)

    def forward(self):
        omega = 1. / (1. + self.alpha/self.s.pow(2))
        coef = omega * torch.mv(self.u.t(), self.g) / self.s
        matrix = torch.mm(self.c_inv, self.vh.t())
        f = torch.mv(matrix, coef)
        g = torch.mv(self.R, f)
        omega_sum = torch.sum(omega)
        return f, g, omega_sum


# %%
input_name = f"screen_{MAT_DIM}_{D_DIM}"
val1 = np.load(input_name + ".npy")
val1 = np.where(np.isnan(val1), 0, val1)
mat = np.load(f"mat_{MAT_DIM}_{D_DIM}.npy")
circle = np.load(f"circle_{D_DIM}.npy")
img = np.load(f"img_{D_DIM}.npy")[:, ::-1] # Target for reconstruction

# %%
model = IWAMA(R=mat, g=val1, circle=circle)
model = model.to(device)
train_y = model.g.detach().to(device)

optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.2)

# train_y = torch.tensor(np.dot(mat, val1))
training_iter = 100
loss_function = GCVloss()
prev_loss = torch.tensor(20000.0)
model.train()

for i in range(training_iter):
    optimizer.zero_grad()
    f, output, omega_sum = model()
    loss = loss_function(output, train_y, omega_sum)
    if prev_loss > loss:
        prev_loss = loss
        prev_f = f
        for n, p in model.named_parameters():
            print(n, p)
    loss.backward(retain_graph=True)
    optimizer.step()

for n, p in model.named_parameters():
    print(n, p)
# %%
f = prev_f.detach().cpu().numpy().reshape(D_DIM, D_DIM)
y = np.linspace(-0.45, 0.45, D_DIM)
x = np.linspace(1.35, 2.25, D_DIM)
X, Y = np.meshgrid(x, y)
rmse = np.linalg.norm(f.flatten() - img.flatten(), ord=2)/D_DIM
ssim_loss = ssim(f, img,
                 data_range=img.max() - img.min())
plt.title(
    f"Phillips-Tikhonov regularization method \n RMSE:{rmse:.2f}, SSIM:{ssim_loss:.2f}")
plt.ylabel("Height (m)")
plt.xlabel("Major radius (m)")
plt.pcolor(X, Y, f, cmap="bwr")
plt.axes().set_aspect('equal')
plt.colorbar(orientation="horizontal", shrink=0.45)
plt.savefig("iwama.png")
# %%
