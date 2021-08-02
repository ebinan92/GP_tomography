# %%
import gc
import numpy as np
import torch


class BayesianTomography(torch.nn.Module):
    def __init__(
            self,
            R,
            d,
            ls_img,
            device,
            noise_rate,
            kernel,
            norm=1,
            eps=1e-6):
        super(BayesianTomography, self).__init__()
        self.device = device
        self.le = torch.nn.Parameter(torch.tensor(1.))
        self.lc = torch.nn.Parameter(torch.tensor(1.))
        self.lm = torch.nn.Parameter(torch.tensor(1.))
        self.l_se = torch.tensor(1.)
        self.sigma_se = torch.nn.Parameter(torch.tensor(0.3))
        self.sigma_ns = torch.nn.Parameter(torch.tensor(1.))
        self.sigma_noise = torch.nn.Parameter(
            torch.tensor(noise_rate, device=self.device))
        self.R = torch.tensor(R).to(self.device, dtype=torch.float32)
        self.kernel = kernel
        self.f_dim = int(np.sqrt(self.R.shape[1]))

        noise = torch.normal(
            mean=0.0,
            std=torch.abs(
                torch.tensor(d)) *
            noise_rate)

        self.d = (
            torch.tensor(d) +
            noise +
            eps).to(
            self.device,
            dtype=torch.float32)
        self.d_cov = self.sigma_noise * \
            torch.diag(torch.abs(self.d)).to(
                self.device, dtype=torch.float32)
        self.d_cov_inv = torch.inverse(
            self.d_cov).to(
            self.device, dtype=torch.float32)

        all_region = [(i, j) for i in range(self.f_dim)
                      for j in range(self.f_dim)]
        outer_region = list(
            zip(*np.where(((np.abs(ls_img) < 0.2) & (np.abs(ls_img) > 0.17)))))
        inner_region = list(
            zip(*np.where(np.abs(ls_img) > 0.99)))
        middle_region = list(
            zip(*np.where(((np.abs(ls_img) < 0.42) & (np.abs(ls_img) > 0.4)))))

        unknown_region = list(
            set(all_region) - (set(outer_region) | set(inner_region) | set(middle_region)))
        self.o_len = len(outer_region)
        self.i_len = len(inner_region)
        self.m_len = len(middle_region)
        self.x = torch.tensor(list(map(lambda x: ((x[0] - int(self.f_dim / 2)) * norm, (x[1] - int(
            self.f_dim / 2)) * norm), all_region))).to(self.device, dtype=torch.float32)
        self.known_region = torch.tensor(list(map(lambda x: ((x[0] - int(self.f_dim / 2)) * norm, (x[1] - int(
            self.f_dim / 2)) * norm), outer_region + inner_region + middle_region))).to(self.device, dtype=torch.float32)
        self.unknown_region = torch.tensor(list(map(lambda x: ((x[0] - int(self.f_dim / 2)) * norm, (x[1] - int(
            self.f_dim / 2)) * norm), unknown_region))).to(self.device, dtype=torch.float32)
        self.known_region_list = list(
            list(zip(*(outer_region + inner_region + middle_region))))
        self.unknown_region_list = list(list(zip(*(unknown_region))))
        self.ls = torch.empty(
            len(outer_region) +
            len(inner_region) +
            len(middle_region)).to(
            self.device,
            dtype=torch.float32)
        del outer_region, inner_region, unknown_region, all_region, middle_region
        gc.collect()

    def forward(self):
        x = self.known_region
        x2 = self.unknown_region
        if self.kernel == "NS":
            X = torch.inverse(self.rbf_kernel(x, x))
            X2 = self.rbf_kernel(x, x2)
            self.ls[:self.o_len] = self.le
            self.ls[self.o_len:int(self.o_len + self.i_len)] = self.lc
            self.ls[int(self.o_len + self.i_len):] = self.lm
            tmp = torch.mean(self.ls).data
            # tmp = ((self.le + self.lc + self.lm) / 3.).data
            ls2 = tmp + torch.mm(X2.T, X).mv(self.ls - torch.mean(self.ls))
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
        '''
        Squared exponential kernel
        '''
        R1 = x1[:, 0].unsqueeze(1).repeat(1, x2.shape[0])
        Z1 = x1[:, 1].unsqueeze(1).repeat(1, x2.shape[0])
        R2 = x2[:, 0].unsqueeze(0).repeat(x1.shape[0], 1)
        Z2 = x2[:, 1].unsqueeze(0).repeat(x1.shape[0], 1)
        d = (R1 - R2).pow(2) + (Z1 - Z2).pow(2)
        del R1, Z1, R2, Z2
        gc.collect()
        return self.sigma_se.pow(2) * torch.exp(-d / (2.0 * self.l_se.pow(2)))

    def mk_l_matrix(self, ls2):
        scale_length_list = torch.empty(
            (self.f_dim, self.f_dim)).to(self.device, dtype=torch.float32)
        scale_length_list[self.known_region_list] = self.ls
        scale_length_list[self.unknown_region_list] = ls2
        return scale_length_list.flatten()

    def nonstationarykernel(self, l_all):
        '''
        Non stationary kernel
        '''
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
        return self.sigma_ns.pow(2) * torch.abs(l1) * \
            torch.abs(l2) * covar * torch.exp(-d * covar)


class TikhonovTomography(torch.nn.Module):
    def __init__(self, R, d, ls_img, device, noise_rate):
        super(TikhonovTomography, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.f_dim = int(np.sqrt(R.shape[1]))
        lap_base = np.array([-4] + [0] * (self.f_dim * self.f_dim - 1))
        ls_img = ls_img.flatten()
        mask = np.where(((np.abs(ls_img) < 0.30) & (np.abs(ls_img) > 0.2658)))
        c = []
        for i in range(self.f_dim * self.f_dim):
            temp = np.roll(lap_base, i)
            if i % self.f_dim != 0:
                temp[i - 1] = 1
            if i % (self.f_dim - 1) != 0:
                temp[i + 1] = 1
            if i >= self.f_dim:
                temp[i - self.f_dim] = 1
            if self.f_dim * self.f_dim > i + self.f_dim:
                temp[i + self.f_dim] = 1
            temp[mask] = 0
            c.append(temp)
        c_inv = np.linalg.pinv(np.array(c), rcond=1e-6)
        # c_inv = torch.tensor(np.identity(self.f_dim*self.f_dim))
        u, s, vh = np.linalg.svd(np.dot(R, c_inv), full_matrices=False)
        noise = torch.normal(
            mean=0.0,
            std=torch.abs(
                torch.tensor(d)) *
            noise_rate)
        self.u = torch.tensor(u).to(device)
        self.s = torch.tensor(s).to(device)
        self.vh = torch.tensor(vh).to(device)
        self.d = (torch.tensor(d) + noise).to(device)
        self.c_inv = torch.tensor(c_inv).to(device)
        self.R = torch.tensor(R).to(device)

    def forward(self):
        omega = 1. / (1. + self.alpha / self.s.pow(2))
        coef = omega * torch.mv(self.u.t(), self.d) / self.s
        matrix = torch.mm(self.c_inv, self.vh.t())
        f = torch.mv(matrix, coef)
        d = torch.mv(self.R, f)
        omega_sum = torch.sum(omega)
        return f, d, omega_sum


class GCVloss(torch.nn.Module):
    '''
    Generalized cross validation loss
    '''

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y, omega_sum):
        mse = self.mse(yhat, y) + self.eps
        loss = mse / (1 - 1 / y.shape[0] * omega_sum)
        return loss


class RMSELoss(torch.nn.Module):
    '''
    Root mean squared loss
    '''

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MGLoss(torch.nn.Module):
    '''
    - log Evidence loss
    '''

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, model):
        term1 = model.f_mu.unsqueeze(0).mm(model.post_cov_inv).mv(model.f_mu)
        term2 = model.d.unsqueeze(0).mm(model.d_cov_inv).mv(model.d)
        post_det = torch.logdet(model.post_cov)
        f_det = torch.logdet(model.f_cov)
        d_det = torch.logdet(model.d_cov)
        loss = -(post_det - f_det - d_det + term1 - term2)
        return loss