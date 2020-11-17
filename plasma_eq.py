# %%
import matplotlib
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import sympy
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
from sympy.vector import CoordSys3D
from skimage import measure
MAT_DIM = 64
plt.rcParams['font.family'] = 'sans-serif'  # 使用するフォント
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
plt.rcParams['font.size'] = 9  # フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ


class Tomography:
    def __init__(self, sx, sy, fx, fy):
        self.R = 1.8
        self.a = 0.45
        m = 4
        n = 2
        flag = False
        self.D_DIM = 42
        self.onesteplen = 0.01
        self.screen_center = np.array([sx, sy, 0.])
        self.focus_position = np.array([fx, fy, 0.])
        self.screen_len = 0.12
        self.sightline = self.mkscreen()
        self.projection_angle = self.getprojectionangle()
        self.f_psi, self.fx_psi, self.get_psi = self.psi_eq()
        self.get_emission = self.emission_eq(m, n, flag)

    def mkscreen(self):
        '''Make detector screen'''
        ''' . . . . .-
            . . . . .|
            . . . . .|-- screen_length, resolution(5)
            . . . . .|
            . . . . .-
        '''
        pixel = self.screen_len / self.D_DIM
        screen_pos_list = []
        x, y, z, t, x_c, y_c = symbols("x y z t x_c y_c", real=True)
        center_line_u = (self.focus_position - self.screen_center) / \
            np.linalg.norm(self.focus_position - self.screen_center, ord=2)
        plane_eq = center_line_u[0]*(x - self.screen_center[0]) + center_line_u[1]*(
            y - self.screen_center[1]) + center_line_u[2]*(z - self.screen_center[2])
        hori_eq = sqrt((x-x_c)**2 + (y-y_c)**2) - pixel * t
        vertical_eq1 = sqrt(
            (x-self.screen_center[0])**2 + (y-self.screen_center[1])**2 + (z)**2) - pixel * t
        xv = y*self.screen_center[0] / self.screen_center[1]
        yv_eq, zv_eq = solve(
            [vertical_eq1.subs(x, xv), plane_eq.subs(x, xv)], [y, z])[0]
        xh_eq, yh_eq = solve([hori_eq, plane_eq], [x, y])[0]

        for i in range(-int(self.D_DIM/2), int(self.D_DIM/2)):
            x_ = y*self.screen_center[0] / self.screen_center[1]
            if i == 0:
                x_, y_, z_ = self.screen_center
            else:
                y_ = yv_eq.subs(t, np.abs(i))
                z_ = zv_eq.subs(t, np.abs(i))
                z_ = self.screen_center[2] + np.sign(i) * z_
                x_ = x_.subs(y, y_)
            for j in range(-int(self.D_DIM/2), int(self.D_DIM/2)):
                if j == 0:
                    x_plus = x_
                    y_plus = y_
                else:
                    x_plus = xh_eq.subs(
                        [(x_c, x_), (y_c, y_), (z, z_), (t, np.abs(j))])
                    y_plus = yh_eq.subs(
                        [(x_c, x_), (y_c, y_), (z, z_), (t, np.abs(j))])
                if j < 0:
                    x_minus = x_ - (x_plus - x_)
                    y_minus = y_ - (y_plus - y_)
                    screen_pos_list.append(
                        np.array([float(re(x_minus)), float(re(y_minus)), float(re(z_))]))
                else:
                    screen_pos_list.append(
                        np.array([float(re(x_plus)), float(re(y_plus)), float(re(z_))]))

        screen_pos_list = np.array(screen_pos_list)
        sightline_list = self.focus_position[None, :] - screen_pos_list
        self.screen_pos_list = screen_pos_list
        sightline_list = sightline_list / \
            np.sqrt(np.einsum('ij,ij->i', sightline_list,
                              sightline_list))[:, None]
        return sightline_list

    def screen_list_show(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x", size=14)
        ax.set_ylabel("y", size=14)
        ax.set_zlabel("z", size=14)
        ax.plot(
            self.screen_pos_list[:, 0], self.screen_pos_list[:, 1], self.screen_pos_list[:, 2], '.')

    def psi_eq(self):
        mu = 1.26
        I_p = 0.1
        lam = 0.001
        x, y, psi = symbols('x y psi', real=True)
        gs = mu*I_p*self.R/2*pi*(ln(8*self.R/x) - 2) - ((mu*I_p/4*pi) *
                                                        (ln(x/self.a) + (lam + 0.5)*(1-self.a**2/x**2)))*x*cos(y) - psi
        dgs = diff(gs, x)
        gs_psi = solve(gs, psi)[0]
        get_psi = lambdify((x, y), gs_psi, 'numpy')
        f_psi = lambdify((x, y, psi), gs, 'numpy')
        fx_psi = lambdify((x, y, psi), dgs, 'numpy')
        return f_psi, fx_psi, get_psi

    def emission_eq(self, m, n, flag):
        x, y = symbols('x y', real=True)
        '''tearing parity'''
        l = 0.04
        mu = 0.02
        mm = 0.3
        emission1 = +exp(-l*(abs(x-mm)-mu)**2/(2*abs(x-mm)*mu**2))*cos(m*y)# * abs(sqrt(1/(x-0.3)**3))
        emission2 = -exp(-l*(abs(x-mm)-mu)**2/(2*abs(x-mm)*mu**2))*cos(m*y)# * abs(sqrt(1/(x-0.3)**3))
        expr = Piecewise((emission1, (x > mm)),
                         (emission2, (x <= mm)))
        # expr = exp(-((x-0.)/0.3)**2)
        f_emission = lambdify((x, y), expr, 'numpy')
        return f_emission

    def emission_eq_circle(self, m, n, flag):
        x, y = symbols('x y', real=True)
        expr =exp(-((x-0.3)/0.1)**2) * sign(x-0.2)
        f_emission = lambdify((x, y), expr, 'numpy')
        return f_emission

    def getprojectionangle(self):
        x, y, px, py, sx, sy, ox, oy = sympy.symbols("x y px py sx sy ox oy")
        eq1 = py + (sy-py)/(sx-px)*(x-px) - y
        eq2 = oy - (sx-px)/(sy-py)*(x-ox) - y
        solution = solve([eq1, eq2], [x, y])
        center = np.array([0., 0.])
        qx = float(solution[x].subs([(px, self.screen_center[0]), (py, self.screen_center[1]), (
            sx, self.focus_position[0]), (sy, self.focus_position[1]), (ox, center[0]), (oy, center[1])]))
        qy = float(solution[y].subs([(px, self.screen_center[0]), (py, self.screen_center[1]), (
            sx, self.focus_position[0]), (sy, self.focus_position[1]), (ox, center[0]), (oy, center[1])]))
        theta_cross = self.arctan(qy, qx)
        return theta_cross

    def safety_factor(self, psi):
        return 1.0 + 0.6/psi**3

    def gs_bisection(self, psi, omega):
        r1 = self.a*1.5
        r0 = 0.02
        r0 = np.full(psi.shape[0], r0)
        r1 = np.full(psi.shape[0], r1)
        for _ in range(60):
            x0 = self.f_psi(r0, omega, psi)
            x1 = self.f_psi(r1, omega, psi)
            x01 = self.f_psi((r1+r0)/2, omega, psi)
            r0 = np.where((np.sign(x0) == np.sign(x01)), (r1+r0)/2, r0)
            r1 = np.where((np.sign(x1) == np.sign(x01)), (r1+r0)/2, r1)
            if np.mean(r1-r0) < 1*10**(-5):
                break
        return np.where(((r1-r0) > 10**(-4)) | (np.isnan(r1+r0)), 0.02, (r1+r0)/2)

    def get_omega(self, psi, omega, angle_diff):
        omega_diff = angle_diff/self.safety_factor(psi)
        omega_diff = omega_diff % (2*np.pi)
        return omega_diff + omega

    def sight_step(self, i):
        # cylindrical coordinate
        sight_pos = self.focus_position + \
            self.sightline * (i * self.onesteplen)
        angle_diff = -self.arctan(
            sight_pos[:, 1], sight_pos[:, 0]) + self.projection_angle
        r = np.sqrt(sight_pos[:, 0]**2+sight_pos[:, 1]**2)

        # tokamak coordinate
        omega = self.arctan(sight_pos[:, 2], r-self.R)
        rho_old = np.sqrt((r-self.R)**2 + sight_pos[:, 2]**2)
        mask = np.where((rho_old < self.a), True, False)
        if np.count_nonzero(mask) != 0:
            psi = self.get_psi(rho_old[mask], omega[mask])
            new_omega = self.get_omega(psi, omega[mask], angle_diff[mask])
            new_rho = self.gs_bisection(psi, new_omega)
            psi = self.get_emission(new_rho, new_omega)
            self.mask = mask
            return new_omega, new_rho,
        else:
            return False, False

    def sight_step_sum(self, i):
        # cylindrical coordinate
        sight_pos = self.focus_position + \
            self.sightline * (i * self.onesteplen)
        angle_diff = -self.arctan(
            sight_pos[:, 1], sight_pos[:, 0]) + self.projection_angle

        r = np.sqrt(sight_pos[:, 0]**2+sight_pos[:, 1]**2)
        # tokamak coordinate
        omega = self.arctan(sight_pos[:, 2], r-self.R)
        rho_old = np.sqrt((r-self.R)**2 + sight_pos[:, 2]**2)
        mask = np.where(rho_old < self.a, True, False)

        if np.count_nonzero(mask) != 0:
            psi = self.get_psi(rho_old[mask], omega[mask])
            new_omega = self.get_omega(psi, omega[mask], angle_diff[mask])
            new_rho = self.gs_bisection(psi, new_omega)
            psi = self.get_emission(new_rho, new_omega)
            self.mask = mask
            return psi
        else:
            return False

    def show_psi(self):
        rho = np.linspace(0, self.a, 120)
        omega = np.linspace(0, 2*np.pi, 120)
        Rho, Omega = np.meshgrid(rho, omega)
        X = self.R + Rho*np.cos(Omega)
        Z = Rho*np.sin(Omega)
        plt.figure()
        Psi = self.get_psi(Rho, Omega)
        temp = Psi[np.where(np.isnan(Psi) == False)]
        print(np.max(temp), np.min(temp))
        plt.pcolormesh(X, Z, Psi)
        plt.ylabel("Height (m)")
        plt.xlabel("Major radius (m)")
        cont = plt.contour(X, Z, Psi, 40, colors="red")
        cont.clabel(fmt='%1.1f', fontsize=14)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("psi.png")
        plt.show()

    def save_emission(self):
        x = np.linspace(-to.a, to.a, MAT_DIM)
        y = np.linspace(-to.a, to.a, MAT_DIM)
        X, Y = np.meshgrid(x, y)
        Rho = np.sqrt(X**2 + Y**2)
        Omega = np.arctan2(Y, X)
        get_emission = self.emission_eq_circle(0, 0, True)
        Psi = get_emission(Rho, Omega)
        Psi = self.minmax_normalize(Psi)
        np.save(f"circle_{MAT_DIM}", Psi)
        x = np.linspace(-to.a, to.a, MAT_DIM)
        y = np.linspace(-to.a, to.a, MAT_DIM)
        X, Y = np.meshgrid(x, y)
        Rho = np.sqrt(X**2 + Y**2)
        Omega = np.arctan2(Y, X)
        Psi2 = self.get_emission(Rho, Omega)
        plt.imshow(Psi2, cmap="bwr")
        plt.colorbar()
        np.save(f"img_{MAT_DIM}", Psi2)

    def show_q(self):
        x = np.linspace(0, self.a, 100)
        theta = np.zeros(100)
        psi = self.get_psi(x, theta)
        plt.figure()
        plt.plot(x / self.a, self.safety_factor(psi))
        plt.ylim((0,6))
        plt.xlabel("normalized poloidal flux")
        plt.ylabel(r"safety factor $q$")
        plt.savefig("q_profile.png")
        plt.show()

    def arctan(self, y, x):
        out = np.arctan2(y, x)
        return out

    def minmax_normalize(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))


# %% 
'''screen_center(1.8, 2.0), focus position(1.8, 1.8)'''
to = Tomography(1.8, 2.0, 1.8, 1.8)
to.save_emission()
# %%
# to.show_q()
# to.show_psi()
# to.screen_list_show()
# %% 
'''Get detector image'''
screen = np.zeros(to.D_DIM**2)
flag = np.full(to.D_DIM**2, False)
rmlist = np.full(to.D_DIM**2, False)
temp = np.full(to.D_DIM**2, 0.)
for i in range(0, math.ceil(to.R*2.5/to.onesteplen)):
    psi = to.sight_step_sum(i)
    if np.any(psi) != False:
        # print(f"ON:{i}")
        rmlist[np.where((flag == True) & (to.mask == False))] = True
        flag[to.mask] = True
        temp[to.mask] = psi
        screen[np.where((rmlist == False) & (to.mask == True))
               ] += temp[np.where((rmlist == False) & (to.mask == True))]
# %% 
'''Save detector image'''
x = to.screen_pos_list[:, 0].reshape(to.D_DIM, to.D_DIM)
z = to.screen_pos_list[:, 2].reshape(to.D_DIM, to.D_DIM)
plt.pcolormesh(x, z, screen.reshape(to.D_DIM, to.D_DIM)[:, ::-1], cmap='bwr')
plt.colorbar()
np.save(f"screen_{to.D_DIM}", screen)
# %% 
'''Get sight line'''
r_list = []
z_list = []
r_plot_list = []
z_plot_list = []
mask_list = []
flag = np.full(to.D_DIM**2, False)
rmlist = np.full(to.D_DIM**2, False)
omega_temp = np.full(to.D_DIM**2, 0.0)
rho_temp = np.full(to.D_DIM**2, 0.0)
for i in range(10, math.ceil(to.R*2.5/to.onesteplen)):
    new_omega, new_rho = to.sight_step(i)
    if np.any(new_omega == False) == False:
        rmlist[np.where((flag == True) & (to.mask == False))] = True
        flag[to.mask] = True
        omega_temp[to.mask] = new_omega
        rho_temp[to.mask] = new_rho
        new_omega = omega_temp[np.where((rmlist == False) & (to.mask == True))]
        new_rho = rho_temp[np.where((rmlist == False) & (to.mask == True))]
        r = np.sign(np.cos(to.projection_angle)) * to.R - \
            np.sign(np.cos(to.projection_angle))*new_rho*np.cos(new_omega)
        z = new_rho*np.sin(new_omega)
        r_list.append(r)
        z_list.append(z)
        r_plot_list.append(r)
        z_plot_list.append(z)
        to.mask[rmlist] = False
        mask_list.append(to.mask)
r_list = np.array(r_list).T
z_list = np.array(z_list).T
# %% 
'''Show sight line'''
plt.figure()
plot_list_r = []
plot_list_z = []
for r, z, mask in zip(r_plot_list, z_plot_list, mask_list):
    r_p = np.full(to.D_DIM**2, to.R)
    z_p = np.zeros(to.D_DIM**2)
    if r.shape != r_p[mask].shape:
        continue
    r_p[mask] = r
    z_p[mask] = z
    plot_list_r.append(r_p)
    plot_list_z.append(z_p)
plot_list_r = np.array(plot_list_r)
plot_list_z = np.array(plot_list_z)
print(plot_list_r.shape, plot_list_z.shape)
for i in range(plot_list_r.shape[1]):
    if i % 50 == 0:
        out_r = plot_list_r[:, i][np.where(plot_list_r[:, i] != 1.8)]
        out_z = plot_list_z[:, i][np.where(plot_list_z[:, i] != 0.)]
        plt.plot(out_r, out_z, '.')


# %%
'''Transfer sight line to poloidal cross section'''
r = np.linspace(np.sign(np.cos(to.projection_angle))*to.R - to.a,
                np.sign(np.cos(to.projection_angle))*to.R + to.a, MAT_DIM+2)
z = np.linspace(-to.a, to.a, MAT_DIM+2)
projection_mat = np.zeros((to.D_DIM**2, MAT_DIM, MAT_DIM))
rr, zz = np.meshgrid(r, z)
A = np.dstack([rr, zz]).reshape(-1, 2)
B = np.dstack([np.roll(rr, 1, axis=1), zz]).reshape(-1, 2)
C = np.dstack([np.roll(rr, 1, axis=1), np.roll(zz, -1, axis=0)]).reshape(-1, 2)
D = np.dstack([rr, np.roll(zz, -1, axis=0)]).reshape(-1, 2)
view_mat = np.zeros((MAT_DIM, MAT_DIM))
for r, z, mask in zip(r_list, z_list, mask_list):
    if r.shape[0] == 0:
        continue
    pos = np.array(list(zip(r, z)))
    Atopos = pos[:, None, :] - A[None, :, :]
    Btopos = pos[:, None, :] - B[None, :, :]
    Ctopos = pos[:, None, :] - C[None, :, :]
    Dtopos = pos[:, None, :] - D[None, :, :]
    C1 = np.sign(np.cross(Atopos, (B-A)))
    C2 = np.sign(np.cross(Btopos, (D-B)))
    C3 = np.sign(np.cross(Dtopos, (A-D)))
    tri1_flag = np.where((C1 == C2) & (C1 == C3), True, False)
    C1 = np.sign(np.cross(Atopos, (C-B)))
    C2 = np.sign(np.cross(Btopos, (D-C)))
    C3 = np.sign(np.cross(Dtopos, (B-D)))
    tri2_flag = np.where((C1 == C2) & (C1 == C3), True, False)
    G = np.where((tri1_flag == True) | (tri2_flag == True), 1,
                 0).reshape(pos.shape[0], MAT_DIM+2, MAT_DIM+2)
    projection_mat[mask, :, :] += G[:, 1:-1, 1:-1]
    view_mat += np.sum(G, axis=0)[1:-1, 1:-1]
# %%
projection_mat = projection_mat.reshape(to.D_DIM**2, MAT_DIM**2)
np.save(f"mat_{to.D_DIM}_{MAT_DIM}", projection_mat)
plt.imshow(view_mat)
plt.colorbar()
plt.savefig("viewmat.png")
# %%