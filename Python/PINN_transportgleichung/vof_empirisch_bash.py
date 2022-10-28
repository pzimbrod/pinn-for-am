import deepxde as dde
import numpy as np
import vis
import csv
import time as tm
import glob

dde.config.set_default_float("float64")

#Domäne
geom = dde.geometry.Rectangle([0.0, 0.0], [5.0, 5.0])
time = dde.geometry.TimeDomain(0.0, 5)
geomtime = dde.geometry.GeometryXTime(geom, time)

#Pde
def pde (x, u): 
    alpha_0, alpha_1  = u[:,0:1], u[:,1:2]

    du_x = dde.grad.jacobian(u, x, j = 0)
    du_y = dde.grad.jacobian(u, x, j = 1)
    du_t = dde.grad.jacobian(u, x, j = 2)

    Transport = (du_t + 1 * (du_x + du_y))
    Bedingung = ((alpha_0 + alpha_1) - 1)

    return [Transport, Bedingung]


def f_ic_a (x):
    return 1.0 * (2.0 <= x[:, 0:1]) * (x[:, 0:1] <= 4.0) * (2.0 <= x[:, 1:2]) * (x[:, 1:2] <=4.0)

def f_ic_b (x):
    return 1 - f_ic_a(x)

ic_a = dde.IC (geomtime, f_ic_a, lambda _, on_initial: on_initial, component = 0)
ic_b = dde.IC (geomtime, f_ic_b, lambda _, on_initial: on_initial, component = 1)

#Boundary Conditions
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

def boundary_r(x, on_boundary): 
    return on_boundary and np.isclose(x[0], 5.0)

def boundary_f(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

def boundary_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 5.0)

bc_a = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component = 0)
bc_b = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component = 1)
bc_a_x0 = dde.PeriodicBC(geomtime, 0, boundary_l, component = 0)
bc_a_x1 = dde.PeriodicBC(geomtime, 0, boundary_r, component = 0)
bc_a_y0 = dde.PeriodicBC(geomtime, 1, boundary_f, component = 0)
bc_a_y1 = dde.PeriodicBC(geomtime, 1, boundary_b, component = 0)
bc_b_x0 = dde.PeriodicBC(geomtime, 0, boundary_l, component = 1)
bc_b_x1 = dde.PeriodicBC(geomtime, 0, boundary_r, component = 1)
bc_b_y0 = dde.PeriodicBC(geomtime, 1, boundary_f, component = 1)
bc_b_y1 = dde.PeriodicBC(geomtime, 1, boundary_b, component = 1)

#, bc_a_x0, bc_a_x1, bc_a_y0, bc_a_y1, bc_b_x0, bc_b_x1, bc_b_y0, bc_b_y1,


lbfsg_weight = 10e10
lw = None #[lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight, lbfsg_weight]
ep_ad = 6000
ep_lb = 24000

#Testparameter
test_train_distribution =   "Sobol" #, "uniform"]
test_num_domain         =   20000
test_num_boundary       =   8000
test_num_initial        =   4000

files = glob.glob("*_empirisch.csv")
i = len(files)

dde.model.optimizers.config.set_LBFGS_options(maxcor=100, ftol= 0, gtol= 0, maxiter=ep_lb, maxfun=ep_lb, maxls=50,)
# ((1.0 * np.finfo(float).eps)**0.5),

header = ('loss 0', 'loss_1', 'Schritte', 'Rechenzeit', 'time')
ergebnis = open('emp_' + str(i) + '__empirisch.csv', 'w')
writer = csv.writer(ergebnis)
writer.writerow(header)

start_time = tm.time()
data = dde.data.TimePDE(geomtime, pde, [bc_a_x0, bc_a_x1, bc_a_y0, bc_a_y1, bc_b_x0, bc_b_x1, bc_b_y0, bc_b_y1, ic_a, ic_b], num_domain=test_num_domain, num_boundary=test_num_boundary, num_initial=test_num_initial, train_distribution=test_train_distribution)
net = dde.nn.FNN([3] + [32] * 3 + [2], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss_weights = lw)
losshistory, trainstate = model.train(epochs=ep_ad)

model.compile("L-BFGS-B", loss_weights= lw)
losshistory, trainstate = model.train(epochs=ep_lb)

time_taken = (tm.time() - start_time)
steps = model.losshistory.steps[-1] 
loss_0, loss_0_matrix, solution = vis.get_mse(vis.get_solution, model, 5, 0)
loss_1, loss_1_matrix, solution = vis.get_mse(vis.get_solution, model, 5, 1)
auswertung = (loss_0, loss_1, steps, time_taken)

writer.writerow(auswertung)
ergebnis.close

cut_x = vis.make_cut(4, 'x', 2, 'final_x_' , model)
np.savetxt('emp_' + str(i) + "_4_s_cut_x.csv", cut_x, delimiter=",")

cut_y = vis.make_cut(4, 'y', 2, 'final_y_' , model)
np.savetxt('emp_' +  str(i) + "_4_s_cut_y.csv", cut_y, delimiter=",")
    
    
print ('done')