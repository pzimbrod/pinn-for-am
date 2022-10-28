import deepxde as dde
import numpy as np
import vis
import csv
import time as tm
import glob
#from deepxde.backend import pytorch
#import torch

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



lw = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1000, 10000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
ep_ad = 6000
ep_lb = 24000

#Testparameter
test_train_distribution =   "Sobol"
test_num_domain         =   20000
test_num_boundary       =   8000
test_num_initial        =   4000
test_nn_layers          =   3
test_activation         =   "tanh"
dde.model.optimizers.config.set_LBFGS_options(maxcor=100, ftol= 0, gtol= 0, maxiter=ep_lb, maxfun=ep_lb, maxls=50,)

files = glob.glob("*_gewichtung_bash.csv")
i = len(files)

j = i % 2

#CSV
header = ('train distribution', 'domain', 'boundary', 'initial', 'activation', 'layers', 'time taken', 'steps', 'Loss Weights', 'steps_L-BFSG', 'loss 0', 'loss 1')
ergebnis = open(str(i) + '_gewichtung_bash.csv', 'w')
writer = csv.writer(ergebnis)
writer.writerow(header)

start_time = tm.time()
data = dde.data.TimePDE(geomtime, pde, [bc_a_x0, bc_a_x1, bc_a_y0, bc_a_y1, bc_b_x0, bc_b_x1, bc_b_y0, bc_b_y1, ic_a, ic_b], num_domain=test_num_domain, num_boundary=test_num_boundary, num_initial=test_num_initial, train_distribution=test_train_distribution)
net = dde.nn.FNN([3] + [32] * test_nn_layers + [2], test_activation, "Glorot normal")
model = dde.Model(data, net)


model.compile("adam", lr=1e-3, loss_weights = lw[j])
losshistory, trainstate = model.train(epochs=ep_ad)
steps_adam = model.losshistory.steps[-1]


model.compile("L-BFGS-B", loss_weights= lw[j])
losshistory, trainstate = model.train(epochs=ep_lb)
steps_LBFSG = model.losshistory.steps[-1] - steps_adam

time_taken = (tm.time() - start_time)

dde.saveplot(losshistory, trainstate) 

loss_0, loss_0_matrix = vis.get_mse(vis.get_solution, model, 5, 0)
loss_1, loss_1_matrix = vis.get_mse(vis.get_solution, model, 5, 1)
auswertung = (test_train_distribution, test_num_domain, test_num_boundary, test_num_initial, test_activation, test_nn_layers, time_taken, model.losshistory.steps[-1], lw[j], steps_LBFSG,  loss_0, loss_1)
writer.writerow(auswertung)

print ('#i:' + str(i))


#RAR
'''X = geomtime.random_points(100000)
X = np.vstack((geomtime.random_points(100000),geomtime.random_points(100000)))

err = 1
while err > 0.005:
    f = model.predict(X, operator=pde)[0]
    f2 = model.predict(X, operator=pde)[1]
    err_eq = np.absolute(f)
    err_eq2 = np.absolute(f2)
    err = np.mean(err_eq)
    err2 = np.mean(err_eq2)
    print("Mean residual: %.3e" % (err))
    print("Mean residual2: %.3e" % (err2))

    x_id = np.argmax(err_eq)
    x_id2 = np.argmax(err_eq2)
    print("Adding new point:", X[x_id], "\n")
    print("Adding new point_2:", X[x_id2], "\n")
    data.add_anchors(X[x_id])
    data.add_anchors(X[x_id2])
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model.compile("adam", lr=1e-3)
    model.train(epochs=70000, disregard_previous_best=True, callbacks=[early_stopping])
    model.compile("L-BFGS")
    losshistory, train_state = model.train()'''



print ('done')