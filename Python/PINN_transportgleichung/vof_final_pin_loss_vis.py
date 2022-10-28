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

bc_a_x0 = dde.PeriodicBC(geomtime, 0, boundary_l, component = 0)
bc_a_x1 = dde.PeriodicBC(geomtime, 0, boundary_r, component = 0)
bc_a_y0 = dde.PeriodicBC(geomtime, 1, boundary_f, component = 0)
bc_a_y1 = dde.PeriodicBC(geomtime, 1, boundary_b, component = 0)
bc_b_x0 = dde.PeriodicBC(geomtime, 0, boundary_l, component = 1)
bc_b_x1 = dde.PeriodicBC(geomtime, 0, boundary_r, component = 1)
bc_b_y0 = dde.PeriodicBC(geomtime, 1, boundary_f, component = 1)
bc_b_y1 = dde.PeriodicBC(geomtime, 1, boundary_b, component = 1)

lw = [1000, 1000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ep_ad = 6000
ep_lb = 24000

#Testparameter
test_train_distribution =   "Sobol"
test_num_domain         =   40000
test_num_boundary       =   16000
test_num_initial        =   8000
test_nn_neuronen        =   32
test_activation         =   "tanh"
dde.model.optimizers.config.set_LBFGS_options(maxcor=100, ftol= 0, gtol= 0, maxiter=100, maxfun=100, maxls=50,)

files = glob.glob("*_final_bash.csv")
i = len(files)

j = i % 2

#CSV
header = ('time taken', 'steps', 'steps_adam', 'steps_L-BFSG', 'loss 0', 'loss 1')
ergebnis = open(str(i) + '_final_bash.csv', 'w')
writer = csv.writer(ergebnis)
writer.writerow(header)

start_time = tm.time()
data = dde.data.TimePDE(geomtime, pde, [bc_a_x0, bc_a_x1, bc_a_y0, bc_a_y1, bc_b_x0, bc_b_x1, bc_b_y0, bc_b_y1, ic_a, ic_b], num_domain=test_num_domain, num_boundary=test_num_boundary, num_initial=test_num_initial, train_distribution=test_train_distribution)
net = dde.nn.FNN([3] + [32] * 3 + [2], test_activation, "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss_weights = lw)

steps = 0
while steps < ep_ad:
    losshistory, trainstate = model.train(epochs=100)
    steps_adam = model.losshistory.steps[-1]
    loss, loss_mx, s = vis.get_mse(vis.get_solution, model, 5, 0)
    steps = model.losshistory.steps[-1]
    vis.plot_2Dmatrix(loss_mx, str(round((model.losshistory.steps[-1])/100)) , 'step: ' + str(model.losshistory.steps[-1]) + ', log(loss): ' + str(round(np.log10(loss), 3)))
    #vis.plot_matrix(loss_mx, model.losshistory.steps[-1])

model.compile("L-BFGS-B", loss_weights= lw)

while model.losshistory.steps[-1] - ep_ad < ep_lb:
    losshistory, trainstate = model.train(epochs=ep_lb)
    steps_LBFSG = model.losshistory.steps[-1] - steps_adam
    loss, loss_mx, s = vis.get_mse(vis.get_solution, model, 5, 0)
    vis.plot_2Dmatrix(loss_mx, str(round((model.losshistory.steps[-1])/100)) , 'step: ' + str(model.losshistory.steps[-1]) + ', log(loss): ' + str(round(np.log10(loss), 3)))
    #vis.plot_matrix(loss_mx, model.losshistory.steps[-1])

time_taken = (tm.time() - start_time)

dde.saveplot(losshistory, trainstate) 

loss_0, loss_0_matrix, solution = vis.get_mse(vis.get_solution, model, 5, 0)
loss_1, loss_1_matrix, solution = vis.get_mse(vis.get_solution, model, 5, 1)

print('--------------')
print(' ')
print('Loss: ' + str(loss_0))
print(' ')
print('--------------')

#make video
vis.video('matrix_2d', 'videos_loss', str(i))

auswertung = (time_taken, model.losshistory.steps[-1], steps_adam, steps_LBFSG, loss_0, loss_1)
writer.writerow(auswertung)




print ('done')