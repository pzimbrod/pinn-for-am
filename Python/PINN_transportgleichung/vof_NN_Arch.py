import deepxde as dde
import numpy as np
import vis
import csv
import time as tm
import pathlib
from deepxde.utils import external

#Domäne
geom = dde.geometry.Rectangle([0.0, 0.0], [5.0, 5.0])
time = dde.geometry.TimeDomain(0.0, 4)
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
    return 1.0 * (2.0 <= x[:, 0:1]) * (x[:, 0:1] <= 3.0) * (3.0 <= x[:, 1:2]) * (x[:, 1:2] <=4.0)

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

data = dde.data.TimePDE(geomtime, pde, [bc_a_x0, bc_a_x1, bc_a_y0, bc_a_y1, bc_b_x0, bc_b_x1, bc_b_y0, bc_b_y1, ic_a, ic_b], num_domain=40000, num_boundary=4000, num_initial=8000, train_distribution="Sobol")

lw = None #[10, 10, 1, 1, 10, 10]
ep_ad = 10000
ep_lb = 12000
videoname = 'vof_Periodic_sobol_net_50_5_'+ str(int(ep_ad/1000)) + 'k_' + str(int(ep_lb/1000)) + 'k_'+ str(lw)

#CSV
header = ('time taken', 'steps', 'pde_loss', 'pde_loss', 'bc_loss', 'bc_loss', 'bc_loss', 'bc_loss', 'bc_loss', 'bc_loss', 'bc_loss', 'bc_loss', 'ic_loss', 'ic_loss')
ergebnis = open('Ergebnisse_nn_arch_50_5.csv', 'w')
writer = csv.writer(ergebnis)
writer.writerow(header)

for i in range (3):
    start_time = tm.time()
    net = dde.nn.FNN([3] + [30] * 3 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights = lw)
    losshistory, trainstate = model.train(epochs=ep_ad)

    dde.model.optimizers.config.set_LBFGS_options(
        maxcor=100,
        ftol=0, #1.0 * np.finfo(float).eps,
        gtol=1e-08,
        maxiter=ep_lb,
        maxfun=10000, #None,
        maxls=50,
    )

    model.compile("L-BFGS-B", loss_weights= lw)
    losshistory, trainstate = model.train(epochs=ep_lb)
    time_taken = (tm.time() - start_time)
    
    dde.saveplot(losshistory, trainstate) 
    external.plot_loss_history(losshistory, 'loss_test' + str(i))

    loss = model.losshistory.loss_train[-1]
    auswertung = (time_taken, model.losshistory.steps[-1], loss[0], loss[1], loss[2], loss[3], loss[4], loss[5], loss[6], loss[7], loss[8], loss[9], loss[10], loss[11])
    writer.writerow(auswertung)

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


    #vis.make_video(0,4,40, ('Ergebnisse_nn_arch_50_5' + str(i)), model, 0)

print ('done')