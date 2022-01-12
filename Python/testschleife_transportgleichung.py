import deepxde as dde
import numpy as np 
import csv

#Domäne
geom = dde.geometry.Interval(0, 5)
time = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(geom, time)

#Pde
def pde (x, y): 
    dy_x = dde.grad.jacobian(y, x, i = 0, j = 0)
    dy_t = dde.grad.jacobian(y, x, i = 0, j = 1)
    return dy_t + 1 * dy_x

#bc: Neumann bc mit dy_x = 0 
bc = dde.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

#ic
ic = dde.IC(geomtime, lambda x: 1 * (1 <= x[:, 0:1]) * (x[:, 0:1] <=2), lambda _, on_initial: on_initial)

#Testparameter
test_train_distribution =   ["Sobol","uniform"]
test_num_domain         =   [2000, 4000, 6000]
test_nn_layers          =   [2, 3, 4]
test_activation         =   ["tanh", "relu","sigmoid"]

#CSV
header = ('train_destribution', 'num_domain', 'net_layers', 'activation function', 'Optimierer', 'Learning rate', 'Best model at step:', 'Loss')
ergebnis = open('Ergebnisse_TransportGleichung.csv', 'w')
writer = csv.writer(ergebnis)
writer.writerow(header)

#Testschleife
for t in range(len(test_train_distribution)):
    for j in range(len(test_num_domain)):
        for n in range (len(test_nn_layers)):
            for a in range (len(test_activation)):

                data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=test_num_domain[j], num_boundary=160, num_initial = 200, train_distribution=test_train_distribution[t])

                #NN
                net = dde.maps.FNN ([2] + [20] * test_nn_layers[n] + [1], test_activation[a], "Glorot normal")

                model = dde.Model(data, net)
                model.compile("adam", lr=1e-3)

                losshistory = trainstate = model.train(epochs=8000)

                model.compile("L-BFGS-B")

                losshistory, trainstate = model.train()

                auswertung = (test_train_distribution[t], test_num_domain[j], net.layer_size, net.activation, model.opt_name, 'lr', model.losshistory.steps[-1], np.sum(model.losshistory.loss_train[-1]))
                writer.writerow(auswertung)

ergebnis.close
print('done')