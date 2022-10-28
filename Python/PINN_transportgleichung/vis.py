#imports for visualation 
from pyexpat import model
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import re
import glob
import pathlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

#torch.linspace

def make_cut (time, axis, cut_at, save_as, model = model, phase=0): 
    X = np.linspace(0.0,5.0,1000)
    Y = np.linspace(0.0,5.0,1000)
    t1 = np.full(1000, time)
    c1 = np.full(1000, cut_at)
    if axis in ['x', 'X']:
        Xt_1 = np.vstack((X, c1, t1)).reshape(3,1000).T
        cut_ax = 'y'
    elif axis in ['y', 'Y']:
        Xt_1 = np.vstack((c1, Y, t1)).reshape(3,1000).T
        cut_ax = 'x'
    else:    
        print ('ERROR #123: specifie axis as x or y')
        exit()
    phi_pred = model.predict(Xt_1)[:,phase]
    fig, ax = plt.subplots()
    ax.plot(X, phi_pred)
    ax.set(xlabel='x', ylabel='u',
        title=('Configuration at ' + str(cut_ax) + ' = ' + str(cut_at)) + ' and t = ' + str(time))
    ax.grid()
    fig.savefig(str(pathlib.Path(__file__).parent.absolute()) + '/' + 't=' + str(time) + 'at_' + str(axis) + '=' + str(cut_at) + '_' + str(save_as) + '.jpg')
    plt.plot(X,phi_pred)
    
    return phi_pred

def solution_cut(time, axis, cut_at, model = model, phase=0):
    X = np.linspace(0.0,5.0,1000)
    Y = np.linspace(0.0,5.0,1000)
    t1 = np.full(1000, time)
    c1 = np.full(1000, cut_at)
    if axis in ['x', 'X']:
        Xt_1 = np.vstack((X, c1, t1)).reshape(3,1000).T
        cut_ax = 'y'
    elif axis in ['y', 'Y']:
        Xt_1 = np.vstack((c1, Y, t1)).reshape(3,1000).T
        cut_ax = 'x'
    else:    
        print ('ERROR #123: specifie axis as x or y')
        exit()
    phi_pred = model.predict(Xt_1)[:,phase]
    comb = (phi_pred, X)
    
    return phi_pred

def make_image (time=0, name='_', model = model, phase  = 0):    
    x,y = np.meshgrid(np.linspace(0.0,5.0,100),np.linspace(0.0,5.0,100))
    X = np.vstack((np.ravel(x),np.ravel(y))).T
    t_1 = np.full(10000,time).reshape(10000,1)
    Xt_1 = np.hstack((X, t_1))
    phi_pred = model.predict(Xt_1)[:,phase].reshape(100,100)
    fig = plt.figure(figsize=(9, 9))
    ax2 = fig.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(x, y, phi_pred,
                                cmap='gray', linewidth=0, antialiased=False) #cmap=mpl.cm.coolwarm
    ax2.legend()
    ax2.set_title(f'Configuration at $t={(time)}$')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('$\phi$ [-]')
    ax2.set_zlim(0,1)

    fig.colorbar(surf2, shrink=0.5, aspect=5)
    plt.savefig(str(pathlib.Path(__file__).parent.absolute()) + '/at_' + str(time) + 's_' + str(name) + '.jpg')
    
def solution_matrix(length, time, model, phase):
    x,y = np.meshgrid(np.linspace(0.0,5.0,length),np.linspace(0.0,5.0,length))
    X = np.vstack((np.ravel(x),np.ravel(y))).T
    t_1 = np.full(length^2,time).reshape(length^2,1)
    Xt_1 = np.hstack((X, t_1))
    phi_pred = model.predict(Xt_1)[:,phase].reshape(length, length)
    return phi_pred

def plot_ic(ic, name):
    x,y = np.meshgrid(np.linspace(0.0,5.0,100),np.linspace(0.0,5.0,100))
    phi_pred = ic(x, y)
    fig = plt.figure(figsize=(9, 9))
    ax2 = fig.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(x, y, phi_pred,
                                cmap='gray', linewidth=0, antialiased=False) #cmap=mpl.cm.coolwarm
    ax2.legend()
    ax2.set_title(f'Configuration at $t={(0)}$ s')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('$\phi$ [-]')
    ax2.set_zlim(0,1)
    plt.colorbar(surf2, shrink=0.5, aspect=5)
    plt.savefig(str(pathlib.Path(__file__).parent.absolute()) + '/images/' + str(name) + '_ic.jpg')
 
def plot_matrix(matrix, name):
    x,y = np.meshgrid(np.linspace(0.0,5.0,100),np.linspace(0.0,5.0,100))
    phi_pred = matrix
    fig = plt.figure(figsize=(9, 9))
    ax2 = fig.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(x, y, phi_pred,
                                cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
    ax2.legend()
    #ax2.set_title(f'Mean squared Error')
    ax2.set_title(str(name))
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('$\phi$ [-]')
    ax2.set_zlim(0,1)
    ax2.view_init(elev=10.)
    fig.colorbar(surf2, shrink=0.5, aspect=5)
    plt.savefig(str(pathlib.Path(__file__).parent.absolute()) + '/matrix/plot_' + str(name) + 's_ic.jpg')

def plot_2Dmatrix(matrix, name, title = None):
    a, b = np.shape(matrix)
    X,Y = np.linspace(0.0,5.0, a),np.linspace(0.0,5.0, b)
    Z = matrix
    plot = plt.figure(figsize=(9, 9))
    plot = plt.pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-8, vmax=0, shading='gouraud') #cmap='RdBu'
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    ax.set_title(title)
    plt.colorbar(plot, cax=cax)
    plt.savefig(str(pathlib.Path(__file__).parent.absolute()) + '/matrix_2d/' + str(name) + 's_ic.jpg')

def f_ic_a (x,y):
    return 1.0 * (2.0 <= x) * (x <= 3.0) * (3.0 <= y) * (y <=4.0)

def get_solution (x,y):
    return 1.0 * (2.0 <= x) * (x <= 4.0) * (2.0 <= y) * (y <= 4.0)


def get_mse (solution, model, time, phase):
    x,y = np.meshgrid(np.linspace(0.0,5.0,100),np.linspace(0.0,5.0,100))
    if phase == 0:
        u = solution(x,y) #at t = time
    elif phase == 1:
        u = (1 - solution(x,y))
    else:
        print('ERROR: Phase must be 0 or 1')
    X = np.vstack((np.ravel(x),np.ravel(y))).T
    t_1 = np.full(10000,time).reshape(10000,1)
    Xt_1 = np.hstack((X, t_1))
    u_pred = model.predict(Xt_1)[:,phase].reshape(100,100)
    #print (sum(u - u_pred))
    #print (np.max(u - u_pred))
    #print (sum(u_pred))
    loss = (((sum(sum((u - u_pred) ** 2))))/10000) #(len(u)**2)
    loss_matrix = np.log10(((u - u_pred) ** 2))
    u_pred = loss_matrix.mean()
    
    return loss, loss_matrix, u_pred

def get_mse_hd (solution, model, time, phase):
    x,y = np.meshgrid(np.linspace(0.0,5.0,1000),np.linspace(0.0,5.0,1000))
    if phase == 0:
        u = solution(x,y) #at t = time
    elif phase == 1:
        u = (1 - solution(x,y))
    else:
        print('ERROR: Phase must be 0 or 1')
    X = np.vstack((np.ravel(x),np.ravel(y))).T
    t_1 = np.full(1000000,time).reshape(1000000,1)
    Xt_1 = np.hstack((X, t_1))
    u_pred = model.predict(Xt_1)[:,phase].reshape(1000,1000)
    #print (sum(u - u_pred))
    #print (np.max(u - u_pred))
    #print (sum(u_pred))
    loss = (((sum(sum((u - u_pred) ** 2))))/1000000) #(len(u)**2)
    loss_matrix = np.log10(((u - u_pred) ** 2))
    
    return loss, loss_matrix, u_pred

def make_video(time_start, time_end, frames, videoname, model, phase = 0):
    for i in range(frames):

        time = ((time_end - time_start)/frames) * i     
        x,y = np.meshgrid(np.linspace(0.0,5.0,100),np.linspace(0.0,5.0,100))
        X = np.vstack((np.ravel(x),np.ravel(y))).T
        t_1 = np.full(10000,time).reshape(10000,1)
        Xt_1 = np.hstack((X, t_1))
        phi_pred = model.predict(Xt_1)[:,phase].reshape(100,100)
        
        fig = plt.figure(figsize=(9, 9))
        ax2 = fig.add_subplot(111, projection='3d')
        surf2 = ax2.plot_surface(x, y, phi_pred,
                                    cmap=mpl.cm.coolwarm, linewidth=0, antialiased=False)
        ax2.legend()
        ax2.set_title(f'Configuration at $t={(np.round(time, 2))}$')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        ax2.set_zlabel('$\phi$ [-]')
        ax2.set_zlim(0,1)

        fig.colorbar(surf2, shrink=0.5, aspect=5)
        plt.savefig(str(pathlib.Path(__file__).parent.absolute()) + '/video_temp/' + str(i) + '.jpg')


    img_array = []
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    for filename in sorted(glob.glob(str(pathlib.Path(__file__).parent.absolute()) + '/video_temp/' + '*.jpg') , key=numericalSort):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(str(pathlib.Path(__file__).parent.absolute()) + '/videos/' + str(videoname) + '.MP4',cv2.VideoWriter_fourcc(*'FMP4'), 24, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
def video(images_at, save_to, videoname):
    img_array = []
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    for filename in sorted(glob.glob(str(pathlib.Path(__file__).parent.absolute()) + '/' + str(images_at) + '/' + '*.jpg') , key=numericalSort):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(str(pathlib.Path(__file__).parent.absolute()) + '/' + str(save_to) + '/' + str(videoname) + '.MP4',cv2.VideoWriter_fourcc(*'FMP4'), 24, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()