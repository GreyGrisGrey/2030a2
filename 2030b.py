#Implement all the optimizers (make em generalizable, with a learning function variable, rare chance to pass a function as an argument)
#Select a handful of functions (4). visualize the functions and how the optimizers react to them.
#Train neural network (use JAX if you're a sane human being). Get training data. 
#Time series of loss over training is required. And how it approximates it in its final iteration. Use all optimizers.
#Optimizers just gonna do each input individually don't even worry about it
#Paper will be full of graphs and need not be perfect. Sub four pages is acceptable. Use LATEX

# schwefel function, 6 hump camel function, michaelewicz function, sum squares function
# reference : the derivative calculator on the internet
# sum squares is straightforward and just proves everything is working right, everything else is fucking deranged
import math
import numpy as np
import matplotlib.pyplot as plt

def griewank(x, y):
    if x >= 600:
        x = 600
    if x <= -600:
        x = -600
    if y >= 600:
        y = 600
    if y <= -600:
        y = -600
    return x**2/4000 + y**2/4000 - (math.cos(x)) * (math.cos(y/math.sqrt(2))) + 1

def derGriewank(x, y):
    if x >= 600:
        x = 600
    if x <= -600:
        x = -600
    if y >= 600:
        y = 600
    if y <= -600:
        y = -600
    return x/2000 + (math.sin(x)) * (math.cos(y/math.sqrt(2))), y/2000 + ((math.cos(x)) * (math.sin(y/math.sqrt(2))))/math.sqrt(2), 

def six(x, y):
    return (4*(x**2) - 2.1*(x**4) + (x**6)/3) + x*y - 4*(y**2) + 4*(y**4)

def derSix(x, y):
    derX = 8*x - 8.4*(x**3) + 2*(x**5) + y
    derY = x - 8*y + 16*(y**3)
    return derX, derY

def michael(x, y):
    return -((math.sin(x) * math.sin((x**2)/math.pi)**10) + (math.sin(y) * math.sin(2*(y**2)/math.pi)**10))

def derMichael(x, y):
    derX = -((math.cos(x)) * math.sin(math.sin(x**2/math.pi)) + (4*x*math.sin(x)*math.sin(x**2/math.pi)*math.cos(x**2/math.pi))/math.pi)
    derY = -((math.cos(y)) * math.sin(math.sin(2*y**2/math.pi)) + (8*x*math.sin(y)*math.sin(2*y**2/math.pi)*math.cos(2*y**2/math.pi))/math.pi)
    return derX, derY

def sums(x, y):
    return x**2 + 2*y**2

def derSum(x, y):
    return 2*x, 4*y

# Optimizer functions take a list of parameters and output the same list of parameters but changed
# Allows less code repitition down the line
# params = (nu, x, m, t, y, s, m2, s2), not all optimizers use all parameters

def gradient_descent(params, function):
    delX, delY = function(params[1], params[4])
    params[3] += 1
    params[1] -= (params[0] * delX)
    params[4] -= (params[0] * delY)
    return params

# 0.9 is the hyperparameter defining friction. Defining it as a variable each step would be inefficient.
def momentum(params, function):
    delX, delY = function(params[1], params[4])
    params[3] += 1
    params[2] = params[2] * (0.9) - params[0] * delX
    params[6] = params[6] * (0.9) - params[0] * delY
    params[1] += params[2]
    params[4] += params[6]
    return params

def nesterov(params, function):
    delX, delY = function((params[1] + (0.9) * params[2]), params[4] + (0.9) * params[6])
    params[3] += 1
    params[2] = params[2] * (0.9) - params[0] * delX
    params[6] = params[6] * (0.9) - params[0] * delY
    params[1] += params[2]
    params[4] += params[6]
    return params

# Once again, the hyperparameter epsilon is hard coded and not defined as a variable.
def adaGrad(params, function):
    delX, delY = function(params[1], params[4])
    params[3] += 1
    params[5] = params[5] + delX * delX
    params[1] -= (delX * params[0]) / math.sqrt(params[5] + 0.0000000001)
    params[7] = params[7] + delY * delY
    params[4] -= (delY * params[0]) / math.sqrt(params[7] + 0.0000000001)
    return params

def RMSprop(params, function):
    delX, delY = function(params[1], params[4])
    params[3] += 1
    params[5] = ((0.9) * params[5]) + ((0.1) * delX * delX)
    params[7] = ((0.9) * params[7]) + ((0.1) * delY * delY)
    params[1] -= (delX * params[0]) / math.sqrt(params[5] + 0.0000000001)
    params[4] -= (delY * params[0]) / math.sqrt(params[7] + 0.0000000001)
    return params

def adam(params, function):
    delX, delY = function(params[1], params[4])
    params[3] += 1
    params[2] = 0.9 * params[2] + (0.1) * delX
    params[5] = 0.999 * params[5] + 0.001 * delX * delX
    mHat = params[2] / (1 - (0.9)**params[3])
    sHat = params[5] / (1 - (0.999)**params[3])
    params[1] -= ((params[0] * mHat) / (math.sqrt(sHat + 0.0000000001)))
    params[6] = 0.9 * params[6] + (0.1) * delY
    params[7] = 0.999 * params[7] + 0.001 * delY * delY
    mHat = params[6] / (1 - (0.9)**params[3])
    sHat = params[7] / (1 - (0.999)**params[3])
    params[4] -= ((params[0] * mHat) / (math.sqrt(sHat + 0.0000000001)))
    return params

def step(params, function, optimizer, loss):
    delP = loss(params[1], params[4])
    params = optimizer(params, function)
    return params, delP

def optimize(params, function, optimizer, loss):
    var1 = -999
    var2 = 999
    record = [[], []]
    while abs(var1 - var2) > 0.0000000000001 and params[3] < 1000000:
        var2 = var1
        params, var1 = step(params, function, optimizer, loss)
        record[0].append(params[1])
        record[1].append(params[4])
    print(params[3], params[1], params[4], loss(params[1], params[4]))
    return record

optimizers = [gradient_descent, momentum, nesterov, adaGrad, RMSprop, adam]
derivatives = [derMichael, derGriewank, derSix, derSum]
functions = [michael, griewank, six, sums]
labels = ["Gradient Descent", "Momentum", "Nesterov Momentum", "AdaGrad", "RMSprop", "Adam"]
names = ["Michalewicz Function", "Griewank Function", "Six-Hump Camel Function", "Sum Squares Function"]
for i in range(4):
    print(names[i])
    records = []
    for j in optimizers:
        records.append(optimize([0.01, 1, 0, 0, 1, 0, 0, 0], derivatives[i], j, functions[i]))
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    xGrid, yGrid = np.meshgrid(x, y)
    removal_frac = []
    for j in range(100):
        removalWedge = []
        for k in range(100):
            removalWedge.append(functions[i](-3 + (k*0.06), -3 + (j*0.06)))
        removal_frac.append(removalWedge)
    plt.contourf(xGrid, yGrid, removal_frac, levels=25, cmap='plasma')
    plt.colorbar()
    for j in range(len(records)):
        plt.plot(records[j][0], records[j][1], label=labels[j])
    plt.legend()
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.gcf().set_size_inches(6, 4)
    plt.title(names[i])

    plt.show()