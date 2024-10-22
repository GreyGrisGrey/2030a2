import math

def L(x, a):
    return math.sin(2*x) + a * math.sin(4*x)

def derL(x, a):
    return 2 * math.cos(2*x) + a * 4 * math.cos(4*x)

# Optimizer functions take a list of parameters and output the same list of parameters but changed
# Allows less code repitition down the line
# params = (nu, p, m, t, a, s), not all optimizers use all parameters

def gradient_descent(params, function):
    delP = function(params[1], params[4])
    params[3] += 1
    params[1] -= (params[0] * delP)
    return params

# 0.9 is the hyperparameter defining friction. Defining it as a variable each step would be inefficient.
def momentum(params, function):
    delP = function(params[1], params[4])
    params[3] += 1
    params[2] = params[2] * (0.9) - params[0] * delP
    params[1] += params[2]
    return params

def nesterov(params, function):
    delP = function((params[1] + (0.9) * params[2]), params[4])
    params[3] += 1
    params[2] = params[2] * (0.9) - params[0] * delP
    params[1] += params[2]
    return params

# Once again, the hyperparameter epsilon is hard coded and not defined as a variable.
def adaGrad(params, function):
    delP = function(params[1], params[4])
    params[3] += 1
    params[5] = params[5] + delP * delP
    params[1] -= (delP * params[0]) / math.sqrt(params[5] + 0.0000000001)
    return params

def RMSprop(params, function):
    delP = function(params[1], params[4])
    params[3] += 1
    params[5] = ((0.9) * params[5]) + ((0.1) * delP * delP)
    params[1] -= (delP * params[0]) / math.sqrt(params[5] + 0.0000000001)
    return params

def adam(params, function):
    delP = function(params[1], params[4])
    params[3] += 1
    params[2] = 0.9 * params[2] + (0.1) * delP
    params[5] = 0.999 * params[5] + 0.001 * delP * delP
    mHat = params[2] / (1 - (0.9)**params[3])
    sHat = params[5] / (1 - (0.999)**params[3])
    params[1] -= ((params[0] * mHat) / (math.sqrt(sHat + 0.0000000001)))
    return params

def step(params, function, optimizer, loss):
    delP = loss(params[1], params[4])
    params = optimizer(params, function)
    return params, delP

def optimize(params, function, optimizer, loss):
    var1 = -999
    var2 = 999
    while abs(var1 - var2) > 0.0000000000001 and params[3] < 10000000:
        var2 = var1
        params, var1 = step(params, function, optimizer, loss)
    print(params[3], params[1], loss(params[1], params[4]), params[5])

optimizers = [gradient_descent, momentum, nesterov, adaGrad, RMSprop, adam]
times = [0.1, 0.01, 0.001]
for j in times:
    print("learning rate:", j)
    for i in optimizers:
        optimize([j, 0.75, 0, 0, 0.499, 0], derL, i, L)