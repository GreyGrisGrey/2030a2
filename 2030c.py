import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
import math

# Hidden units number and hidden layers number were controlled directly here.
def MLP(x, y, hidden_units=[64, 64], activation=jax.nn.tanh):

  # Get shape of input and output function
  xin = x.shape[-1]
  yin = y.shape[-1]

  # Add input and output layer sizes to layer stack
  units = [xin] + hidden_units + [yin]

  # Initialize weights and biases (using the so-called Xavier initialization)
  def init(key=jax.random.PRNGKey(1)):
    keys = jax.random.split(key, len(units))
    params = []
    additionalsM = []
    additionalsS = []
    additionalsHat = []
    for i, (u_in, u_out) in enumerate(zip(units[:-1], units[1:])):
      w_key, b_key = jax.random.split(keys[i])
      scale = jnp.sqrt(2/(u_in+u_out))
      params.append([scale*jax.random.normal(w_key, (u_in, u_out)), jnp.zeros((u_out,))])
      #The additional matrices are used to keep track of values of m and s later, as well as serving to hold other values for later use
      additionalsM.append([jnp.zeros((u_out,)), jnp.zeros((u_out,))])
      additionalsS.append([jnp.zeros((u_out,)), jnp.zeros((u_out,))])
      additionalsHat.append([jnp.zeros((u_out,)), jnp.zeros((u_out,))])
    return params, additionalsM, additionalsS, additionalsHat

  # Apply the MLP
  @jax.jit
  def apply(params, x):
    for w, b in params[:-1]:
      x = activation(jnp.dot(x, w) + b)
    return jnp.dot(x, params[-1][0]) + params[-1][1]

  return init, apply

@jax.jit
def SGD(params, inputs, outputs, addM, addS, addHat, epoch, lr=0.01):
  grads = jax.grad(loss)(params, inputs, outputs)
  params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
  return params, addM, addS

def momentum(params, inputs, outputs, addM, addS, addHat, epoch, lr=0.01):
  grads = jax.grad(loss)(params, inputs, outputs)
  addM = jax.tree_util.tree_map(lambda p, g: 0.9 * p - lr * g, addM, grads)
  params = jax.tree_util.tree_map(lambda p, g: p + g, params, addM)
  return params, addM, addS

def nesterov(params, inputs, outputs, addM, addS, addHat, epoch, lr=0.01):
  addHat = jax.tree_util.tree_map(lambda p, g: p + 0.9 * g, params, addM)
  grads = jax.grad(loss)(addHat, inputs, outputs)
  addM = jax.tree_util.tree_map(lambda p, g: 0.9 * p - lr * g, addM, grads)
  params = jax.tree_util.tree_map(lambda p, g: p + g, params, addM)
  return params, addM, addS

def adagrad(params, inputs, outputs, addM, addS, addHat, epoch, lr=0.01):
  grads = jax.grad(loss)(params, inputs, outputs)
  addS = jax.tree_util.tree_map(lambda p, g: p + g * g, addS, grads)
  grads = jax.tree_util.tree_map(lambda p, g: lr * g / jax.numpy.sqrt(p + 0.0000000001), addS, grads)
  params = jax.tree_util.tree_map(lambda p, g: p - g, params, grads)
  return params, addM, addS

def RMS(params, inputs, outputs, addM, addS, addHat, epoch, lr=0.01):
  grads = jax.grad(loss)(params, inputs, outputs)
  addS = jax.tree_util.tree_map(lambda p, g: 0.9 *p + 0.1 * g * g, addS, grads)
  grads = jax.tree_util.tree_map(lambda p, g: lr * g / jax.numpy.sqrt(p + 0.0000000001), addS, grads)
  params = jax.tree_util.tree_map(lambda p, g: p - g, params, grads)
  return params, addM, addS

#More complicated than it probably should be but it appears to work correctly
def adam(params, inputs, outputs, addM, addS, addHat, epoch, lr=0.01):
    grads = jax.grad(loss)(params, inputs, outputs)
    #Step 1
    addM = jax.tree_util.tree_map(lambda p, g: 0.9 * p + 0.1 * g, addM, grads)
    #Step 2
    addS = jax.tree_util.tree_map(lambda p, g: 0.999 * p + 0.001 * g * g, addS, grads)
    #Step 3 (get mHat)
    grads = jax.tree_util.tree_map(lambda p, g: lr * (g / (1 - 0.9**epoch)), grads, addM)
    #Step 4 (get sHat)
    addHat = jax.tree_util.tree_map(lambda p, g: jax.numpy.sqrt((g / (1 - 0.999**epoch)) + 0.0000000001), addHat, addS)
    #Step 5.1 (calculate right side of equation)
    grads = jax.tree_util.tree_map(lambda p, g: p / g, grads, addHat)
    #Step 5.2 (calculate full equation)
    params = jax.tree_util.tree_map(lambda p, g: p - g, params, grads)
    return params, addM, addS

# MSE loss function
@jax.jit
def loss(params, x, y):
  yout = apply(params, x)
  return jnp.mean((yout - y)**2)

def train(params, x, y, epochs, addM, addS, addHat, method):

  epoch_loss = np.zeros(epochs)
  step = 0

  for i in range(epochs):
    params, addM, addS = method(params, x, y, addM, addS, addHat, i+1)

    epoch_loss[i] = loss(params, x, y)
    if i%1000 == 0:
      print(i)

  return params, epoch_loss

labels = ["Gradient Descent", "Momentum", "Nesterov Momentum", "AdaGrad", "RMSprop", "Adam"]
optimizers = [SGD, momentum, nesterov, adagrad, RMS, adam]
n = 128
x = jnp.linspace(0, 1, n).reshape((-1,1))
y = (1-x)*jnp.sin(2*x) + (1-x)**2*jnp.sin(10*x)
plt.plot(x, y, '.', label='Ground Truth')
for i in range(6):
    # Initialize the parameters of the neural network
    init, apply = MLP(x, y)
    params, addM, addS, addHat = init()
    epochs = 10000
    params_sgd, loss_sgd = train(params, x, y, epochs, addM, addS, addHat, optimizers[i])
    y_sgd = apply(params_sgd, x)
    plt.plot(x, y_sgd, label=labels[i] + f': {loss(params_sgd, x, y):.3g}')

plt.title(f'Numerical results')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
