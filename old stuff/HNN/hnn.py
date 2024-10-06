# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
# reducted by Denis Andric' | 2024

import torch
import numpy as np

import os, torch, pickle, zipfile
import imageio, shutil
import scipy, scipy.misc, scipy.integrate
import device_util


def integrate_model(model, t_span, y0, fun=None, **kwargs):
  def default_fun(t, np_x):
      x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
      x = x.view(1, np.size(np_x)) # batch size of 1
      dx = model.time_derivative(x).data.numpy().reshape(-1)
      return dx
  fun = default_fun if fun is None else fun
  return scipy.integrate.solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def rk4(fun, y0, t, dt, *args, **kwargs):
  dt2 = dt / 2.0
  k1 = fun(y0, t, *args, **kwargs)
  k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
  k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
  k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy


def L2_loss(u, v):
  return (u-v).pow(2).mean()


def read_lipson(experiment_name, save_dir):
  desired_file = experiment_name + ".txt"
  with zipfile.ZipFile('{}/invar_datasets.zip'.format(save_dir)) as z:
    for filename in z.namelist():
      if desired_file == filename and not os.path.isdir(filename):
        with z.open(filename) as f:
            data = f.read()
  return str(data)


def str2array(string):
  lines = string.split('\\n')
  names = lines[0].strip("b'% \\r").split(' ')
  dnames = ['d' + n for n in names]
  names = ['trial', 't'] + names + dnames
  data = [[float(s) for s in l.strip("' \\r,").split( )] for l in lines[1:-1]]

  return np.asarray(data), names


def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing
# Denis Andric
def function_act(name):
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "sin":
        return Sin()
    else:
        return torch.nn.Identity()
  #Denis Andric  
class Sin(torch.nn.Module):
    def forward(self, x):
        return torch.sin(1.0 * x)

# Denis Andric
class mlp(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,hid_layers,acts,bias=True): # input layer,hidden_layersout,put_layer
        super(mlp,self).__init__()
        if len(acts) != 2+hid_layers:
            print("mlp is wrong")
        modules = []
        modules.append(torch.nn.Linear(input_dim,hidden_dim))
        modules.append(function_act(acts[0]))
        for i in range(1,hid_layers+1):
            modules.append(torch.nn.Linear(hidden_dim,hidden_dim,bias=bias))
            if acts[i] != "":
                modules.append(function_act(acts[i]))
        modules.append(torch.nn.Linear(hidden_dim,output_dim))
        if acts[i] != "":
            modules.append(function_act(acts[-1]))
        self.net = torch.nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.normal_(m.weight,mean=0.,std=0.1)
                torch.nn.init.constant_(m.bias,val=0)
        
    def forward(self,y):
        return self.net(y.float())


def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl






class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)


class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "{} Output tensor should have shape [batch_size, 2]".format(y.shape)
        return y.split(1,1)

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)
    
    def rollout(self,x0,t):
      out = device_util.DEVICETensor(len(t),2)
      out[0,:] = x0
      for i in range(1,len(t)):
          print(i)
          dt = t[i]-t[i-1]
          out[i:i+1,:] = out[i-1:i,:] + self.rk4_time_derivative(out[i-1:i,:],dt)
      return out
           
     
          
          
          
       

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x) # traditional forward pass

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M
