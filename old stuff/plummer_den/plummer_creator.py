import numpy as np
import torch
import math
import matplotlib.pyplot as plt

##define a as 1 

def pos_on_sphere(r,phi,theta):
    out=torch.zeros(3,1)
    out[0] = math.sin(theta) * math.cos(phi)
    out[1] = math.sin(theta) * math.sin(phi)
    out[2] = math.cos(theta)
    return r*out




def mass_r(M,a,r):
    return M*(1+(a**2)/(r**2))**(-3/2)

def r_phys(a,M,m):
    return a/torch.sqrt((M/m**(2/3))-1)

def function(q):
    out = ((1-(q**2))**(7/2))*(q**2)
    return out


ms = torch.Tensor([1,2,3,4])

rs = r_phys(1,10,ms)

print(rs)

q = np.linspace(0,1.0,1000)
plt.figure()
plt.plot(q,function(q))
plt.show()







