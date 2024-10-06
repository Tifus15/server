import device_util
import numpy as np
from device_util import DEVICETensor
import torch


### the size [t,b,1,dim_problem]
def make_batches(dataset,batch_size):
    idx = torch.randperm(dataset.shape[1])
    shuffled = dataset[:,idx,:,:]
    batch_list = torch.split(shuffled,batch_size,dim=1)
    return batch_list 

def make_traj_coords(SAMPLES,TIME_POINTS,TIME_BATCH):
    b = SAMPLES
    t = int(TIME_POINTS-TIME_BATCH)
    coords = np.indices((b,t)).reshape(2,-1).T

    return coords

def make_batch_odeint(data,BATCH_SIZE,BATCH_TIME,coordsbatch):
    output = DEVICETensor(BATCH_TIME,BATCH_SIZE,1,data.shape[-1])
    for i in range(BATCH_SIZE):
        coords = coordsbatch[i,:]
        output[:,i,:,:]= data[coords[1]:coords[1]+BATCH_TIME,coords[0],:,:]
    return output 

def make_batch_mlp(data,BATCH_SIZE,BATCH_TIME,coordsbatch):
    sample = make_batch_odeint(data,BATCH_SIZE,BATCH_TIME,coordsbatch)
    inputs = sample[:-1,:,:,:].to(device_util.DEVICE)
    outputs = sample[1:,:,:,:].to(device_util.DEVICE)
    return inputs, outputs  
    
if __name__ == "__main__":
    
    dataset = torch.Tensor(100,10,1,4)
    list = make_batches(dataset,2)
    print(len(list))

    print(list[0].shape)
    print(3.56765466)
    a= round(3.56765466,2)
    print(a)
 

        
    