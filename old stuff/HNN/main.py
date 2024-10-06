import torch
import torch.nn as nn
import numpy as np
from hnn import mlp,HNN,integrate_model
import device_util
from device_util import DEVICETensor

from osci import oscilator
import sys
from tqdm import tqdm



def make_traj_coords(SAMPLES,TIME_POINTS,TIME_BATCH):
    b = SAMPLES
    t = int(TIME_POINTS-TIME_BATCH)
    coords = np.indices((b,t)).reshape(2,-1).T

    return coords

def makebatchHNNsingle(dataset,coords,batch_time,batch_size):
    output = DEVICETensor(batch_time,batch_size,2)
    for i in range(batch_size):
        coord = coords[i,:]
        output[:,i,:] = dataset[coord[1]:coord[1]+batch_time,:]
    return output



###dataset creating

k = 0.5
m = 2

data_creator = oscilator(m,k)

dataset, t = data_creator.make_one(1024) #random H 1-5
HNNdataset = dataset.squeeze()

nnmodel = mlp(2,256,2,1,["tanh","relu",""])
model = HNN(2,nnmodel,field_type="solenoidal")
optim = torch.optim.Adam(model.parameters(), 1e-3)
batch_size = 1
batch_time = 32

epochs = 1

coords = make_traj_coords(1,1024,32)
if len(coords) % batch_size == 0:
    BATCHES = int(len(coords)/batch_size)
else:
    sys.exit("batches doesnt have same size")
        
print("BATCHES pro epoch: {}".format(BATCHES))


lossfn = nn.HuberLoss()
testloss_fn = nn.HuberLoss()
acc_fn = nn.MSELoss()
tb =t[0:batch_time].to(device_util.DEVICE)

for epoch in tqdm(range(epochs)):
    model.train()
    np.random.shuffle(coords)
    train_loss = 0
    for batch in range(BATCHES):
        optim.zero_grad()
        coord = coords[batch:batch+batch_size,:]

        batchHNN = makebatchHNNsingle(HNNdataset,coord,batch_time,batch_size).squeeze().to(device_util.DEVICE)
        pred = model.rollout(batchHNN[0:1,:],tb)
        Loss = lossfn(batchHNN,pred)
        train_loss += Loss.item()
        Loss.backward()
        optim.step()


    model.eval()

















