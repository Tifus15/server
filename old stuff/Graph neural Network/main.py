import torch
from model import ODEBlock
import torch_geometric
import torch_geometric.nn as geom_nn
from threebody import threebody
from samples import SAMPLES
import dgl
from device_util import DEVICE
import numpy as np
import torch.optim as optim
import torch.nn as nn

batch_time = 16
batch_size = 16

def make_batches(dataset,batch_time,batch_size):
    samples = dataset.shape[0]-batch_time
    batched_dataset = torch.zeros(int(samples/batch_size),batch_time,batch_size,3,4)

    coords = np.indices((1,samples)).reshape(2,-1).T
    
    np.random.shuffle(coords)
    for i in range(int(samples/batch_size)):
        for j in range(batch_size):
          
            batched_dataset[i,:,j,:,:] = dataset[coords[int(j+7*i),1]:coords[int(j+7*i),1]+batch_time,0,:,:]

    return batched_dataset



##### edge_index of threebody
edge_index = torch.Tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]).T.int().to(DEVICE)

dataset_creator = threebody(SAMPLES)
dataset, t = dataset_creator.dataset_one(128,"yarn")
dataset_graph = dataset.reshape(128,1,3,4)
batched_dataset = make_batches(dataset_graph,batch_time,batch_size).to(DEVICE)
print(batched_dataset.shape)

x0 = batched_dataset[2,0,:,:,:].to(DEVICE)
print(x0.shape)

model = geom_nn.GCN(4,128,2,4).to(DEVICE)
gde = ODEBlock(odefunc=model, method='rk4', atol=1e-3, rtol=1e-4, adjoint=True).to(DEVICE)

x = model(x0,edge_index)
print(x.shape)

### training
optimizer = optim.AdamW(gde.parameters(),lr=1e-3)
loss_fn = nn.HuberLoss()
epochs = 10
batches = 7
loss_train = torch.zeros(batches,epochs)
loss_test = torch.zeros(epochs)
for epoch in range(epochs):
    batched_dataset = make_batches(dataset_graph,batch_time,batch_size).to(DEVICE)
    model.train()
    for batch in range(batches):
        optimizer.zero_grad()
        xt = batched_dataset[batch,:,:,:,:].to(DEVICE)
        x0 = xt[0,:,:,:].to(DEVICE)
        tb = t[0:batch_size].to(DEVICE)
        xp = gde(x0,tb[-1]).to(DEVICE)
        loss_tr = loss_fn(xp,xt)
        loss_train[batch,epoch] = loss_tr.item() 
        loss_tr.backward()
        optimizer.step()

    


    model.eval()
    with torch.no_grad():
        xtest = gde(dataset_graph[0,:,:,:].to(DEVICE),t[-1]).to(DEVICE)
        loss_t = loss_fn(xtest,dataset_graph.to(DEVICE))
        loss_test[epoch] = loss_t.item()


        print("EPOCH: {}, train {:.6f}, test {:.6f}".format(epoch,torch.mean(loss_train[:,epoch]),loss_test))









