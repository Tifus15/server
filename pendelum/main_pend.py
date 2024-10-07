import torch
import dgl
from utils import *
import random
import yaml
from ghnn_model import *
import wandb
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
#from torch_symplectic_adjoint import odeint
from torchdiffeq import odeint
import os
"""
def full_server(config):
    WANDB = config["wandb"]
 
    CUDA = config["cuda"]
 
    MONITORING = config["monitoring"]
    LR=config["lr"]
    BIAS = config["bias"]
    SET_MAX_TRAIN_BATCHES = config["train_break"] # set -1 without
    SET_MAX_TEST_BATCHES = config["test_break"] # set -1 without
    SOB = config["sob"]
    BATCHTIME = config["tbatch"]
    BATCHSIZE = config["sbatch"]
    SAMPLES = config["samples"] # RAM issues for 16GiB - 75 samples * 4 types is optimal
    EPOCHS = config["epochs"]
    LAYER_SIZE = config["size"]
    E_SIZE = config["e_size"]
    LAYER = config["layer"]    
    #DROPSNAPS = 0.4 # whole dataset is only 60% - make it smaller
    TIME = config["max_sim_time"]
    SOB = config["sob"]
    print("loaded")

    real_t = torch.linspace(0,1.27,128)
    loss_container = torch.Tensor(EPOCHS,8)
    modGHNN =GNN_maker_HNN(g,2,LAYER_SIZE,E_SIZE,["softplus"," "],BIAS,LAYER,0.3)
    modGRUGHNN = rollout_GNN_GRU(g,2,LAYER_SIZE,E_SIZE,["softplus"," "],BIAS,LAYER,0.3)
    
    optiGHNN = torch.optim.AdamW(modGHNN.parameters(),lr=LR)
    optiGRUGHNN =torch.optim.AdamW(modGRUGHNN.parameters(),lr=LR)

    lossfn = nn.MSELoss()
    ts = real_t[0:BATCHTIME]




    ####dof1

    src = src_list(1)
    dst = dst_list(1)
    g = dgl.graph((src,dst))
    data = torch.load("pend_1.pt").requires_grad_(False)

    x_temp = data[:,:,:,0:4]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
    
    print(xs[0].shape)
    border = int(0.9*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    

    train_snapshots = create_pend1dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend1dof_graph_snapshots(test,h_test,src,dst)

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(BATCHTIME*BATCHSIZE):
        src = src_list(1)
        dst = dst_list(1)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_GHNN_roll":0,  "train_GHNN_vec":0, "train_GHNN_h" :0, 
             "train_GHNN_summary" :0,
             "test_GRUGHNN_roll":0,  "test_GRUGHNN_vec":0, "test_GHNN_h" :0, 
             "test_GRUGHNN_summary" :0 }

    wandb.watch(modGHNN,log='all')
    wandb.watch(modGRUGHNN,log='all')

    






"""
def full_server4(configs,dic_base):
    print("begin 4dof")
    train4dof(configs,dic_base)
    print("end 4dof")
    
def full_server3(configs,dic_base):
    print("begin 3dof")
    train3dof(configs,dic_base)
    print("end 3dof")
   
   
def full_server43(configs,dic_base):
    print("begin 4dof")
    train4dof(configs,dic_base)
    print("end 4dof")
    print("begin 3dof")
    train3dof(configs,dic_base)
    print("end 3dof")
   

def full_server34(configs,dic_base):
    """
    print("begin 1dof")
    train1dof(configs)
    print("end 1dof")
    print("begin 2dof")
    
    train2dof(configs)
    print("end 2dof")
    """
    print("begin 3dof")
    train3dof(configs,dic_base)
    print("end 3dof")
    print("begin 4dof")
    train4dof(configs,dic_base)
    print("end 4dof")
    
def train1dof(configs):
    

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    NOLOOPS = configs["noloops"]
    WANDB = True
    BIAS = configs["bias"]
    S = configs["samples"]
    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "1dof pendelum"
    print(EPOCHS)
    DOF = 1
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("traj_1dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(1)
    dst = dst_list(1)
    if NOLOOPS and DOF != 1 :
        src,dst = del_loops(src,dst)
        
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    
    
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias=BIAS)
    print(model)
    
    
    data[:,:,:,0] = angle_transformer(data[:,:,:,0])
    
    
    #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = eval[:,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)

    x_temp = data[:,:,:,0:4]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    

    train_snapshots = create_pend1dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend1dof_graph_snapshots(test,h_test,src,dst)
    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    #dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    

 
    ts = t[0:TIME_SIZE]

   
    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)

    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

    
    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if DOF != 1:
            src, dst = make_graph_no_loops(1,0)
        else:
            src = src_list(1)
            dst = dst_list(1)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d1":0,  "train_H_d1":0, "test_loss_d1" :0, "test_H_d1" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            h_tr = train_sample.ndata["h"].transpose(0,1)
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = test_sample.ndata["h"].transpose(0,1)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d1"] = container[0,epoch]
        metrics["test_loss_d1"] = container[2,epoch]
        metrics["train_H_d1"] = container[1,epoch]
        metrics["test_H_d1"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
       
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        dict={}
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
    
    visualize_loss("loss of 1dof pendelum",container)
    torch.save(model.state_dict(),"server_1dof.pth")
    #torch.save(model,"whole_model_dof1.pt")
    
def train2dof(configs):
    S = configs["samples"]
    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    NOLOOPS = configs["noloops"]
    WANDB = True
    BIAS = configs["bias"]

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
 

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "2dof pendelum"
    print(EPOCHS)
    
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("traj_2dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(2)
    dst = dst_list(2)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:2] = angle_transformer(data[:,:,:,0:2])

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = eval[:,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)
    
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]

    
    xs, hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
  
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    
    train_snapshots = create_pend2dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend2dof_graph_snapshots(test,h_test,src,dst)
    
    
    
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias =BIAS)
    if os.path.isfile("server_1dof.pth"):
        print("loading prevoius model")
        model = load_model(model,"server_1dof.pth")
    

    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)

    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

   
    
 

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if NOLOOPS:
            src, dst = make_graph_no_loops(2,0)
        else:
            src = src_list(2)
            dst = dst_list(2)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d2":0,  "train_H_d2":0, "test_loss_d2" :0, "test_H_d2" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = h_tr = correct_ham_data(test_sample)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d2"] = container[0,epoch]
        metrics["test_loss_d2"] = container[2,epoch]
        metrics["train_H_d2"] = container[1,epoch]
        metrics["test_H_d2"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
        dict={}
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
   
    
    visualize_loss("loss of 2dof pendelum",container)
    torch.save(model.state_dict(),"server_2dof.pth")
    #torch.save(model,"whole_model_dof2.pt")
    
def train3dof(configs,dic_base):
    
    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    BIAS = configs["bias"]
    WANDB = True

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]
    S= configs["samples"]
    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "3dof pendelum"
    print(EPOCHS)
    NOLOOPS = configs["noloops"]
    REG = "ridge"
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("pend_3.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    """
    src = src_list(3)
    dst = dst_list(3)
    """
    src = [0,0,0,1,1,1,2,2,2]
    dst = [0,1,2,0,1,2,0,1,2]
    if NOLOOPS:
        src,dst = del_loops(src,dst)
    
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:3] = angle_transformer(data[:,:,:,0:3]) # to be sure

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = data[:,-1,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)

    
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
    
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    train_snapshots = create_pend3dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend3dof_graph_snapshots(test,h_test,src,dst)
    
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = rollout_GNN_GRU(graph,2,128,8,["tanh"," "],bias=BIAS,type = MODEL,dropout=0.65)
    #model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias=BIAS)
    #if os.path.isfile("server_2dof.pth"):
        #print("loading prevoius model")
        #model = load_model(model,"server_2dof.pth")
    
    dic = "res_4dof/"
    if os.path.isfile(dic_base+"/"+dic+"server_4dof.pth"):
        print("loading prevoius model")
        model.train()
        model = load_model(model,dic_base+"/"+dic+"server_4dof.pth")
        
        

    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)
    if os.path.isfile(dic_base+"/"+dic+"server_3dof.pth"):
        opti.load_state_dict(torch.load(dic_base+"/"+dic+"server_opti.pth"))

    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
   
    
    metrics={"train_sum":0,  "train_roll":0, "train_vec" :0, "train_h" :0,
             "test_sum":0,  "test_roll":0, "test_vec" :0, "test_h" :0}
        

    container = torch.zeros(8,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    #wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossvec=0
            lossroll=0
            reg = 0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:].requires_grad_()
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            #model.change_graph(roll_g)
            #print(roll_g)
            #x_tr_flat = x_tr.reshape(-1,2)
            model.change_graph(train_sample)
            x_pred, dx_pred, h_pred = model(ts,x0)
            #print(x_tr_flat.shape)
            #h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            lossvec = lossfn(dx_pred[:,:,0],dx_tr[:,:,0])+lossfn(dx_pred[:,:,1],dx_tr[:,:,1])
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])
            #if REG == "ridge":
            #    reg= sum(p.square().sum() for p in model.parameters())
            #if REG == "lasso":
            #    reg= sum(p.abs().sum() for p in model.parameters())
            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            loss += s_alpha[2]*lossvec
            loss += 0.01 * reg
            container[0,epoch] += loss.item()
            container[1,epoch]+=lossroll.item()
            container[2,epoch]+=lossvec.item()
            container[3,epoch]+=lossH.item()
            
            loss.backward()
            opti.step()
        container[0:4,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossvect=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = correct_ham_data(test_sample)
            #print(h_ts)
            #print(h_ts.shape)
            
            model.change_graph(test_sample)
            x_ts = x_ts.requires_grad_()
            x0 = x_ts[0,:,:]
            #x_ts_flat = x_ts.reshape(-1,2)
            #h_pred = model(x_ts_flat)
            x_pred,dx_pred,h_pred =model(ts,x0.requires_grad_()) 
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())

            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
            lossvect = lossfn(dx_pred[:,:,0],dx_ts[:,:,0])+lossfn(dx_pred[:,:,1],dx_ts[:,:,1])
        
            
            losst+=s_alpha[0] * lossROLLt
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[2] * lossvect 
                
           
            container[4,epoch]+=losst.item()
            container[5,epoch]+=lossROLLt.item()
            container[6,epoch]+=lossvect.item()
            container[7,epoch] += lossHt.item()
        container[4:8,epoch]/=N_test
    
        metrics["train_sum"] = container[0,epoch]
        metrics["train_roll"] = container[1,epoch]
        metrics["train_vec"] = container[2,epoch]
        metrics["train_h"] = container[3,epoch]
        metrics["test_sum"] = container[4,epoch]
        metrics["test_roll"] = container[5,epoch]
        metrics["test_vec"] = container[6,epoch]
        metrics["test_h"] = container[7,epoch]
        #wandb.log(metrics)
            #wandb.log_artifact(model)
        print("\nEpoch: {}\nLOSS: train: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f} \n".format(epoch+1,
                                                                                            container[0,epoch],
                                                                                            container[1,epoch],
                                                                                            container[2,epoch],
                                                                                            container[3,epoch],)+
                                "      test: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f}".format(container[4,epoch],
                                                                                            container[5,epoch],
                                                                                            container[6,epoch],
                                                                                            container[7,epoch]))
        
   
   
    model.train()
    dic = "res_3dof/"
    #visualize_loss("loss of 3dof pendelum",container)
    torch.save(model.state_dict(),dic_base+"/"+dic+"server_3dof.pth")
    torch.save(opti.state_dict(),dic_base+"/"+dic+"server_opti.pth")
    torch.save(container,dic_base+"/"+dic+"losses.pt")
    model.eval()
    torch.save(model,dic_base+"/"+dic+"model.pt")
    torch.save(eval,dic_base+"/"+dic+"eval.pt")
    torch.save(H,dic_base+"/"+dic+"eval_H.pt")
    #torch.save(model,"whole_model_dof3.pt")
    
def train4dof(configs,dic_base):

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this

    WANDB = True

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]
    S = configs["samples"]
    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "4dof pendelum"
    BIAS = configs["bias"]
    print(EPOCHS)
    
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
    NOLOOPS = configs["noloops"]
    data = torch.load("pend_4.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(4)
    dst = dst_list(4)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:4] = angle_transformer(data[:,:,:,0:4])

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = data[:,-1,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]
    
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE)
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    train_snapshots = create_pend4dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend4dof_graph_snapshots(test,h_test,src,dst)
    
    ts = t[0:TIME_SIZE]
    dic = "res_3dof/"
    #half = int(dim/6) 
    model = rollout_GNN_GRU(graph,2,128,8,["tanh"," "],bias=BIAS,type = MODEL,dropout=0.65)
    if os.path.isfile(dic_base+"/"+dic+"server_3dof.pth"):
        print("loading prevoius model")
        model.train()
        model = load_model(model,dic_base+"/"+dic+"server_3dof.pth")
        
        

    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)
    if os.path.isfile(dic+"server_3dof.pth"):
        opti.load_state_dict(torch.load(dic_base+"/"+dic+"server_opti.pth"))
        
    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if NOLOOPS:
            src, dst = make_graph_no_loops(4,0)
        else:
            src = src_list(4)
            dst = dst_list(4)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_sum":0,  "train_roll":0, "train_vec" :0, "train_h" :0,
             "test_sum":0,  "test_roll":0, "test_vec" :0, "test_h" :0}
        

    container = torch.zeros(8,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    #wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossvec=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:].requires_grad_()
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            #model.change_graph(roll_g)
            #print(roll_g)
            #x_tr_flat = x_tr.reshape(-1,2)
            model.change_graph(train_sample)
            x_pred, dx_pred, h_pred = model(ts,x0)
            #print(x_tr_flat.shape)
            #h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            lossvec = lossfn(dx_pred[:,:,0],dx_tr[:,:,0])+lossfn(dx_pred[:,:,1],dx_tr[:,:,1])
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            loss += s_alpha[2]*lossvec
            
            container[0,epoch] += loss.item()
            container[1,epoch]+=lossroll.item()
            container[2,epoch]+=lossvec.item()
            container[3,epoch]+=lossH.item()
            
            loss.backward()
            opti.step()
        container[0:4,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossvect=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts =  correct_ham_data(test_sample)
            #print(h_ts)
            #print(h_ts.shape)
            model.change_graph(test_sample)
            x_ts = x_ts.requires_grad_()
            x0 = x_ts[0,:,:]
            #x_ts_flat = x_ts.reshape(-1,2)
            #h_pred = model(x_ts_flat)
            x_pred,dx_pred,h_pred =model(ts,x0.requires_grad_()) 
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())

            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
            lossvect = lossfn(dx_pred[:,:,0],dx_ts[:,:,0])+lossfn(dx_pred[:,:,1],dx_ts[:,:,1])
        
            
            losst+=s_alpha[0] * lossROLLt
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[2] * lossvect 
                
           
            container[4,epoch]+=losst.item()
            container[5,epoch]+=lossROLLt.item()
            container[6,epoch]+=lossvect.item()
            container[7,epoch] += lossHt.item()
        container[4:8,epoch]/=N_test
    
        metrics["train_sum"] = container[0,epoch]
        metrics["train_roll"] = container[1,epoch]
        metrics["train_vec"] = container[2,epoch]
        metrics["train_h"] = container[3,epoch]
        metrics["test_sum"] = container[4,epoch]
        metrics["test_roll"] = container[5,epoch]
        metrics["test_vec"] = container[6,epoch]
        metrics["test_h"] = container[7,epoch]
        #wandb.log(metrics)
            #wandb.log_artifact(model)
        
        print("Epoch: {}\nLOSS: train: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f} \n".format(epoch+1,
                                                                                            container[0,epoch],
                                                                                            container[1,epoch],
                                                                                            container[2,epoch],
                                                                                            container[3,epoch])+
                                "test: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f}".format(container[4,epoch],
                                                                                            container[5,epoch],
                                                                                            container[6,epoch],
                                                                                            container[7,epoch]))
   
   
    
    visualize_loss("loss of 4dof pendelum",container)
    dic4 = "res_4dof/"
    torch.save(model,dic_base+"/"+dic4+"model.pt")
    torch.save(container,dic_base+"/"+dic4+"losses.pt")
    torch.save(eval,dic_base+"/"+dic4+"eval.pt")
    torch.save(H,dic_base+"/"+dic4+"eval_H.pt")


if __name__ == "__main__":
    
    with open("configs/pend.yaml", 'r') as f:
        configs = yaml.load(f, yaml.Loader)
    folder_name1 = "a34"
    full_server34(configs,folder_name1)
    folder_name2 = "a43"
    full_server43(configs,folder_name2)
   # wandb.init()
   # print('Config file content:')
   # print(configs)
   # train1dof(configs)
    
   # train2dof(configs)
   # train3dof(configs)
   # train4dof(configs)
    #wandb.finish()
    
