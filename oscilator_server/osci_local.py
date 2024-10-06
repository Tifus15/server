import os
import numpy as np
#from viz import visualize_dataset_traj_osci
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from  helpers import make_batches,make_traj_coords,make_batch_mlp,make_batch_odeint

from torchdiffeq import odeint_adjoint as odeint

import device_util
from device_util import DEVICETensor,CPUTensor
from models import mlp, ode_mlp, GRU_ODEINT,GRU_ODEINT_EULER,RNN_ODEINT,RNN_ODEINT_EULER
from decimal import * 


from tqdm import tqdm

import matplotlib.pyplot as plt
import sys

from oscilator import oscilator

import wandb
import yaml




## MY EXPERIMENT
def osci(configs):
    seed=41

    TRAIN =configs["train"]
    TEST = configs["test"]
    EVAL = configs["eval"] 

    NSAMPLES = TRAIN + TEST + EVAL
    dim = configs["dim"]
    BATCH_SIZE = configs["batch-size"]
    BATCH_TIME = configs["batch-time"]
    H_span = configs["H_span"]
    lr = configs["learning_rate"]
    TIMEPOINTS = configs["timepoints"]

    torch.manual_seed(seed)
    np.random.seed(seed) 

    k = configs["k"]
    m = configs["m"]
    #dt = configs["dt"]

    with torch.no_grad():
        data_maker=oscilator(k,m)
        #T = round(data_maker.getT(),abs(Decimal(str(dt)).as_tuple().exponent))
        #TIMEPOINTS = int(T / dt) + 1
        #print("T = {}\n dt = {}\n points = {}".format(T,dt,TIMEPOINTS))
        
        dataset, t = data_maker.make_dataset(TIMEPOINTS,NSAMPLES,T=0,H_span=H_span)
        #visualize_dataset_traj_osci(dataset.cpu())
        trainset = dataset[:,0:TRAIN,:,:]
        testset = dataset[:,TRAIN:TRAIN+TEST,:,:].to(device_util.DEVICE)
        eval = dataset[:,(TRAIN+TEST):,:,:].to(device_util.DEVICE)
        print("eval shape: {}".format(eval.shape))

        t = t.to(device_util.DEVICE)

        H_test = data_maker.hamiltonian(testset)
        H_eval = data_maker.hamiltonian(eval)
        
        coords = make_traj_coords(TRAIN,TIMEPOINTS,BATCH_TIME)
    ##### LEARNING RATES
    LR_RATE_MLP = lr
    LR_RATE_ODE = lr
    LR_RATE_GRU = lr
    LR_RATE_RNN = lr 
    LR_RATE_GRU_EULER = lr
    LR_RATE_RNN_EULER = lr

    #### MODELS

    modelODE = ode_mlp(dim,128,dim,1,["tanh","relu",""]).to(device_util.DEVICE)
    modelMLP = mlp(dim,128,dim,2,["tanh","relu","relu",""]).to(device_util.DEVICE)
    modelGRU = GRU_ODEINT(dim,dim,256).to(device_util.DEVICE)
    modelRNN = RNN_ODEINT(dim,dim,256).to(device_util.DEVICE)
    modelGRE = GRU_ODEINT_EULER(dim,dim,256).to(device_util.DEVICE)
    modelRNE = RNN_ODEINT_EULER(dim,dim,256).to(device_util.DEVICE)

    #### optimizers

    optiODE = optim.AdamW(modelODE.parameters(),lr=LR_RATE_ODE)
    optiMLP = optim.AdamW(modelMLP.parameters(),lr=LR_RATE_MLP)
    optiGRU = optim.AdamW(modelGRU.parameters(),lr=LR_RATE_GRU)
    optiRNN = optim.AdamW(modelGRE.parameters(),lr=LR_RATE_GRU_EULER)
    optiGRE = optim.AdamW(modelRNN.parameters(),lr=LR_RATE_RNN)
    optiRNE = optim.AdamW(modelRNE.parameters(),lr=LR_RATE_RNN_EULER)




    lossfn = nn.HuberLoss()
    testloss_fn = nn.HuberLoss()
    acc_fn = nn.MSELoss()

    WARMUP = configs["warmup"]
    EPOCHS =configs["epochs"]

    loss_container=torch.zeros(EPOCHS,18).to(device_util.DEVICE)

    if len(coords) % BATCH_SIZE == 0:
        BATCHES = int(len(coords)/BATCH_SIZE)
    else:
        sys.exit("batches doesnt have same size")
        
    print("BATCHES pro epoch: {}".format(BATCHES))
    for epoch in tqdm(range(-WARMUP,EPOCHS)):
        
        if epoch < 0:
            print("WARMUP")
        metrics={"mlp":{"train_loss": 0,"test_loss" : 0,"ham_acc":0},
            "ode":{"train_loss": 0,"test_loss" : 0,"ham_acc":0},
            "gru":{"train_loss": 0,"test_loss" : 0,"ham_acc":0},
            "gre":{"train_loss": 0,"test_loss" : 0,"ham_acc":0},
            "rnn":{"train_loss": 0,"test_loss" : 0,"ham_acc":0},
            "rne":{"train_loss": 0,"test_loss" : 0,"ham_acc":0}}
        train_accum = []
        for i in range(len(metrics.keys())):
            train_accum.append(0)
        
        modelMLP.train()
        modelODE.train()
        modelGRU.train()
        modelGRE.train()
        modelRNN.train()
        modelRNE.train()
        np.random.shuffle(coords)
        for batch in tqdm(range(BATCHES)):
            coord = coords[batch:batch+BATCH_SIZE,:]
            tb =t[0:BATCH_TIME].to(device_util.DEVICE)
            optiMLP.zero_grad()
            optiODE.zero_grad()
            optiGRU.zero_grad()
            optiRNN.zero_grad()
            optiGRE.zero_grad()
            optiRNE.zero_grad()
            
            
            dim4_batch = make_batch_odeint(trainset,BATCH_SIZE,BATCH_TIME,coord)
            dim3_batch =torch.squeeze(dim4_batch).to(device_util.DEVICE)
            
            ode_pred = odeint(modelODE,dim4_batch[0,:,:,:],tb,method="rk4")
            mlp_pred = modelMLP(dim4_batch[:-1,:,:,:])
            gru_pred = modelGRU(dim3_batch[0,:,:].reshape(1,BATCH_SIZE,-1),tb)
            rnn_pred = modelRNN(dim3_batch[0,:,:].reshape(1,BATCH_SIZE,-1),tb)
            gre_pred = modelGRE(dim3_batch[0,:,:].reshape(1,BATCH_SIZE,-1),tb)
            rne_pred = modelRNE(dim3_batch[0,:,:].reshape(1,BATCH_SIZE,-1),tb)
            
            lossmlp = lossfn(mlp_pred,dim4_batch[1:,:,:,:])
            lossode = lossfn(ode_pred,dim4_batch)
            lossgru = lossfn(gru_pred,dim3_batch)
            lossrnn = lossfn(rnn_pred,dim3_batch)
            lossgre = lossfn(gre_pred,dim3_batch)
            lossrne = lossfn(rne_pred,dim3_batch)
            
            train_accum[0] += lossmlp.item()
            train_accum[1] += lossode.item()
            train_accum[2] += lossgru.item()
            train_accum[3] += lossrnn.item()
            train_accum[4] += lossgre.item()
            train_accum[5] += lossrne.item()
            
            lossmlp.backward()
            lossode.backward()
            lossgru.backward()
            lossrnn.backward()
            lossgre.backward()
            lossrne.backward()
            
            optiMLP.step()
            optiODE.step()
            optiGRU.step()
            optiRNN.step()
            optiGRE.step()
            optiRNE.step()
        
        with torch.no_grad():
            
            modelMLP.eval()
            modelODE.eval()
            modelGRU.eval()
            modelGRE.eval()
            modelRNN.eval()
            modelRNE.eval()
            t=t.to(device_util.DEVICE)
            ode_test = odeint(modelODE,testset[0,:,:,:],t,method="rk4")
            mlp_test = modelMLP(testset[:-1,:,:,:])
            gru_test = modelGRU(testset.squeeze()[0,:,:].reshape(1,TEST,-1),t)
            rnn_test = modelRNN(testset.squeeze()[0,:,:].reshape(1,TEST,-1),t)
            gre_test = modelGRE(testset.squeeze()[0,:,:].reshape(1,TEST,-1),t)
            rne_test = modelRNE(testset.squeeze()[0,:,:].reshape(1,TEST,-1),t)
            
            lossode_test = testloss_fn(ode_test,testset)
            lossmlp_test = testloss_fn(mlp_test,testset[1:,:,:,:])
            lossgru_test = testloss_fn(gru_test,testset.squeeze())
            lossrnn_test = testloss_fn(rnn_test,testset.squeeze())
            lossgre_test = testloss_fn(gre_test,testset.squeeze())
            lossrne_test = testloss_fn(rne_test,testset.squeeze())
            
            metrics["mlp"]["train_loss"] = train_accum[0]/BATCHES
            metrics["ode"]["train_loss"] = train_accum[1]/BATCHES
            metrics["gru"]["train_loss"] = train_accum[2]/BATCHES
            metrics["rnn"]["train_loss"] = train_accum[3]/BATCHES
            metrics["gre"]["train_loss"] = train_accum[4]/BATCHES
            metrics["rne"]["train_loss"] = train_accum[5]/BATCHES
            
            metrics["mlp"]["test_loss"] = lossmlp_test.item()
            metrics["ode"]["test_loss"] = lossode_test.item()
            metrics["gru"]["test_loss"] = lossgru_test.item()
            metrics["rnn"]["test_loss"] = lossrnn_test.item()
            metrics["gre"]["test_loss"] = lossgre_test.item()
            metrics["rne"]["test_loss"] = lossrne_test.item()
            
            
            
            
            mlp_eval = DEVICETensor(len(t),TEST,1,dim)
            mlp_eval[0,:,:,:]= testset[0,:,:,:]
            for i in range(1,len(t)):
                    mlp_eval[i,:,:] = modelMLP(mlp_eval[i-1,:,:])
                    
            
            H_ode = data_maker.hamiltonian(ode_test)
            H_mlp = data_maker.hamiltonian(mlp_eval)
            H_gru = data_maker.hamiltonian(gru_test)
            H_rnn = data_maker.hamiltonian(rnn_test)
            H_gre = data_maker.hamiltonian(gre_test)
            H_rne = data_maker.hamiltonian(rne_test)
            
            metrics['mlp']['ham_acc'] = acc_fn(H_mlp,H_test)
            metrics['ode']['ham_acc'] = acc_fn(H_ode,H_test)
            metrics['gru']['ham_acc'] = acc_fn(H_gru,H_test)
            metrics['rnn']['ham_acc'] = acc_fn(H_rnn,H_test)
            metrics['gre']['ham_acc'] = acc_fn(H_gre,H_test)
            metrics['rne']['ham_acc'] = acc_fn(H_rne,H_test)
            
            loss_container[epoch,0] = metrics["mlp"]["train_loss"]
            loss_container[epoch,1] = metrics["ode"]["train_loss"]
            loss_container[epoch,2] = metrics["gru"]["train_loss"]
            loss_container[epoch,3] = metrics["rnn"]["train_loss"]
            loss_container[epoch,4] = metrics["gre"]["train_loss"]
            loss_container[epoch,5] = metrics["rne"]["train_loss"]
            
            loss_container[epoch,6] = metrics["mlp"]["test_loss"]
            loss_container[epoch,7] = metrics["ode"]["test_loss"]
            loss_container[epoch,8] = metrics["gru"]["test_loss"]
            loss_container[epoch,9] = metrics["rnn"]["test_loss"]
            loss_container[epoch,10] = metrics["gre"]["test_loss"]
            loss_container[epoch,11] = metrics["rne"]["test_loss"]
            
            loss_container[epoch,12] = metrics["mlp"]["ham_acc"]
            loss_container[epoch,13] = metrics["ode"]["ham_acc"]
            loss_container[epoch,14] = metrics["gru"]["ham_acc"]
            loss_container[epoch,15] = metrics["rnn"]["ham_acc"]
            loss_container[epoch,16] = metrics["gre"]["ham_acc"]
            loss_container[epoch,17] = metrics["rne"]["ham_acc"]
            
            
            
        
            
            print("####################################################\n"+
                "EPOCH: {}\n".format(epoch+1) +
                "              train                    test                  ham_acc    \n"+
                "  MLP:        {}                       {}                    {}         \n".format(metrics["mlp"]["train_loss"], metrics["mlp"]["test_loss"], metrics["mlp"]["ham_acc"]) +
                "  ODE:        {}                       {}                    {}         \n".format(metrics["ode"]["train_loss"], metrics["ode"]["test_loss"], metrics["ode"]["ham_acc"]) +
                "  GRU:        {}                       {}                    {}         \n".format(metrics["gru"]["train_loss"], metrics["gru"]["test_loss"], metrics["gru"]["ham_acc"]) +
                "  RNN:        {}                       {}                    {}         \n".format(metrics["rnn"]["train_loss"], metrics["rnn"]["test_loss"], metrics["rnn"]["ham_acc"]) +
                "  GRE:        {}                       {}                    {}         \n".format(metrics["gre"]["train_loss"], metrics["gre"]["test_loss"], metrics["gre"]["ham_acc"]) +
                "  RNE:        {}                       {}                    {}         \n".format(metrics["rne"]["train_loss"], metrics["rne"]["test_loss"], metrics["rne"]["ham_acc"]))
                
            
            wandb.log(metrics)
            
    

    if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        
    torch.save(modelMLP,results_dir+"/MLPmodel.pt")    
    torch.save(modelODE,results_dir+"/ODEmodel.pt")
    torch.save(modelRNN,results_dir+"/RNNmodel.pt")
    torch.save(modelGRU,results_dir+"/GRUmodel.pt")
    torch.save(modelGRE,results_dir+"/GREmodel.pt")
    torch.save(modelRNE,results_dir+"/RNEmodel.pt")
    torch.save(loss_container,results_dir+"/losses.pt")
    torch.save(eval,results_dir+"/eval.pt")
    #torch.save(H_eval,results_dir+"/h_eval.pt") 



if __name__ == "__main__":
    wandb.login()


    config_file_path: str = './config/osci.yaml'

    #######################################
    # MANDATORY

    results_dir: str = 'res_osci'

    with open(config_file_path, 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)
    wandb.init(config=configs,project="osci_benchmark")
    osci(configs)
    wandb.finish()
    print("DONE")
               
                
                
                
            
            
    
    
    

    

    
            
    
    
    
         
    
    
    
  
   
   
   
   