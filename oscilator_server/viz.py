import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from device_util import *
from oscilator import *
#from twobody import *
#from samples import SAMPLES
#from threebody import *

import matplotlib as mpl




def osci_pics():
    datamaker = oscilator(1,0.8)
    evalset = torch.load("res_osci/eval.pt")
    eval = evalset[:,0,:,:]
    T = datamaker.getT()
    t = torch.linspace(0,T,128)
    H = datamaker.hamiltonian(eval).cpu()
    H_eval = datamaker.hamiltonian(evalset).cpu()

    
    
    modelODE = torch.load("res_osci/ODEmodel.pt")
    print(modelODE) 
    ode_eval = odeint(modelODE,eval[0,:,:],t,method="rk4")
    print(ode_eval.shape)
    H_ode = datamaker.hamiltonian(ode_eval).cpu()
    print(H_ode.shape)
    
    modelMLP = torch.load("res_osci/MLPmodel.pt")
    print(modelMLP) 
    mlp_eval = DEVICETensor(len(t),1,2)
    mlp_eval[0,:,:]= eval[0,:,:]
    for i in range(1,len(t)):
        mlp_eval[i,:,:] = modelMLP(mlp_eval[i-1,:,:])
    print(mlp_eval.shape)
    H_mlp = datamaker.hamiltonian(mlp_eval).cpu()
    print(H_mlp.shape)
    
    modelGRU = torch.load("res_osci/GRUmodel.pt")
    print(modelGRU) 
    gru_eval = modelGRU(eval.squeeze()[0,:].reshape(1,-1),t)
    print(ode_eval.shape)
    H_gru = datamaker.hamiltonian(gru_eval).cpu()
    print(H_gru.shape)
    
    modelRNN= torch.load("res_osci/RNNmodel.pt")
    print(modelRNN) 
    rnn_eval = modelRNN(eval.squeeze()[0,:].reshape(1,-1),t)
    print(rnn_eval.shape)
    H_rnn = datamaker.hamiltonian(rnn_eval).cpu()
    print(H_rnn.shape)
    
    modelGRE= torch.load("res_osci/GREmodel.pt")
    print(modelGRE) 
    gre_eval = modelGRE(eval.squeeze()[0,:].reshape(1,-1),t)
    print(gre_eval.shape)
    H_gre = datamaker.hamiltonian(gre_eval).cpu()
    print(H_gre.shape)
    
    modelRNE= torch.load("res_osci/RNEmodel.pt")
    print(modelRNE) 
    rne_eval = modelRNE(eval.squeeze()[0,:].reshape(1,-1),t)
    print(rne_eval.shape)
    H_rne = datamaker.hamiltonian(rne_eval).cpu()
    print(H_rne.shape)
    
    losses = torch.load("res_osci/losses.pt").cpu()
    
    print(losses.shape)
    epochs_t = torch.linspace(1,losses.shape[0],losses.shape[0])
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,0],c="b")
    ax[0].semilogy(epochs_t,losses[:,6],c="r")
    
    ax[1].plot(epochs_t,losses[:,12],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("MLP")
    #plt.show()
    
    
    
    
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,1],c="b")
    ax[0].semilogy(epochs_t,losses[:,7],c="r")
    
    ax[1].plot(epochs_t,losses[:,13],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("NeuralODE")
   #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,2],c="b")
    ax[0].semilogy(epochs_t,losses[:,8],c="r")
    
    ax[1].plot(epochs_t,losses[:,14],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("GRU")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,3],c="b")
    ax[0].semilogy(epochs_t,losses[:,9],c="r")
    
    ax[1].plot(epochs_t,losses[:,15],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("RNN")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,4],c="b")
    ax[0].semilogy(epochs_t,losses[:,10],c="r")
    
    ax[1].plot(epochs_t,losses[:,16],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("GRU Stepper")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,5],c="b")
    ax[0].semilogy(epochs_t,losses[:,11],c="r")
    
    ax[1].plot(epochs_t,losses[:,17],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("RNN Stepper")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("MLP")
    ax[0].set_title("Phase space")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("p")
    ax[1].set_title("Total Energy")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("H")
    
    ax[0].plot(mlp_eval[:,0,0].cpu().detach().numpy(),mlp_eval[:,0,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].legend(["predicition","ground_truth"])
    stdHmlp = torch.std(H_mlp).cpu().detach().numpy()
    meanHmlp = torch.mean(H_mlp).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_mlp.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHmlp-3*stdHmlp,meanHmlp+3*stdHmlp)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("NeuralODE")
    ax[0].set_title("Phase space")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("p")
    ax[1].set_title("Total Energy")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("H")
    
    ax[0].plot(ode_eval[:,0,0].cpu().detach().numpy(),ode_eval[:,0,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].legend(["predicition","ground_truth"])
    stdHode = torch.std(H_ode).cpu().detach().numpy()
    meanHode = torch.mean(H_ode).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_ode.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHode-3*stdHode,meanHode+3*stdHode)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("GRU")
    ax[0].set_title("Phase space")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("p")
    ax[1].set_title("Total Energy")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("H")
    
    ax[0].plot(gru_eval[:,0].cpu().detach().numpy(),gru_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].legend(["predicition","ground_truth"])
    stdHgru = torch.std(H_gru).cpu().detach().numpy()
    meanHgru = torch.mean(H_gru).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_gru.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHgru-3*stdHgru,meanHgru+3*stdHgru)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("RNN")
    ax[0].set_title("Phase space")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("p")
    ax[1].set_title("Total Energy")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("H")
    
    ax[0].plot(rnn_eval[:,0].cpu().detach().numpy(),rnn_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].legend(["predicition","ground_truth"])
    stdHrnn = torch.std(H_rnn).cpu().detach().numpy()
    meanHrnn = torch.mean(H_rnn).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_rnn.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHrnn-3*stdHrnn,meanHrnn+3*stdHrnn)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("GRU Stepper")
    ax[0].set_title("Phase space")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("p")
    ax[1].set_title("Total Energy")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("H")
    
    ax[0].plot(gre_eval[:,0].cpu().detach().numpy(),gre_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].legend(["predicition","ground_truth"])
    stdHgre = torch.std(H_gre).cpu().detach().numpy()
    meanHgre = torch.mean(H_gre).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_gre.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHgre-3*stdHgre,meanHgre+3*stdHgre)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("RNN Stepper")
    ax[0].set_title("Phase space")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("p")
    ax[1].set_title("Total Energy")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("H")
    
    ax[0].plot(rne_eval[:,0].cpu().detach().numpy(),rne_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].legend(["predicition","ground_truth"])
    stdHrne = torch.std(H_rne).cpu().detach().numpy()
    meanHrne = torch.mean(H_rne).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_rnn.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHrne-3*stdHrne,meanHrne+3*stdHrne)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    plt.show()
    
    eval = evalset
    
    print("eval shape {}".format(evalset.shape))
    
    modelODE = torch.load("res_osci/ODEmodel.pt")
    print(modelODE) 
    ode_eval = odeint(modelODE,eval[0,:,:,:],t,method="rk4")
    print(ode_eval.shape)
    H_ode = datamaker.hamiltonian(ode_eval).cpu()
    print(H_ode.shape)
    H_mean=torch.mean(H_eval-H_ode,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_ode,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    #print(H_mean)
    
    
    plt.figure()
    plt.title("NeuralODE zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()


    modelMLP = torch.load("res_osci/MLPmodel.pt")
    print(modelMLP) 
    mlp_eval = DEVICETensor(len(t),3,1,2)
    mlp_eval[0,:,:,:]= eval[0,:,:,:]
    for i in range(1,len(t)):
        mlp_eval[i,:,:,:] = modelMLP(mlp_eval[i-1,:,:,:])
    print(mlp_eval.shape)
    H_mlp = datamaker.hamiltonian(mlp_eval).cpu()
    print(H_mlp.shape)
    
    H_mean=torch.mean(H_eval-H_mlp,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_mlp,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("MLP zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()

    
    
    
    modelGRU = torch.load("res_osci/GRUmodel.pt")
    print(modelGRU) 
    print(eval.shape)
    gru_eval = modelGRU(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(gru_eval.shape)
    H_gru = datamaker.hamiltonian(gru_eval).cpu()
    print(H_gru.shape)
    
    H_mean=torch.mean(H_eval-H_gru,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_gru,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("GRU hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    
    
    modelRNN= torch.load("res_osci/RNNmodel.pt")
    print(modelRNN) 
    rnn_eval = modelRNN(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(rnn_eval.shape)
    H_rnn = datamaker.hamiltonian(rnn_eval).cpu()
    print(H_rnn.shape)
    
    H_mean=torch.mean(H_eval-H_rnn,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_rnn,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("RNN hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    modelGRE= torch.load("res_osci/GREmodel.pt")
    print(modelGRE) 
    gre_eval = modelGRE(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(gre_eval.shape)
    H_gre = datamaker.hamiltonian(gre_eval).cpu()
    print(H_gre.shape)
    
    H_mean=torch.mean(H_eval-H_gre,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_gre,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("GRU Stepper hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    modelRNE= torch.load("res_osci/RNEmodel.pt")
    print(modelRNE) 
    rne_eval = modelRNE(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(rne_eval.shape)
    H_rne = datamaker.hamiltonian(rne_eval).cpu()
    print(H_rne.shape)
    
    H_mean=torch.mean(H_eval-H_rne,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_rne,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("RNN Stepper hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    
    


    
    
    
#def twobody_pics():
    datamaker = twobody(1,3,1)
    evalset = torch.load("res_twobody/eval.pt")
    print(evalset.shape)
    eval=evalset[:,0,:,:]
    r = torch.linalg.vector_norm(eval[0,0,0:2])
    T = datamaker.getT(r)
    t = torch.linspace(0,T,128)
    H = datamaker.hamiltonian(eval).cpu()
    H_eval = datamaker.hamiltonian(evalset).cpu()
    
    
    modelODE = torch.load("res_twobody/ODEmodel.pt")
    print(modelODE) 
    ode_eval = odeint(modelODE,eval[0,:,:],t,method="rk4")
    print(ode_eval.shape)
    H_ode = datamaker.hamiltonian(ode_eval).cpu()
    print(H_ode.shape)
    
    modelMLP = torch.load("res_twobody/MLPmodel.pt")
    print(modelMLP) 
    mlp_eval = DEVICETensor(len(t),1,8)
    mlp_eval[0,:,:]= eval[0,:,:]
    for i in range(1,len(t)):
        mlp_eval[i,:,:] = modelMLP(mlp_eval[i-1,:,:])
    print(mlp_eval.shape)
    H_mlp = datamaker.hamiltonian(mlp_eval).cpu()
    print(H_mlp.shape)
    
    modelGRU = torch.load("res_twobody/GRUmodel.pt")
    print(modelGRU) 
    gru_eval = modelGRU(eval.squeeze()[0,:].reshape(1,-1),t)
    print(ode_eval.shape)
    H_gru = datamaker.hamiltonian(gru_eval).cpu()
    print(H_gru.shape)
    
    modelRNN= torch.load("res_twobody/RNNmodel.pt")
    print(modelRNN) 
    rnn_eval = modelRNN(eval.squeeze()[0,:].reshape(1,-1),t)
    print(rnn_eval.shape)
    H_rnn = datamaker.hamiltonian(rnn_eval).cpu()
    print(H_rnn.shape)
    
    modelGRE= torch.load("res_twobody/GREmodel.pt")
    print(modelGRE) 
    gre_eval = modelGRE(eval.squeeze()[0,:].reshape(1,-1),t)
    print(gre_eval.shape)
    H_gre = datamaker.hamiltonian(gre_eval).cpu()
    print(H_gre.shape)
    
    modelRNE= torch.load("res_twobody/RNEmodel.pt")
    print(modelRNE) 
    rne_eval = modelRNE(eval.squeeze()[0,:].reshape(1,-1),t)
    print(rne_eval.shape)
    H_rne = datamaker.hamiltonian(rne_eval).cpu()
    print(H_rne.shape)
    
    losses = torch.load("res_twobody/losses.pt").cpu()
    
    print(losses.shape)
    epochs_t = torch.linspace(1,losses.shape[0],losses.shape[0])
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,0],c="b")
    ax[0].semilogy(epochs_t,losses[:,6],c="r")
    
    ax[1].plot(epochs_t,losses[:,12],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("MLP")
    #plt.show()
    
    
    
    
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,1],c="b")
    ax[0].semilogy(epochs_t,losses[:,7],c="r")
    
    ax[1].plot(epochs_t,losses[:,13],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("NeuralODE")
   #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,2],c="b")
    ax[0].semilogy(epochs_t,losses[:,8],c="r")
    
    ax[1].plot(epochs_t,losses[:,14],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("GRU")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,3],c="b")
    ax[0].semilogy(epochs_t,losses[:,9],c="r")
    
    ax[1].plot(epochs_t,losses[:,15],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("RNN")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,4],c="b")
    ax[0].semilogy(epochs_t,losses[:,10],c="r")
    
    ax[1].plot(epochs_t,losses[:,16],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("GRU Stepper")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,5],c="b")
    ax[0].semilogy(epochs_t,losses[:,11],c="r")
    
    ax[1].plot(epochs_t,losses[:,17],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("RNN Stepper")
    #plt.show() 
    
    fig, ax  = plt.subplots(2,2)
    fig.suptitle("MLP phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    
    ax[0,0].plot(mlp_eval[:,0,0].cpu().detach().numpy(),mlp_eval[:,0,4].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,4].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(mlp_eval[:,0,1].cpu().detach().numpy(),mlp_eval[:,0,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(mlp_eval[:,0,2].cpu().detach().numpy(),mlp_eval[:,0,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(mlp_eval[:,0,3].cpu().detach().numpy(),mlp_eval[:,0,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(2,2)
    fig.suptitle("NeuralODE phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    
    ax[0,0].plot(ode_eval[:,0,0].cpu().detach().numpy(),ode_eval[:,0,4].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,4].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(ode_eval[:,0,1].cpu().detach().numpy(),ode_eval[:,0,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(ode_eval[:,0,2].cpu().detach().numpy(),ode_eval[:,0,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(ode_eval[:,0,3].cpu().detach().numpy(),ode_eval[:,0,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(2,2)
    fig.suptitle("GRU phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    
    ax[0,0].plot(gru_eval[:,0].cpu().detach().numpy(),gru_eval[:,4].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,4].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(gru_eval[:,1].cpu().detach().numpy(),gru_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(gru_eval[:,2].cpu().detach().numpy(),gru_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(gru_eval[:,3].cpu().detach().numpy(),gru_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(2,2)
    fig.suptitle("RNN phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    
    ax[0,0].plot(rnn_eval[:,0].cpu().detach().numpy(),rnn_eval[:,4].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,4].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(rnn_eval[:,1].cpu().detach().numpy(),rnn_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(rnn_eval[:,2].cpu().detach().numpy(),rnn_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(rnn_eval[:,3].cpu().detach().numpy(),rnn_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    
    
    
    
    
    fig, ax  = plt.subplots(2,2)
    fig.suptitle("GRU Stepper phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    
    ax[0,0].plot(gre_eval[:,0].cpu().detach().numpy(),gre_eval[:,4].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,4].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(gre_eval[:,1].cpu().detach().numpy(),gre_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(gre_eval[:,2].cpu().detach().numpy(),gre_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(gre_eval[:,3].cpu().detach().numpy(),gre_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(2,2)
    fig.suptitle("RNN Stepper phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    
    ax[0,0].plot(rne_eval[:,0].cpu().detach().numpy(),rne_eval[:,4].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,4].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(rne_eval[:,1].cpu().detach().numpy(),rne_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(rne_eval[:,2].cpu().detach().numpy(),rne_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(rne_eval[:,3].cpu().detach().numpy(),rne_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("MLP trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(mlp_eval[:,0,0].cpu().detach().numpy(),mlp_eval[:,0,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(mlp_eval[:,0,2].cpu().detach().numpy(),mlp_eval[:,0,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHmlp = torch.std(H_mlp).cpu().detach().numpy()
    meanHmlp = torch.mean(H_mlp).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_mlp.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHmlp-3*stdHmlp,meanHmlp+3*stdHmlp)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("NeuralODE trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(ode_eval[:,0,0].cpu().detach().numpy(),ode_eval[:,0,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(ode_eval[:,0,2].cpu().detach().numpy(),ode_eval[:,0,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHode = torch.std(H_ode).cpu().detach().numpy()
    meanHode = torch.mean(H_ode).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_ode.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHode-3*stdHode,meanHode+3*stdHode)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("GRU trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(gru_eval[:,0].cpu().detach().numpy(),gru_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(gru_eval[:,2].cpu().detach().numpy(),gru_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHgru = torch.std(H_gru).cpu().detach().numpy()
    meanHgru = torch.mean(H_gru).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_gru.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHgru-3*stdHgru,meanHgru+3*stdHgru)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("RNN trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(rnn_eval[:,0].cpu().detach().numpy(),rnn_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(rnn_eval[:,2].cpu().detach().numpy(),rnn_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHrnn = torch.std(H_rnn).cpu().detach().numpy()
    meanHrnn = torch.mean(H_rnn).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_rnn.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHrnn-3*stdHrnn,meanHrnn+3*stdHrnn)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    
    
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("GRU Stepper trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(gre_eval[:,0].cpu().detach().numpy(),gre_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(gre_eval[:,2].cpu().detach().numpy(),gre_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHgre = torch.std(H_gre).cpu().detach().numpy()
    meanHgre = torch.mean(H_gre).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_gre.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHgre-3*stdHgre,meanHgre+3*stdHgre)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("RNN Stepper trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(rne_eval[:,0].cpu().detach().numpy(),rne_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(rne_eval[:,2].cpu().detach().numpy(),rne_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHrne = torch.std(H_rne).cpu().detach().numpy()
    meanHrne = torch.mean(H_rne).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_rne.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHrne-3*stdHrne,meanHrne+3*stdHrne)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    plt.show()
    
    
    
    
    eval = evalset
    
    print("eval shape {}".format(evalset.shape))
    
    modelODE = torch.load("res_twobody/ODEmodel.pt")
    print(modelODE) 
    ode_eval = odeint(modelODE,eval[0,:,:,:],t,method="rk4")
    print(ode_eval.shape)
    H_ode = datamaker.hamiltonian(ode_eval).cpu()
    print(H_ode.shape)
    H_mean=torch.mean(H_eval-H_ode,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_ode,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    #print(H_mean)
    
    
    plt.figure()
    plt.title("NeuralODE zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()


    modelMLP = torch.load("res_twobody/MLPmodel.pt")
    print(modelMLP) 
    mlp_eval = DEVICETensor(len(t),3,1,8)
    mlp_eval[0,:,:,:]= eval[0,:,:,:]
    for i in range(1,len(t)):
        mlp_eval[i,:,:,:] = modelMLP(mlp_eval[i-1,:,:,:])
    print(mlp_eval.shape)
    H_mlp = datamaker.hamiltonian(mlp_eval).cpu()
    print(H_mlp.shape)
    
    H_mean=torch.mean(H_eval-H_mlp,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_mlp,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("MLP zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()

    
    
    
    modelGRU = torch.load("res_twobody/GRUmodel.pt")
    print(modelGRU) 
    print(eval.shape)
    gru_eval = modelGRU(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(gru_eval.shape)
    H_gru = datamaker.hamiltonian(gru_eval).cpu()
    print(H_gru.shape)
    
    H_mean=torch.mean(H_eval-H_gru,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_gru,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("GRU hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    
    
    modelRNN= torch.load("res_twobody/RNNmodel.pt")
    print(modelRNN) 
    rnn_eval = modelRNN(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(rnn_eval.shape)
    H_rnn = datamaker.hamiltonian(rnn_eval).cpu()
    print(H_rnn.shape)
    
    H_mean=torch.mean(H_eval-H_rnn,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_rnn,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("RNN hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    modelGRE= torch.load("res_twobody/GREmodel.pt")
    print(modelGRE) 
    gre_eval = modelGRE(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(gre_eval.shape)
    H_gre = datamaker.hamiltonian(gre_eval).cpu()
    print(H_gre.shape)
    
    H_mean=torch.mean(H_eval-H_gre,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_gre,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("GRU Stepper hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    modelRNE= torch.load("res_twobody/RNEmodel.pt")
    print(modelRNE) 
    rne_eval = modelRNE(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(rne_eval.shape)
    H_rne = datamaker.hamiltonian(rne_eval).cpu()
    print(H_rne.shape)
    
    H_mean=torch.mean(H_eval-H_rne,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_rne,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean))
    
    plt.figure()
    plt.title("RNN Stepper hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    
    
#def threebody_pics():
    datamaker = threebody(SAMPLES)
    evalset = torch.load("res_threebody/eval.pt")
    print("evalset size {}".format(evalset.shape))
    eval = evalset[:,0,:,:]
    H = datamaker.hamiltonian(eval)
    T = SAMPLES["fig8"]["T"]
    t = torch.linspace(0,T,128)
    H_eval = datamaker.hamiltonian(evalset)
    modelODE = torch.load("res_threebody/ODEmodel.pt")
    print(modelODE) 
    ode_eval = odeint(modelODE,eval[0,:,:],t,method="rk4")
    print(ode_eval.shape)
    H_ode = datamaker.hamiltonian(ode_eval).cpu()
    print(H_ode.shape)
    
    modelMLP = torch.load("res_threebody/MLPmodel.pt")
    print(modelMLP) 
    mlp_eval = DEVICETensor(len(t),1,12)
    mlp_eval[0,:,:]= eval[0,:,:]
    for i in range(1,len(t)):
        mlp_eval[i,:,:] = modelMLP(mlp_eval[i-1,:,:])
    print(mlp_eval.shape)
    H_mlp = datamaker.hamiltonian(mlp_eval).cpu()
    print(H_mlp.shape)
    
    modelGRU = torch.load("res_threebody/GRUmodel.pt")
    print(modelGRU) 
    gru_eval = modelGRU(eval.squeeze()[0,:].reshape(1,-1),t)
    print(ode_eval.shape)
    H_gru = datamaker.hamiltonian(gru_eval).cpu()
    print(H_gru.shape)
    
    modelRNN= torch.load("res_threebody/RNNmodel.pt")
    print(modelRNN) 
    rnn_eval = modelRNN(eval.squeeze()[0,:].reshape(1,-1),t)
    print(rnn_eval.shape)
    H_rnn = datamaker.hamiltonian(rnn_eval).cpu()
    print(H_rnn.shape)
    
    modelGRE= torch.load("res_threebody/GREmodel.pt")
    print(modelGRE) 
    gre_eval = modelGRE(eval.squeeze()[0,:].reshape(1,-1),t)
    print(gre_eval.shape)
    H_gre = datamaker.hamiltonian(gre_eval).cpu()
    print(H_gre.shape)
    
    modelRNE= torch.load("res_threebody/RNEmodel.pt")
    print(modelRNE) 
    rne_eval = modelRNE(eval.squeeze()[0,:].reshape(1,-1),t)
    print(rne_eval.shape)
    H_rne = datamaker.hamiltonian(rne_eval).cpu()
    print(H_rne.shape)
    
    losses = torch.load("res_threebody/losses.pt").cpu()
    
    print(losses.shape)
    epochs_t = torch.linspace(1,losses.shape[0],losses.shape[0]) 
    

    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,0],c="b")
    ax[0].semilogy(epochs_t,losses[:,6],c="r")
    
    ax[1].plot(epochs_t,losses[:,12],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("MLP")
    #plt.show()
    
    
    
    
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,1],c="b")
    ax[0].semilogy(epochs_t,losses[:,7],c="r")
    
    ax[1].plot(epochs_t,losses[:,13],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("NeuralODE")
   #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,2],c="b")
    ax[0].semilogy(epochs_t,losses[:,8],c="r")
    
    ax[1].plot(epochs_t,losses[:,14],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("GRU")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,3],c="b")
    ax[0].semilogy(epochs_t,losses[:,9],c="r")
    
    ax[1].plot(epochs_t,losses[:,15],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("RNN")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,4],c="b")
    ax[0].semilogy(epochs_t,losses[:,10],c="r")
    
    ax[1].plot(epochs_t,losses[:,16],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("GRU Stepper")
    #plt.show()
    
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Train/Test Loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].set_title("Energy accuracy")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    
    
    ax[0].semilogy(epochs_t,losses[:,5],c="b")
    ax[0].semilogy(epochs_t,losses[:,11],c="r")
    
    ax[1].plot(epochs_t,losses[:,17],c="b")
    ax[0].legend(["train loss","test_loss"])
    fig.suptitle("RNN Stepper")
    
    fig, ax  = plt.subplots(3,2)
    fig.suptitle("MLP phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    ax[2,0].set_title("body3 x")
    ax[2,0].set_xlabel("q")
    ax[2,0].set_ylabel("p")
    ax[2,1].set_title("body3 y")
    ax[2,1].set_xlabel("q")
    ax[2,1].set_ylabel("p")
    
    ax[0,0].plot(mlp_eval[:,0,0].cpu().detach().numpy(),mlp_eval[:,0,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(mlp_eval[:,0,1].cpu().detach().numpy(),mlp_eval[:,0,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(mlp_eval[:,0,2].cpu().detach().numpy(),mlp_eval[:,0,8].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,8].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(mlp_eval[:,0,3].cpu().detach().numpy(),mlp_eval[:,0,9].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,9].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    ax[2,0].plot(mlp_eval[:,0,4].cpu().detach().numpy(),mlp_eval[:,0,10].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,10].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,0].legend(["predicition","ground_truth"])
    ax[2,1].plot(mlp_eval[:,0,5].cpu().detach().numpy(),mlp_eval[:,0,11].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,1].plot(eval[:,0,5].cpu().detach().numpy(),eval[:,0,11].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,1].legend(["predicition","ground_truth"])
    
    
    
    fig, ax  = plt.subplots(3,2)
    fig.suptitle("NeuralODE phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    ax[2,0].set_title("body3 x")
    ax[2,0].set_xlabel("q")
    ax[2,0].set_ylabel("p")
    ax[2,1].set_title("body3 y")
    ax[2,1].set_xlabel("q")
    ax[2,1].set_ylabel("p")
    
    ax[0,0].plot(ode_eval[:,0,0].cpu().detach().numpy(),ode_eval[:,0,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(ode_eval[:,0,1].cpu().detach().numpy(),ode_eval[:,0,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(ode_eval[:,0,2].cpu().detach().numpy(),ode_eval[:,0,8].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,8].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(ode_eval[:,0,3].cpu().detach().numpy(),ode_eval[:,0,9].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,9].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    ax[2,0].plot(ode_eval[:,0,4].cpu().detach().numpy(),ode_eval[:,0,10].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,10].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,0].legend(["predicition","ground_truth"])
    ax[2,1].plot(ode_eval[:,0,5].cpu().detach().numpy(),ode_eval[:,0,11].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,1].plot(eval[:,0,5].cpu().detach().numpy(),eval[:,0,11].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(3,2)
    fig.suptitle("GRU phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    ax[2,0].set_title("body3 x")
    ax[2,0].set_xlabel("q")
    ax[2,0].set_ylabel("p")
    ax[2,1].set_title("body3 y")
    ax[2,1].set_xlabel("q")
    ax[2,1].set_ylabel("p")
    
    ax[0,0].plot(gru_eval[:,0].cpu().detach().numpy(),gru_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(gru_eval[:,1].cpu().detach().numpy(),gru_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(gru_eval[:,2].cpu().detach().numpy(),gru_eval[:,8].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,8].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(gru_eval[:,3].cpu().detach().numpy(),gru_eval[:,9].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,9].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    ax[2,0].plot(gru_eval[:,4].cpu().detach().numpy(),gru_eval[:,10].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,10].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,0].legend(["predicition","ground_truth"])
    ax[2,1].plot(gru_eval[:,5].cpu().detach().numpy(),gru_eval[:,11].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,1].plot(eval[:,0,5].cpu().detach().numpy(),eval[:,0,11].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(3,2)
    fig.suptitle("RNN phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    ax[2,0].set_title("body3 x")
    ax[2,0].set_xlabel("q")
    ax[2,0].set_ylabel("p")
    ax[2,1].set_title("body3 y")
    ax[2,1].set_xlabel("q")
    ax[2,1].set_ylabel("p")
    
    ax[0,0].plot(rnn_eval[:,0].cpu().detach().numpy(),rnn_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(rnn_eval[:,1].cpu().detach().numpy(),rnn_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(rnn_eval[:,2].cpu().detach().numpy(),rnn_eval[:,8].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,8].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(rnn_eval[:,3].cpu().detach().numpy(),rnn_eval[:,9].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,9].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    ax[2,0].plot(rnn_eval[:,4].cpu().detach().numpy(),rnn_eval[:,10].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,10].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,0].legend(["predicition","ground_truth"])
    ax[2,1].plot(rnn_eval[:,5].cpu().detach().numpy(),rnn_eval[:,11].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,1].plot(eval[:,0,5].cpu().detach().numpy(),eval[:,0,11].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(3,2)
    fig.suptitle("GRU Stepper phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    ax[2,0].set_title("body3 x")
    ax[2,0].set_xlabel("q")
    ax[2,0].set_ylabel("p")
    ax[2,1].set_title("body3 y")
    ax[2,1].set_xlabel("q")
    ax[2,1].set_ylabel("p")
    
    ax[0,0].plot(gre_eval[:,0].cpu().detach().numpy(),gre_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(gre_eval[:,1].cpu().detach().numpy(),gre_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(gre_eval[:,2].cpu().detach().numpy(),gre_eval[:,8].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,8].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(gre_eval[:,3].cpu().detach().numpy(),gre_eval[:,9].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,9].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    ax[2,0].plot(gre_eval[:,4].cpu().detach().numpy(),gre_eval[:,10].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,10].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,0].legend(["predicition","ground_truth"])
    ax[2,1].plot(gre_eval[:,5].cpu().detach().numpy(),gre_eval[:,11].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,1].plot(eval[:,0,5].cpu().detach().numpy(),eval[:,0,11].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(3,2)
    fig.suptitle("RNN Stepper phase spaces")
    ax[0,0].set_title("body1 x")
    ax[0,0].set_xlabel("q")
    ax[0,0].set_ylabel("p")
    ax[0,1].set_title("body1 y")
    ax[0,1].set_xlabel("q")
    ax[0,1].set_ylabel("p")
    ax[1,0].set_title("body2 x")
    ax[1,0].set_xlabel("q")
    ax[1,0].set_ylabel("p")
    ax[1,1].set_title("body2 y")
    ax[1,1].set_xlabel("q")
    ax[1,1].set_ylabel("p")
    ax[2,0].set_title("body3 x")
    ax[2,0].set_xlabel("q")
    ax[2,0].set_ylabel("p")
    ax[2,1].set_title("body3 y")
    ax[2,1].set_xlabel("q")
    ax[2,1].set_ylabel("p")
    
    ax[0,0].plot(rne_eval[:,0].cpu().detach().numpy(),rne_eval[:,6].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,6].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,0].legend(["predicition","ground_truth"])
    ax[0,1].plot(rne_eval[:,1].cpu().detach().numpy(),rne_eval[:,7].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0,1].plot(eval[:,0,1].cpu().detach().numpy(),eval[:,0,7].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0,1].legend(["predicition","ground_truth"])
    ax[1,0].plot(rne_eval[:,2].cpu().detach().numpy(),rne_eval[:,8].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,8].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,0].legend(["predicition","ground_truth"])
    ax[1,1].plot(rne_eval[:,3].cpu().detach().numpy(),rne_eval[:,9].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[1,1].plot(eval[:,0,3].cpu().detach().numpy(),eval[:,0,9].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[1,1].legend(["predicition","ground_truth"])
    ax[2,0].plot(rne_eval[:,4].cpu().detach().numpy(),rne_eval[:,10].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,10].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,0].legend(["predicition","ground_truth"])
    ax[2,1].plot(rne_eval[:,5].cpu().detach().numpy(),rne_eval[:,11].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[2,1].plot(eval[:,0,5].cpu().detach().numpy(),eval[:,0,11].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[2,1].legend(["predicition","ground_truth"])
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("MLP trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(mlp_eval[:,0,0].cpu().detach().numpy(),mlp_eval[:,0,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(mlp_eval[:,0,2].cpu().detach().numpy(),mlp_eval[:,0,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(mlp_eval[:,0,4].cpu().detach().numpy(),mlp_eval[:,0,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHmlp = torch.std(H_mlp).cpu().detach().numpy()
    meanHmlp = torch.mean(H_mlp).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_mlp.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHmlp-3*stdHmlp,meanHmlp+3*stdHmlp)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("NeuralODE trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(ode_eval[:,0,0].cpu().detach().numpy(),ode_eval[:,0,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(ode_eval[:,0,2].cpu().detach().numpy(),ode_eval[:,0,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(ode_eval[:,0,4].cpu().detach().numpy(),ode_eval[:,0,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    
    ax[0].legend(["predicition","ground_truth"])
    stdHode = torch.std(H_ode).cpu().detach().numpy()
    meanHode = torch.mean(H_ode).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_ode.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHode-3*stdHode,meanHode+3*stdHode)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("GRU trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(gru_eval[:,0].cpu().detach().numpy(),gru_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(gru_eval[:,2].cpu().detach().numpy(),gru_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(gru_eval[:,4].cpu().detach().numpy(),gru_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    
    ax[0].legend(["predicition","ground_truth"])
    stdHgru = torch.std(H_gru).cpu().detach().numpy()
    meanHgru = torch.mean(H_gru).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_gru.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHgru-3*stdHgru,meanHgru+3*stdHgru)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("RNN trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(rnn_eval[:,0].cpu().detach().numpy(),rnn_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(rnn_eval[:,2].cpu().detach().numpy(),rnn_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(rnn_eval[:,4].cpu().detach().numpy(),rnn_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHrnn = torch.std(H_rnn).cpu().detach().numpy()
    meanHrnn = torch.mean(H_rnn).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_rnn.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHrnn-3*stdHrnn,meanHrnn+3*stdHrnn)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    
    
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("GRU Stepper trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(gre_eval[:,0].cpu().detach().numpy(),gre_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(gre_eval[:,2].cpu().detach().numpy(),gre_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(gre_eval[:,4].cpu().detach().numpy(),gre_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHgre = torch.std(H_gre).cpu().detach().numpy()
    meanHgre = torch.mean(H_gre).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_gre.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHgre-3*stdHgre,meanHgre+3*stdHgre)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    
    fig, ax  = plt.subplots(1,2)
    fig.suptitle("RNN Stepper trajectories and Energy")
    ax[0].set_title("trajectory")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Total Energy")
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("time")
    
    ax[0].plot(rne_eval[:,0].cpu().detach().numpy(),rne_eval[:,1].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,0].cpu().detach().numpy(),eval[:,0,1].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(rne_eval[:,2].cpu().detach().numpy(),rne_eval[:,3].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,2].cpu().detach().numpy(),eval[:,0,3].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    ax[0].plot(rne_eval[:,4].cpu().detach().numpy(),rne_eval[:,5].cpu().detach().numpy(), c="r",linewidth=2.0)
    ax[0].plot(eval[:,0,4].cpu().detach().numpy(),eval[:,0,5].cpu().detach().numpy(), c="b",linewidth=4.0,alpha=0.2)
    
    ax[0].legend(["predicition","ground_truth"])
    stdHrne = torch.std(H_rne).cpu().detach().numpy()
    meanHrne = torch.mean(H_rne).cpu().detach().numpy()
    ax[1].plot(t.cpu().detach().numpy(),H_rne.cpu().detach().numpy(),linewidth=2.0)
    ax[1].plot(t.cpu().detach().numpy(),H.cpu().detach().numpy(),linewidth=2.0)
    #ax[1].legend(["mean: {}".format(meanHmlp),"std: {}".format(stdHmlp)])
    ax[1].set_ylim(meanHrne-3*stdHrne,meanHrne+3*stdHrne)
    ax[1].set_xlabel("time")
    ax[1].legend(["prediction","real"])
    ax[1].set_ylabel("H")
    plt.show()
    
    
    eval = evalset
    
    print("eval shape {}".format(evalset.shape))
    
    modelODE = torch.load("res_threebody/ODEmodel.pt")
    print(modelODE) 
    ode_eval = odeint(modelODE,eval[0,:,:,:],t,method="rk4")
    print(ode_eval.shape)
    H_ode = datamaker.hamiltonian(ode_eval).cpu()
    print(H_ode.shape)
    H_eval = H_eval.cpu()
    H_mean=torch.mean(H_eval-H_ode,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_ode,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean.shape))
    #print(H_mean)
    
    
    plt.figure()
    plt.title("NeuralODE zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()


    modelMLP = torch.load("res_threebody/MLPmodel.pt")
    print(modelMLP) 
    mlp_eval = DEVICETensor(len(t),3,1,12)
    mlp_eval[0,:,:,:]= eval[0,:,:,:]
    for i in range(1,len(t)):
        mlp_eval[i,:,:,:] = modelMLP(mlp_eval[i-1,:,:,:])
    print(mlp_eval.shape)
    H_mlp = datamaker.hamiltonian(mlp_eval).cpu()
    print(H_mlp.shape)
    
    H_mean=torch.mean(H_eval-H_mlp,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_mlp,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean.shape))
    
    plt.figure()
    plt.title("MLP zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()

    
    
    
    modelGRU = torch.load("res_threebody/GRUmodel.pt")
    print(modelGRU) 
    print(eval.shape)
    gru_eval = modelGRU(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(gru_eval.shape)
    H_gru = datamaker.hamiltonian(gru_eval).cpu()
    print(H_gru.shape)
    
    H_mean=torch.mean(H_eval-H_gru,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_gru,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean.shape))
    
    plt.figure()
    plt.title("GRU hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    
    
    modelRNN= torch.load("res_threebody/RNNmodel.pt")
    print(modelRNN) 
    rnn_eval = modelRNN(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(rnn_eval.shape)
    H_rnn = datamaker.hamiltonian(rnn_eval).cpu()
    print(H_rnn.shape)
    
    H_mean=torch.mean(H_eval-H_rnn,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_rnn,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean.shape))
    
    plt.figure()
    plt.title("RNN hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    modelGRE= torch.load("res_threebody/GREmodel.pt")
    print(modelGRE) 
    gre_eval = modelGRE(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(gre_eval.shape)
    H_gre = datamaker.hamiltonian(gre_eval).cpu()
    print(H_gre.shape)
    
    H_mean=torch.mean(H_eval-H_gre,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_gre,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean.shape))
    
    plt.figure()
    plt.title("GRU Stepper hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
    
    modelRNE= torch.load("res_threebody/RNEmodel.pt")
    print(modelRNE) 
    rne_eval = modelRNE(eval.squeeze()[0,:,:].reshape(1,3,-1),t)
    print(rne_eval.shape)
    H_rne = datamaker.hamiltonian(rne_eval).cpu()
    print(H_rne.shape)
    
    H_mean=torch.mean(H_eval-H_rne,dim=1).detach().numpy()
    H_std =torch.std(H_eval-H_rne,dim=1).detach().numpy()
    print("mean shape {}".format(H_mean.shape))
    
    plt.figure()
    plt.title("RNN Stepper hamiltonian zero mean/std plot")
    plt.plot(t,H_mean)
    plt.fill_between(t,H_mean - H_std,H_mean+ H_std,alpha = 0.2)
    plt.show()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    


osci_pics()
#twobody_pics()
#threebody_pics()