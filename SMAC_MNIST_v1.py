
import numpy as np
import torch
import torch.nn as nn

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt

import time

from smac import HyperparameterOptimizationFacade, Scenario

from ConfigSpace import Configuration, ConfigurationSpace, Float


def get_optimizer(model, lr, momentum):
    optimizer_class = optim.SGD
    return optimizer_class(model.parameters(), lr=lr, momentum=momentum)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
    
class Worker():
    def __init__(self, i, train_data, test_data, lr, momentum, device):
        super().__init__()
        self.i = i
        self.device = device
        self.hyperparam = [lr, momentum]
        self.best_hyperparam = self.hyperparam
        
        self.score = -1.0
        
        self.lr = self.hyperparam[0]
        self.momentum = self.hyperparam[1]
        self.batch_size = 20
        
        model = Net().to(device)
        optimizer = get_optimizer(model, self.lr, self.momentum)
        self.trainer = Trainer(model=model,
                                optimizer=optimizer,
                                loss_fn=nn.CrossEntropyLoss(),
                                train_data=train_data,
                                test_data=test_data,
                                batch_size=self.batch_size,
                                device=self.device)


    def set_optimizer(self, hyperparam):
        model = Net().to(device)
        self.hyperparam = hyperparam
        
        
        self.lr = self.hyperparam[0]
        self.momentum = self.hyperparam[1]

        optimizer = get_optimizer(model, self.lr, self.momentum)
        self.trainer = Trainer(model=model,
                                optimizer=optimizer,
                                loss_fn=nn.CrossEntropyLoss(),
                                train_data=train_data,
                                test_data=test_data,
                                batch_size=self.batch_size,
                                device=self.device)
        

    def run(self):
        self.trainer.train()
        score = self.trainer.eval()
        if score > self.score:
            self.score = score
            self.best_hyperparam = self.hyperparam
        return score



class Trainer:

    def __init__(self, model, optimizer, loss_fn=None, train_data=None,
                  test_data=None, batch_size=None, device=None):        

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.task_id = None
        self.device = device

    def set_id(self, num):
        self.task_id = num

    def train(self):
        self.model.train()
        dataloader = tqdm.tqdm(DataLoader(self.train_data, self.batch_size, True),
                                desc='Train (task {})'.format(self.task_id),
                                ncols=80, leave=True)
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self):        
        self.model.eval()
        dataloader = tqdm.tqdm(DataLoader(self.test_data, self.batch_size, True),
                                desc='Eval (task {})'.format(self.task_id),
                                ncols=80, leave=True)
        correct = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = output.argmax(1)
            correct += pred.eq(y).sum().item()
        accuracy = 100. * correct / (len(dataloader) * self.batch_size)
        print(accuracy)
        return accuracy



def train(config: Configuration, seed: int = 0) -> float:
    w = Worker(0, train_data, test_data, config["LR"], config["MM"], device)
    scores = w.run()
    return 1/(np.mean(scores)+1)


    

if __name__ == "__main__":    
    device = "cuda"
    if not torch.cuda.is_available():
        device = 'cpu'       
    print("device: ", device)
    
    inicio = time.time()        
    
    train_data_path = test_data_path = './data'
    
    train_data = FashionMNIST(train_data_path, True, transforms.ToTensor(), download=True)
    test_data = FashionMNIST(test_data_path, False, transforms.ToTensor(), download=True)
    
    
    lr_min = 0.0001
    lr_max =0.9999
    
    mm_min = 0.0001
    mm_max = 0.9999
    

    configspace = ConfigurationSpace({"LR": (lr_min, lr_max), "MM": (mm_min, mm_max)})
    
    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=250)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train)


    incumbent = smac.optimize()


    fin = time.time()
    print("execution time: ", fin-inicio)
    
    

    #print el espacio de parametros y best    
    space_lr = np.linspace(lr_min, lr_max)    
    space_momentum = np.linspace(mm_min, mm_max)
           

    print(incumbent['LR'], incumbent['MM'])
    
    
    #realizamos 10 pruebas con la configuracion obtenida
    n_test = 10
    print(incumbent['LR'])
    print(incumbent['MM'])

    res_scores = []
    for i in range(n_test):
        w = Worker(0, train_data, test_data, incumbent['LR'], incumbent['MM'], device)
        scores = w.run()
        print(scores)
        res_scores.append(scores)
        
        
    fig, ax = plt.subplots()
    ax.boxplot(res_scores)     
    plt.show()            
    print(res_scores)
    
        
    fig, ax = plt.subplots()
    
    ax.scatter(incumbent['LR'], incumbent['MM'])   
    ax.annotate(scores, (incumbent['LR'], incumbent['MM']))
    
    plt.xlim([np.min(space_lr), np.max(space_lr)])
    plt.ylim([np.min(space_momentum), np.max(space_momentum)])
    
    plt.title("espacio de hiperparámetros")
    plt.xlabel("Learning Rate")
    plt.ylabel("Momentum")   
    
       
    plt.grid()
    plt.show()
