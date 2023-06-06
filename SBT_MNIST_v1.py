
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
import torch.nn.functional as F
import torch.optim as optim
import random
import time

#from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import Problem

from matplotlib import pyplot as plt


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
    def __init__(self, ident, train_data, test_data, lr, momentum, device):
        super().__init__()
        self.ident = ident
        
        self.device = device
        self.hyperparam = [lr, momentum]
        self.best_hyperparam = self.hyperparam
        
        self.score = -1.0
        
        self.lr = self.hyperparam[0]
        self.momentum = self.hyperparam[1]
        self.batch_size = 20
        
        self.model = Net().to(device)
        optimizer = get_optimizer(self.model, self.lr, self.momentum)
        self.trainer = Trainer(model=self.model,
                                optimizer=optimizer,
                                loss_fn=nn.CrossEntropyLoss(),
                                train_data=train_data,
                                test_data=test_data,
                                batch_size=self.batch_size,
                                device=self.device)


    def set_optimizer(self, hyperparam):
        self.model = Net().to(device)
        self.hyperparam = hyperparam
        
        
        self.lr = self.hyperparam[0]
        self.momentum = self.hyperparam[1]

        optimizer = get_optimizer(self.model, self.lr, self.momentum)
        self.trainer = Trainer(model=self.model,
                                optimizer=optimizer,
                                loss_fn=nn.CrossEntropyLoss(),
                                train_data=train_data,
                                test_data=test_data,
                                batch_size=self.batch_size,
                                device=self.device)
        

    def run(self):
        self.trainer.train()
        score = self.trainer.eval()
        self.score = score
        #if score > self.score:
        #    self.score = score
        #    self.best_hyperparam = self.hyperparam
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


def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, w_max=1.0, c=0.1):
  # Initialise new velocity array
  num_particle = len(particle)
  new_velocity = np.array([0.0 for i in range(num_particle)])
  # Randomly generate r1, r2 and inertia weight from normal distribution
  r1 = random.uniform(0,w_max)
  r2 = random.uniform(0,w_max)
  w = random.uniform(w_min,w_max)
  c1 = c
  c2 = c
  # Calculate new velocity
  for i in range(num_particle):
    new_velocity[i] = w*velocity[i] + c1*r1*(pbest[i]-particle[i])+c2*r2*(gbest[i]-particle[i])
  return new_velocity

def update_position(particle, velocity):
  # Move particles by adding velocity
  new_particle = abs(particle + velocity)
  return new_particle


if __name__ == "__main__":    
    device = "cuda"
    if not torch.cuda.is_available():
        device = 'cpu'
       
    print("device: ", device)
    
    inicio = time.time()   
    
    generation = 10
    population_size = 10
    dimension = 2
    
    lr_min = 0.0001
    lr_max =0.9999
    
    mm_min = 0.0001
    mm_max = 0.9999
    
    space_lr = np.linspace(lr_min, lr_max)    
    space_momentum = np.linspace(mm_min, mm_max)
    
    
    #hyperparams space
    hyperparams_space = [space_lr, space_momentum]    
    print(hyperparams_space)    

    plt.xlim([np.min(space_lr), np.max(space_lr)])
    plt.ylim([np.min(space_momentum), np.max(space_momentum)])
       
       
    plt.title("espacio de hiperparámetros")
    plt.xlabel("Learning Rate")
    plt.ylabel("Momentum")
       
    plt.grid()
    plt.show()
             

    problem = Problem(n_var=2, xl=[np.min(space_lr), np.min(space_momentum)], xu=[np.max(space_lr), np.max(space_momentum)])
    sampling = FloatRandomSampling()
    
    X = sampling(problem, population_size).get("X")
    
    particles = []
    for i in range(population_size):        
        particles.append( [ X[i][0], X[i][1] ] )
        
    print(particles)
    train_data_path = test_data_path = './data'
    
    train_data = FashionMNIST(train_data_path, True, transforms.ToTensor(), download=True)
    test_data = FashionMNIST(test_data_path, False, transforms.ToTensor(), download=True)
    
    
    workers = [Worker(i, train_data, test_data, particles[i][0], particles[i][1], device) for i in range(population_size)]
    
    pbest_position = particles
    
    
    #graph de muestreo inicial
    muestra_lr = []
    muestra_momentum = []
    n = []
    for w in workers:
        muestra_lr.append(w.hyperparam[0])
        muestra_momentum.append(w.hyperparam[1])
        n.append(w.ident)
        
        
    fig, ax = plt.subplots()
    
    plt.xlim([np.min(space_lr), np.max(space_lr)])
    plt.ylim([np.min(space_momentum), np.max(space_momentum)])
        
     
    ax.scatter(muestra_lr, muestra_momentum)
    for i, txt in enumerate(n):
        ax.annotate(txt, (muestra_lr[i], muestra_momentum[i]))
    

    plt.title("espacio de hiperparámetros")
    plt.xlabel("Learning Rate")
    plt.ylabel("Momentum")   
       
       
    plt.grid()
    plt.show()
    
    #PSO
    pbest_fitness = []
    for i in range(population_size):
        pbest_fitness.append(workers[i].run())
        
        
    print(pbest_fitness)  
    
    gbest_index = np.argmax(pbest_fitness)
    # Global best particle position
    gbest_position = pbest_position[gbest_index]
    # Velocity (starting from 0 speed)
    velocity = [[0.0 for j in range(dimension)] for i in range(population_size)]
    
    
    result=[]
    # Loop for the number of generation
    for t in range(generation):        
        #calcular la velocidad y posicion de la particula
        for n in range(population_size):
            # Update the velocity of each particle
            velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
            # Move the particles to new position
            particles[n] = update_position(particles[n], velocity[n])
            
            
        
        # Calculate the fitness value
        pbest_fitness = []
        for i in range(population_size):
            workers[i].set_optimizer([particles[i][0], particles[i][1]])
            
            w = random.uniform(0.5, 1)
             
            workers[i].model.conv1.weight.data = torch.add(
                torch.add(workers[i].model.conv1.weight.data, w*np.mean(velocity[i])), 
                torch.add(workers[gbest_index].model.conv1.weight.data, -1*workers[i].model.conv1.weight.data)
                )
            
            workers[i].model.conv2.weight.data = torch.add(
                torch.add(workers[i].model.conv2.weight.data, w*np.mean(velocity[i])), 
                torch.add(workers[gbest_index].model.conv2.weight.data, -1*workers[i].model.conv2.weight.data)
                )
            
            workers[i].model.fc1.weight.data = torch.add(
                torch.add(workers[i].model.fc1.weight.data, w*np.mean(velocity[i])), 
                torch.add(workers[gbest_index].model.fc1.weight.data, -1*workers[i].model.fc1.weight.data)
                )
            
            workers[i].model.fc2.weight.data = torch.add(
                torch.add(workers[i].model.fc2.weight.data, w*np.mean(velocity[i])), 
                torch.add(workers[gbest_index].model.fc2.weight.data, -1*workers[i].model.fc2.weight.data)
                )
            
   
            pbest_fitness.append(workers[i].run())
            
       
        result.append(pbest_fitness)
        

        # Find the index of the best particle
        gbest_index = np.argmax(pbest_fitness)
        # Update the position of the best particle
        gbest_position = pbest_position[gbest_index]
        
        
     
    fin = time.time()
    print("execution time: ", fin-inicio)
           
    # Print the results
    print('Global Best Position: ', gbest_position)
    print('Best Fitness Value: ', max(pbest_fitness))
    print('Average Particle Best Fitness Value: ', np.average(pbest_fitness))
    print('Number of Generation: ', t)
    
    
    print(pbest_fitness)
    print(result)
    print(result[1])
    
    data_0 = []
    for i in range(population_size):
        data_1 = []
        for e in range(generation):
            data_1.append(result[e][i])
            
        data_0.append(data_1)    
    
    print(data_0)
    
    my_dict={}
    for i in range(population_size):
        my_dict[i]=data_0[i]

        
    print(my_dict)
    
    #graph resultados    
    #SERIES DE TIEMPO POR GENERACIONES
    fig, ax = plt.subplots()
    for i in range(population_size):
        ax.plot(data_0[i], label=i)            
    ax.set_title('Individuos por generación')
    ax.legend(loc="upper left")
    plt.show()
    
    
    
        
    #BOXPLOT
    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.show()
    
    
    
    
    #graph resultados
    muestra_lr = []
    muestra_momentum = []
    n = []
    for w in workers:
        muestra_lr.append(w.hyperparam[0])
        muestra_momentum.append(w.hyperparam[1])
        n.append(w.ident)
        
    fig, ax = plt.subplots()
    #ax.scatter(space_lr, space_momentum)
    ax.scatter(muestra_lr, muestra_momentum)
    for i, txt in enumerate(n):
        ax.annotate(txt, (muestra_lr[i], muestra_momentum[i]))
    
    plt.xlim([np.min(space_lr), np.max(space_lr)])
    plt.ylim([np.min(space_momentum), np.max(space_momentum)])
     
    plt.title("espacio de hiperparámetros")
    plt.xlabel("Learning Rate")
    plt.ylabel("Momentum")   
    
    plt.grid()
    plt.show()
    
    
    
    #graph primeros resultados
    muestra_lr = []
    muestra_momentum = []
    n = []
    for w in workers:
        muestra_lr.append(w.hyperparam[0])
        muestra_momentum.append(w.hyperparam[1])
        n.append(w.score)
        
    fig, ax = plt.subplots()
    #ax.scatter(space_lr, space_momentum)
    ax.scatter(muestra_lr, muestra_momentum)
    for i, txt in enumerate(n):
        ax.annotate(txt, (muestra_lr[i], muestra_momentum[i]))
    
    plt.xlim([np.min(space_lr), np.max(space_lr)])
    plt.ylim([np.min(space_momentum), np.max(space_momentum)])
     
    plt.title("espacio de hiperparámetros")
    plt.xlabel("Learning Rate")
    plt.ylabel("Momentum")   
    
    plt.grid()
    plt.show()
    
    
        
    
    #realizamos 5 pruebas con la configuracion obtenida
    n_test = 10
    print(gbest_position[0])
    print(gbest_position[1])
    
    res_scores = []
    for i in range(n_test):
        w = Worker(0, train_data, test_data, gbest_position[0], gbest_position[1], device)
        scores = w.run()
        print(scores)
        res_scores.append(scores)
        
        
    fig, ax = plt.subplots()
    ax.boxplot(res_scores)     
    plt.show()            
    print(res_scores)
    