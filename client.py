
import copy
import torch
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from FedSR_pers import *
from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, cls=None, net_model=None, test_client=False):
        self.args = args
        self.dataset = dataset
        # da decommentare quando usi femnist
        self.name = self.dataset.client_name
        # colpevole di saturare la ram in niid!
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean') # per ora userei questo
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_POC = 0
        self.par=[]
        num_classes=62
        z_dim=512
        #self.cls=nn.Linear(512,62)
        self.cls=nn.Linear(args.z_dim,62)
        self.r_mu = nn.Parameter(torch.zeros(num_classes,z_dim))
        self.r_sigma = nn.Parameter(torch.ones(num_classes,z_dim))
        self.C = nn.Parameter(torch.ones([]))
        self.L2R_coeff=0.01
        self.CMI_coeff=0.001
        self.optimizer=None
        self.flag=0
        self.net=None

    def __str__(self):
        return self.name
    
    def init_model(self, model):
        # solo qunado serve usare il modello dei client
        self.model=copy.deepcopy(model)
        return

    def kill_model(self):
        #cancella il modello locale-->libera la memoria
        del self.model
        return


    @staticmethod
    def update_metric(metric, outputs, labels):
        _, predictions = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        for key, value in zip(labels, predictions):
            metric[key]=value

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            # errore credo sul type di images, da errore sul type di oggetto anche solo chiamando self.model(images)
            return self.model(images)
        if self.args.model == 'cnn': #non va salvato un modello locale?
            #device = torch.device('cuda')
            #images = images.to(device)
            return self.model(images)
        else:
            raise NotImplementedError
        
    '''def run_epoch_POC(self, cur_epoch, num_batches):
    
        self.model.train()
        total_loss = 0

        for cur_step, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self._get_outputs(images)
            loss = self.reduction(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if cur_step + 1 == num_batches:
                break

        average_loss = total_loss / num_batches
        return average_loss'''

    def run_epoch(self, cur_epoch):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        
        if self.args.fedSR:
            #par,acc,loss,model=compute_loss(self.model,self.criterion, optimizer, self.train_loader,self.device,param=self.par)
            #self.model=copy.deepcopy(model)
            #self.optimizer=copy.deepcopy(par[3])
            #self.par=par
            #return par,acc,loss
            #def compute_loss(model,criterion, optim, dataset,device,lr=0.001, num_classes=62, z_dim=31,L2R_coeff=0.01,CMI_coeff=0.001,param=[]):
          
          
          self.cls=self.model[-1] # ho aggiunto cls dal modello grande al posto di passare due elementi ogni volta
          self.cls.to(self.device) # vedere se serve passare cls passato da server o se serve passarlo nuovo ogni volta, bisogna PROVARE
          #self.cls_med.to(self.device)
          #self.net = nn.Sequential(self.model,self.cls)
          #self.net.to(self.device)
          self.net=nn.Sequential(*self.model[:-1])
          self.net.to(self.device)
          self.model.train()
          self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
          total_loss = 0
          total_regL2R = 0
          total_regCMI = 0
          total_regNegEnt = 0
          total_samples = 0
          correct=0
          
          #dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
          if self.flag==0:
            
            self.optimizer.add_param_group({'params':[self.r_mu,self.r_sigma,self.C],'lr':0.001,'momentum':0.9})
            self.r_sigma=self.r_sigma.to(self.device)
            self.r_mu=self.r_mu.to(self.device)
            self.C=self.C.to(self.device)
            self.flag=1
            
          
          for x, y in self.train_loader:
              x, y = x.to(self.device), y.to(self.device)

              z, (z_mu, z_sigma) = featurize(self.net,x,self.args.z_dim) #passare tutto il modello a featurize, non solo il net
              logits = self.cls(z) 
              
              loss = F.cross_entropy(logits, y) #qua è il punto cruciale

              obj = loss
              regL2R = torch.zeros_like(obj)
              regCMI = torch.zeros_like(obj)
              regNegEnt = torch.zeros_like(obj)
              
              
              if self.L2R_coeff != 0.0:
                  regL2R = z.norm(dim=1).mean()
                  obj = obj + self.L2R_coeff * regL2R

              if self.CMI_coeff != 0.0:
                  
                  r_sigma_softplus = F.softplus(self.r_sigma)
                  
                
                  r_mu_loc = self.r_mu[y]
                  r_sigma_loc = r_sigma_softplus[y]
                  
                  z_mu_scaled = z_mu * self.C
                  z_sigma_scaled = z_sigma *self.C
                  regCMI = torch.log(r_sigma_loc) - torch.log(z_sigma_scaled) + \
                          (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu_loc) ** 2) / (2 * r_sigma_loc ** 2) - 0.5
                  regCMI = regCMI.sum(1).mean()
                  obj = obj + self.CMI_coeff * regCMI

              #z_dist = dist.Independent(dist.Normal(z_mu, z_sigma), 1)
              #mix_coeff = dist.Categorical(x.new_ones(x.shape[0]))
              #mixture = dist.MixtureSameFamily(mix_coeff, z_dist)
              #log_prob = mixture.log_prob(z)
              #regNegEnt = log_prob.mean()

              self.optimizer.zero_grad()
              obj.backward()
              self.optimizer.step()


              batch_size = x.shape[0]
              total_loss += loss.item() * batch_size
              #total_regL2R += regL2R.item() * batch_size
              #total_regCMI += regCMI.item() * batch_size
              #total_regNegEnt += regNegEnt.item() * batch_size
              total_samples += y.size(0)
              _, prediction = torch.max(logits.data, 1)
              correct += (logits.argmax(1)== y).sum().item()

          loss_avg = total_loss / total_samples
          #regL2R_avg = total_regL2R / total_samples
          #regCMI_avg = total_regCMI / total_samples
          #regNegEnt_avg = total_regNegEnt / total_samples
          acc=(correct/total_samples)*100

          return 'roba', loss_avg, acc
        else:
            self.model.train()
            total_loss = 0
            #total_metric = defaultdict(float)
            total = 0
            correct =0
            #loss_function=self.criterion()

            for cur_step, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                images = images.to(self.device)  # Move images to the same device as model and labels
                labels = labels.to(self.device)

                outputs = self._get_outputs(images)
                
                loss= self.criterion(outputs,labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                #print('this are putputs',outputs)
                #self.update_metric(total_metric, outputs, labels)
                _, prediction = torch.max(outputs.data, 1)
                #print('this is the prediction:' ,prediction)  # grab prediction as one-dimensional tensor
                total += labels.size(0)
                #labels = labels.cuda()
                #print('this are the labels:',labels)
                correct += (prediction == labels).sum().item()

            train_loss = total_loss / len(self.train_loader)
            train_acc = 100 * correct / total

            return len(self.train_loader),train_loss,train_acc
    

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        #n_samples = len(self.train_loader.dataset) #esce spesso un errore, capire come risolvere nel caso
        #local_model = copy.deepcopy(self.model) #state dict tira fuori il dizionario: non ha utilità ma salvare il locale si 

         #da vedere se salvare il locale da modificare poi per fare tuning
        #self.optimizer=self.optimizer.to(device)
        self.model=self.model.to(self.device)
        
        
        for epoch in range(self.args.num_epochs):
            if self.args.fedSR:
                _, loss,accuracy  = self.run_epoch(epoch)
            else:
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
                _, loss,accuracy  = self.run_epoch(epoch)
            print(f'Client {self.name}, Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.5f}')

        return loss

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        if self.args.fedSR:
              self.cls=self.model[-1] # ho aggiunto cls dal modello grande al posto di passare due elementi ogni volta
              self.cls.to(self.device)
              self.net=nn.Sequential(*self.model[:-1])
              self.net.to(self.device)

        self.model.eval()  # Imposta il modello in modalità di valutazione (non addestramento)
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.cuda()
                
                if self.args.fedSR:
                      z, (z_mu, z_sigma) = featurize(self.net,images,self.args.z_dim) #passare tutto il modello a featurize, non solo il net
                      preds = torch.softmax(self.cls(z),dim=1)
                      preds = preds.view([self.num_samples,-1,self.num_classes]).mean(0)
                      outputs=torch.log(preds) #ho corretto il problema potenziale del FedSR
                else:
                    outputs = self.model(images)  # Esegue l'inferenza sulle immagini
                
                self.update_metric(metric, outputs, labels)  # Aggiorna la metrica
                
        # self.model.train()  # Riporta il modello in modalità di addestramento

    def get_len(self):
        return len(self.train_loader)

'''import copy
import torch
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from FedSR_pers import *
from utils.utils import HardNegativeMining, MeanReduction
from MyDataset import *

class Client:

    def __init__(self, args, dataset, model, transformation=None,test_client=False):
        self.args = args
        self.dataset = dataset
        # da decommentare quando usi femnist
        self.name = self.dataset.client_name
        self.model = copy.deepcopy(model)
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True,transform=transformation) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean') # per ora userei questo
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_POC = 0

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, predictions = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        for key, value in zip(labels, predictions):
            metric[key]=value

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            # errore credo sul type di images, da errore sul type di oggetto anche solo chiamando self.model(images)
            return self.model(images)
        if self.args.model == 'cnn': #non va salvato un modello locale?
            #device = torch.device('cuda')
            #images = images.to(device)
            return self.model(images)
        else:
            raise NotImplementedError
        
    def run_epoch_POC(self, cur_epoch, optimizer, num_batches):
    
        self.model.train()
        total_loss = 0

        for cur_step, (images, labels) in enumerate(self.train_loader):
            optimizer.zero_grad()
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self._get_outputs(images)
            loss = self.reduction(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if cur_step + 1 == num_batches:
                break

        average_loss = total_loss / num_batches
        return average_loss

    def run_epoch(self, cur_epoch, optimizer,FedSR=True):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        if FedSR:
            stats,acc,loss,model=compute_loss(self.model,self.criterion, optimizer, self.train_loader,self.device)
            self.model=copy.deepcopy(model)
            return stats,acc,loss
        else:
            self.model.train()
            total_loss = 0
            #total_metric = defaultdict(float)
            total = 0
            correct =0
            #loss_function=self.criterion()

            for cur_step, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)  # Move images to the same device as model and labels
                labels = labels.to(self.device)
                outputs = self._get_outputs(images)
                # self.reduction è MeanReduction,
                # fa solo una media dei valori di outputs, non guarda nemmeno le labels!
                # va bene come loss?
                #loss = self.reduction(outputs, labels)
                loss= self.criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                #print('this are putputs',outputs)
                #self.update_metric(total_metric, outputs, labels)
                _, prediction = torch.max(outputs.data, 1)
                #print('this is the prediction:' ,prediction)  # grab prediction as one-dimensional tensor
                total += labels.size(0)
                #labels = labels.cuda()
                #print('this are the labels:',labels)
                correct += (prediction == labels).sum().item()

            train_loss = total_loss / len(self.train_loader)
            train_acc = 100 * correct / total

            return len(self.train_loader),train_loss,train_acc
    

    def train(self,POC):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        #n_samples = len(self.train_loader.dataset) #esce spesso un errore, capire come risolvere nel caso
        #local_model = copy.deepcopy(self.model) #state dict tira fuori il dizionario: non ha utilità ma salvare il locale si 

        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9) #da vedere se salvare il locale da modificare poi per fare tuning
        self.model=self.model.to(self.device)
        if POC:
            for epoch in range(self.args.num_epochs):
                loss = self.run_epoch_POC(epoch, optimizer,15)
                return loss
        else:
            for epoch in range(self.args.num_epochs):
                _, loss,accuracy  = self.run_epoch(epoch, optimizer)
                print(f'Client {self.name}, Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.5f}')

        return 'done'

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.eval()  # Imposta il modello in modalità di valutazione (non addestramento)
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.cuda()
                
                outputs = self.model(images)  # Esegue l'inferenza sulle immagini
                
                self.update_metric(metric, outputs, labels)  # Aggiorna la metrica
                
        # self.model.train()  # Riporta il modello in modalità di addestramento

    def get_len(self):
            return len(self.train_loader)'''