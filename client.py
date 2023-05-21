import copy
import torch
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        # da decommentare quando usi femnist
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

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
            device = torch.device('cuda')
            images = images.to(device)
            return self.model(images)
        else:
            raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        self.model.train()
        total_loss = 0
        #total_metric = defaultdict(float)
        total = 0
        correct =0
        loss_function=self.criterion()

        for cur_step, (images, labels) in enumerate(self.train_loader):
            optimizer.zero_grad()
            outputs = self._get_outputs(images)
            # self.reduction è MeanReduction,
            # fa solo una media dei valori di outputs, non guarda nemmeno le labels!
            # va bene come loss?
            #loss = self.reduction(outputs, labels)
            loss= loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print('this are putputs',outputs)
            #self.update_metric(total_metric, outputs, labels)
            _, prediction = torch.max(outputs.data, 1)
            print('this is the prediction:' ,prediction)  # grab prediction as one-dimensional tensor
            total += labels.size(0)
            labels = labels.cuda()
            print('this are the labels:',labels)
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

        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9) #da vedere se salvare il locale

        for epoch in range(self.args.num_epochs):
            _, loss,accuracy  = self.run_epoch(epoch, optimizer)
            print(f'Client {self.name}, Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.5f}')

        return

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
