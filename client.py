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
        self.model = copy.deepcopy(model) #se è test invece?
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        if self.args.model == 'cnn': #non va salvato un modello locale?
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
        total_metric = defaultdict(float)

        for cur_step, (images, labels) in enumerate(self.train_loader):
            optimizer.zero_grad()
            outputs = self._get_outputs(images)
            loss = self.reduction(self.criterion, outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            self.update_metric(total_metric, outputs, labels)

        return len(self.train_loader), total_loss / len(self.train_loader), total_metric


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
            _, loss, metric = self.run_epoch(epoch, optimizer)
            print(f'Client {self.name}, Epoch [{epoch + 1}/{self.args.num_epochs}], Loss: {loss:.4f}')

        return n_samples

    def test(self,optimizer): #ma quindi è utile?
        """
        This method tests the model on the local dataset of the client.
        :return: accuracy of the model on the local test set
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                outputs = self._get_outputs(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.reduction(self.criterion, outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        accuracy = 100 * correct / total
        return accuracy
