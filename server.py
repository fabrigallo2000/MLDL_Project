import copy
from collections import OrderedDict

import numpy as np
import torch


class Server:
    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.prob = {
            'clients_per_round': 10,
            'prob_10_clients': 0.5,
            'prob_30_clients': 0.0001
        }

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)
    
    def smart_select_clients(self):
        
        num_clients = self.args.clients_per_round
        client_probs = np.ones(len(self.train_clients))

        # Set probability for 10% of clients
        num_10_clients = int(0.1 * num_clients)
        client_probs[:num_10_clients] = self.prob['prob_10_clients']

        # Set probability for 30% of clients
        num_30_clients = int(0.3 * num_clients)
        client_probs[num_10_clients:num_10_clients + num_30_clients] = self.prob['prob_30_clients']

        # Normalize probabilities
        client_probs /= np.sum(client_probs)

        selected_clients = np.random.choice(self.train_clients, num_clients, replace=False, p=client_probs)
        return selected_clients

    def train_round(self, clients):
        """
        This method trains the model with the dataset of the clients. It handles the training at single round level.
        :param clients: list of all the clients to train
        :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, c in enumerate(clients):
           
            # train the client 
            
            #c.model.load_state_dict(self.model_params_dict)
            c.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            c.train()
            # Train the client model using its dataset
            '''for _ in range(self.args.local_epochs):
                for _, (data, target) in enumerate(c.train_loader):
                    data, target = data.to(self.args.device), target.to(self.args.device)

                    self.args.optimizer.zero_grad()
                    output = client_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    client_optimizer.step()'''

            # Get the updated model's parameters
            updated_params = copy.deepcopy(c.model.state_dict())

            # Compute the difference between the current and updated parameters
            updates.append(OrderedDict({key: updated_params[key] - self.model_params_dict[key] for key in updated_params}))

        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation.
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        aggregated_params = OrderedDict()
        num_updates = len(updates)

        for key in updates[0].keys():
            # Sum the updates for each parameter
            param_sum = sum([updates[i][key] for i in range(num_updates)])
            # Calculate the average update
            avg_param = param_sum / num_updates
            # Apply the average update to the server's model parameters
            aggregated_params[key] = self.model_params_dict[key] + avg_param

        return aggregated_params

    def train(self):
        """
        This method orchestrates the training, evaluations, and tests at the round level.
        """
        for r in range(self.args.num_rounds):
            # Select clients for this round
            clients = self.select_clients()

            # Train clients and gather updates
            updates = self.train_round(clients)

            # Aggregate the updates
            aggregated_params = self.aggregate(updates)

            # Update the server's model parameters
            self.model.load_state_dict(aggregated_params)

            # Evaluate on train clients
            train_loss, train_accuracy = self.eval_train()
            print(f"Round {r + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Test on test clients
            # attualmente self.test non ritorna nulla/ non fa nulla
            #test_loss, test_accuracy = self.test()
            #print(f"Round {r + 1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    def eval_train(self):
        """This method handles the evaluation on the train clients. """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            # Set the model in evaluation mode

            self.model.eval()

            for client in self.train_clients:
                client_samples = 0
                client_correct = 0
                client_loss = 0.0

                '''for _, (data, target) in enumerate(client.train_loader):
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    output = self.model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    predictions = output.argmax(dim=1)
                    correct = (predictions == target).sum().item()

                    client_loss += loss.item() * data.size(0)
                    client_correct += correct
                    client_samples += data.size(0)'''

                for _, (data, target) in enumerate(client.train_loader):
                    inputs = data
                    labels = target
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = self.model(inputs)

                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    
                    client_loss += loss.item()
                    _, prediction = torch.max(outputs.data, 1)
                    client_samples += labels.size(0)
                    client_correct += (prediction == labels).sum().item()

                # Accumulate results for each client
                total_loss += client_loss
                total_correct += client_correct
                total_samples += client_samples

        # Calculate average metrics across all clients
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
    
    def test(self):
        """
        This method handles the test on the test clients
        """
        for client in self.test_clients:
            metric = self.metrics  
            client.model.load_state_dict(self.model_params_dict)
            client.test(metric)  # Esegue il test utilizzando il metodo test del client

            # Stampa i risultati della metrica per il client corrente
            print(f"Testing  client {client.name}")
        # metric e dict, dict non ha funzione get results
        # print(f"Accuracy: {metric.get_results()}")


'''def test(self):
    """
    This method handles the test on the test clients.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        # Set the model in evaluation mode
        self.model.eval()

        for client in self.test_clients:
            client_samples = 0
            client_correct = 0
            client_loss = 0.0

            for _, (data, target) in enumerate(client.test_loader):
                inputs = data
                labels = target
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs)

                loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                client_loss += loss.item()
                _, prediction = torch.max(outputs.data, 1)
                client_samples += labels.size(0)
                client_correct += (prediction == labels).sum().item()
            
            total_loss += client_loss
            total_correct += client_correct
            total_samples += client_samples
            

    # Calculate average metrics across all clients
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy'''
'''import torch

def test(self, metric):
    """
    This method tests the model on the local dataset of the client.
    :param metric: StreamMetric object
    """
    self.model.eval()  # Imposta il modello in modalità di valutazione (non addestramento)
    with torch.no_grad():
        for i, (images, labels) in enumerate(self.test_loader):
            #images = images.to(self.device)
            #labels = labels.to(self.device)
            
            outputs = self.model(images)  # Esegue l'inferenza sulle immagini
            self.update_metric(metric, outputs, labels)  # Aggiorna la metrica
            
    #self.model.train()  # Riporta il modello in modalità di addestramento'''

    