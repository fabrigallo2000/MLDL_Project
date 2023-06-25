import torch
import torchvision
import torch.utils.data as tdata
from torchvision import transforms, datasets
from torchvision.transforms import Normalize, ToTensor
import torch.nn as nn  # neural network
import torch.optim as optim  # optimization layer
import torch.nn.functional as F  # activation functions
import matplotlib.pyplot as plt
import argparse
import time
from collections import OrderedDict
from models.YourCNN import YourCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

# load data in
train_set = datasets.EMNIST(root="/content/MLDL_Project/data", split="balanced",
                            train=True, transform=transforms.Compose([ToTensor()]),
                           download=True
                           )
test_set = datasets.EMNIST(root="/content/MLDL_Project/data", split="balanced", 
                           train=False,transform=transforms.Compose([ToTensor()]),
                           download=True
                          )
entire_trainset = torch.utils.data.DataLoader(train_set, shuffle=True)

# ruota validation set dell'angolo lasciato fuori da train
val_set.dataset.transform = transforms.Compose([transforms.ToTensor(),
        nptr.Rotate(60)
    ])

print(f'train set size: {train_size}, validation set size: {valid_size}')

def train(net, optimizer, criterion, args):
    '''
    Returns validation loss and accuracy
    
        Parameters:
            net (CNN): a convolutional neural network to train
            optimizer: optimizer
            criterion (loss function): a loss function to evaluate the model on
            args (ArgumentParser): hyperparameters
        
        Returns:
            net (CNN): a trained model
            train_loss (float): train loss
            train_acc (float): train accuracy
    '''
    train_loader = tdata.DataLoader(train_set, batch_size=args.train_batch, shuffle=True)
    
    net.train()
    
    correct = 0
    total = 0
    train_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        
        optimizer.zero_grad()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # the class with the highest value is the prediction
        _, prediction = torch.max(outputs.data, 1)  # grab prediction as one-dimensional tensor
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return net, train_loss, train_acc  # net is returned to be fed to the test function later
def validate(net, criterion, args):
    '''
    Returns validation loss and accuracy
    
        Parameters:
            net (CNN): a convolutional neural network to validate
            criterion (loss function): a loss function to evaluate the model on
            args (ArgumentParser): hyperparameters
        
        Returns:
            val_loss (float): validation loss
            val_acc (float): validation accuracy
    '''
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch, shuffle=True)
    
    net.eval()

    correct = 0
    total = 0
    val_loss = 0 
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

    return val_loss, val_acc
def test(net, args):
    '''
    Returns test accuracy
    
        Parameters:
            net (CNN): a trained model
            args (ArgumentParser): hyperparameters
        
        Returns:
            test_acc (float): test accuracy of a trained model
    '''
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch, shuffle=True)

    net.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total

    return test_acc
def main(args):
    '''
    Execute train and validate functions epoch-times to train a CNN model.
    Each time, store train & validation loss and accuracy.
    Then, test the model and return the result.
    
        Parameter:
            args (ArgumentParser): hyperparameters
        
        Returns:
            vars(args) (Dictionary): settings of the model
            results (OrderedDict): stored stats of each epoch + test accuracy
    '''
    net = YourCNN( num_classes = args.out_dim)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    
    # select an optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # containers to keep track of statistics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    time_total = 0
    df_accuracy = pd.DataFrame(columns=['x_round', 'y'])

    for epoch in range(args.epoch):  # number of training to be completed
        time_start = time.time()
        net, train_loss, train_acc = train(net, optimizer, criterion, args)
        val_loss, val_acc = validate(net, criterion, args)
        time_end = time.time()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        time_duration = round(time_end - time_start, 2)
        time_total += time_duration
        
        # dataframe update
        df_accuracy.loc[len(df_accuracy)] = [epoch+1, val_accs]

        # print results of each iteration
        print(f'Epoch {epoch+1}, Accuracy(train, validation):{round(train_acc, 2), round(val_acc, 2)}, '
              f'Loss(train, validation):{round(train_loss, 4), round(val_loss, 4)}, Time: {time_duration}s')
    
    df_accuracy.to_csv('centralized_Rotete_LOO.csv')
    #test_acc = test(net, args)

    results = OrderedDict()
    results['train_losses'] = [round(x, 4) for x in train_losses]
    results['val_losses'] = [round(x, 4) for x in val_losses]
    results['train_accs'] = [round(x, 2) for x in train_accs]
    results['val_accs'] = [round(x, 2) for x in val_accs]
    results['train_acc'] = round(train_acc, 2)
    results['val_acc'] = round(val_acc, 2)
    #results['test_acc'] = round(test_acc, 2)
    results['time_total'] = round(time_total, 2)
    print(results['test_acc'])
    
    return vars(args), results

parser = argparse.ArgumentParser()
args = parser.parse_args("")

#### Model Capacity ####
args.out_dim = 62

#### Regularization ####

#### Optimization ####
args.optim = 'sgd'
args.lr = 0.001  # learning rate
args.epoch = 10
args.train_batch = 256
args.test_batch = 256
main(args)