import torch
from model import Net
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

class learn(Net):
    def __init__(self, network, device, lr, m):
        super().__init__(network)

        self.net = Net(network)
        self.criterion = nn.CrossEntropyLoss().to('cuda')  ## 22 78 ##weight=torch.Tensor([0.1, 0.35])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  # 0.001 ##adamw
        self.network = network
        self.device = device

    def train(self, epoch_num, trainloader, validloader):
        print("Training Started")
        train_loss = []
        train_accuracy =[]
        validation_accuracy =[]

        for epoch in range(epoch_num):  # loop over the dataset multiple times


            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                model = self.net
                model.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 9:    # print every 2000 mini-batches
                    argmax_outputs = torch.argmax(outputs, dim=1)
                    acc = accuracy_score(argmax_outputs, labels)
                    train_loss.append(loss.item())
                    train_accuracy.append(acc * 100)
                    valid_acc = 0
                    for i, data in enumerate(validloader, 0):
                        with torch.no_grad():
                            inputs, labels = data
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            self.optimizer.zero_grad()
                            model = self.net
                            model.to(self.device)
                            outputs = model(inputs)
                            argmax_outputs = torch.argmax(outputs, dim=1)
                            loss = self.criterion(outputs, labels)
                            valid_acc += accuracy_score(argmax_outputs, labels)
                            valid_iter_num = i
                    validation_accuracy.append(valid_acc / valid_iter_num * 100)
                    if validation_accuracy[-1] == max(validation_accuracy):
                        best_validation_accuracy = validation_accuracy[-1]
                        best_model = model.first_layer.weight.data.cpu().numpy()
            print('Validation accuracy at epoch' + str(epoch) + " is %" + str(validation_accuracy[-1]))

        print('Training finished with accuracy: %' + str(train_accuracy[-1]))

        return np.array(train_accuracy), np.array(train_loss), np.array(validation_accuracy), best_validation_accuracy, best_model

    def test(self,testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            model = self.net
            model.to(self.device)
            for data in testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        return 100 * correct / total