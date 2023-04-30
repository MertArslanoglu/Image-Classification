import torch
from model import Net
import torch.nn as nn
import numpy as np


class learn(Net):
    def __init__(self, network, activation_type, device, lr, m, part, optimizer, scheduled):
        # inheritance of the Net class:
        super().__init__(network, activation_type)
        #  initialization of Net class:
        self.net = Net(network, activation_type)
        #  parameters are initialized and accessible by the subroutines:
        self.criterion = nn.CrossEntropyLoss().to('cuda')
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr, m)
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters())

        self.network = network
        self.device = device
        self.part = part
        self.scheduled = scheduled

    def train(self, epoch_num, trainloader, validloader):
        print("Training Started")
        # list initializations of the recordings:
        train_loss = []
        train_accuracy =[]
        validation_accuracy =[]
        loss_gradient_list = []
        best_validation_accuracy = 0
        best_model = None


        for epoch in range(epoch_num): #  loop over the dataset multiple times == epoch
            if self.scheduled == True:
                print("Scheduled")
                # learning rate schedule changes the learning rate at scheduled epochs:
                if epoch == 0:
                    self.optimizer = torch.optim.SGD(self.net.parameters(), 0.1, 0)
                if epoch == 5:
                    self.optimizer = torch.optim.SGD(self.net.parameters(), 0.01, 0)
                if epoch == 20:
                    self.optimizer = torch.optim.SGD(self.net.parameters(), 0.001, 0)

            #  data initalizations to sum over these values:
            running_loss = 0.0
            total_frame = 0
            true_frame = 0
            loss_gradient = 0.0
            for i, data in enumerate(trainloader, 0):  # iteration of batches of data from train set
                #  separation labels and images of batches and moving to gpu:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #  initialize the gradient as 0 before calculate new gradient:
                self.optimizer.zero_grad()
                # get the last version of model and move it into gpu:
                model = self.net
                model.to(self.device)
                # evaluate the batch with the model
                outputs = model(inputs)
                # one hot encoded probabilities to output labels:
                argmax_output = torch.argmax(outputs, dim=1)
                # comparison between the output labels and ground-truth labels:
                mask = torch.eq(argmax_output, labels)
                # number of the true predictions:
                true_frame += sum(mask).item()
                # number of all input samples so far:
                total_frame += mask.size()[0]
                # loss calculation:
                loss = self.criterion(outputs, labels)
                # back propagation:
                loss.backward()
                # gradient vector reshape into dim = 2 to calculate magnitude:
                grad = torch.reshape(model.first_layer.weight.grad,(2,-1,))
                # record magnitude of gradient cumulatively:
                loss_gradient += torch.linalg.matrix_norm(grad).cpu()
                # updating model:
                self.optimizer.step()
                # recording the loss cumulatively:
                running_loss += loss.item()
                if i % 10 == 9:    # check accuracy every 10 mini-batches:
                    # add loss to the list and reinitialize as 0:
                    train_loss.append(running_loss)
                    running_loss = 0.0
                    if self.part == 4: # record gradient by averaging if part 4 and reinitialize as 0:
                        loss_gradient_list.append(loss_gradient/total_frame)
                        loss_gradient = 0.0
                        total_frame = 0
                    if self.part == 3 or self.part == 5: # record train accuracy by averaging if part 3 or 5 and reinitialize as 0:
                        train_accuracy.append(true_frame/total_frame * 100)
                        total_frame = 0
                        true_frame = 0
                        # initialize validation check parameters as 0:
                        valid_true_frame = 0
                        valid_total_frame = 0
                        # validation loop identical with tarining loop
                        # without gradient calcualtion to increase computation speed:
                        for i, data in enumerate(validloader, 0):
                            with torch.no_grad():
                                inputs, labels = data
                                inputs, labels = inputs.to(self.device), labels.to(self.device)
                                self.optimizer.zero_grad()
                                model = self.net
                                model.to(self.device)
                                outputs = model(inputs)
                                argmax_output = torch.argmax(outputs, dim=1)
                                mask = torch.eq(argmax_output, labels)
                                valid_true_frame += sum(mask).item()
                                valid_total_frame += mask.size()[0]
                        # add the validation accuracy at every 10 mini batches:
                        validation_accuracy.append(valid_true_frame / valid_total_frame * 100)
                        # record best validation accuracy and the correspondent model:
                        # (not asked by the homework, but I realize late)
                        if validation_accuracy[-1] == max(validation_accuracy):
                            best_validation_accuracy = validation_accuracy[-1]
                            best_model = model.first_layer.weight.data.cpu().numpy()
        # validaiton accuracy feedback after each epoch
        if self.part == 3 or self.part == 5:
            print('Validation accuracy at epoch' + str(epoch) + " is %" + str(validation_accuracy[-1]))


        return np.array(train_accuracy), np.array(train_loss), np.array(validation_accuracy), best_validation_accuracy, best_model, np.array(loss_gradient_list)

    def test(self,testloader):
        # test parameters initalziation:
        correct = 0
        total = 0
        # without gradient calcualtion to increase computation speed:
        with torch.no_grad():
            # get the last version of model and move it into gpu:
            model = self.net
            model.to(self.device)
            for data in testloader:
                #  separation labels and images of batches and moving to gpu:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # evaluate the batch with the model
                outputs = model(images)
                # one hot encoded probabilities to output labels:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # comparison between the output labels and ground-truth labels and true prediciton number:
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        return 100 * correct / total