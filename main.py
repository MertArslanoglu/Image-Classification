import torch
from train import learn
import pickle
from dataset_creater import dataset_creater
"""This python file runs the experiments for the Homework-1 of EE 447 Computational Intelligence course at METU.
The configurable parameters to adjust the experiments are :
*part : The compatible inputs are 3,4, and 5
*list_of_arch : The architectures which the experiment will be conducted with. Type: list 
                Compatible elements: "mlp_1", "mlp_2", "cnn_3", "cnn_4", "cnn_5"
*batchsize
*epoch_num
*learning_rate
*momentum
"""

#THE CONFIGURABLE PARAMETERS FOR EXPERIMENT:
part = 5
batchsize = 50
learning_rate = 0.01
momentum = 0.0

#DEVICE SELECTION IF CUDA IS AVAILABLE:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


if part == 3:
    """That block conducts experiment for part-3 of the homework:
run_num: number of distinct and identical experiments for each architecture.
optimizer: Adam
activation: relu
Epoch number: 15
It records these information: loss_curve, train_acc_curve, val_acc_curve, test_acc, best_val_acc, weights
learning_rate scheduled : FALSE

"""
    list_of_arch = ["mlp_1", "mlp_2", "cnn_3", "cnn_4", "cnn_5"]
    epoch_num = 15
    run_num = 10
    optimizer = "adam"
    scheduled = False
    for arch in list_of_arch:
        activation_type = "relu"
        #  dictionary initialization:
        dict = {}
        dict["name"] = arch

        for it in range(run_num):
            print("Iteration :" +str(it))
            train_generator, validation_generator, test_generator = dataset_creater(batchsize, False)  # dataset is
            # created at the start of each run because it is random
            a = learn(arch, activation_type, device, learning_rate, momentum, part, optimizer, scheduled)  # class initialization with
            # parameters
            if it == 0:

                #  FIRST RUN : train and test functions sequentially outputs are recorded:
                train_accuracy_h, train_loss_h, validation_accuracy_h, best_validation_accuracy, best_model, grad_curve = a.train(epoch_num, train_generator, validation_generator)
                test_acc = a.test(test_generator)

            else:
                #  OTHER THAN FIRST RUN : train and test functions sequentially outputs are recorded:
                train_accuracy, train_loss, validation_accuracy, best_validation_accuracy_h, best_model_h, grad_curve = a.train(epoch_num,                                                                                                     train_generator,
                                                                                                                validation_generator)
                test_acc_h = a.test(test_generator)

                #  Summing up the outputs of each run:
                train_accuracy_h += train_accuracy
                train_loss_h += train_loss
                validation_accuracy_h += validation_accuracy

                #  Seeking for the best among all runs:
                if best_validation_accuracy_h > best_validation_accuracy:
                    best_validation_accuracy = best_validation_accuracy_h
                    best_model = best_model_h
                #  Seeking for the best among all runs:
                if test_acc_h > test_acc: test_acc = test_acc_h
        #  The final outputs are written on a dictionary and saved:
        dict["loss_curve"] = train_loss_h/run_num
        dict["train_acc_curve"] = train_accuracy_h/run_num
        dict["val_acc_curve"] = validation_accuracy_h/run_num
        dict["weights"] = best_model
        dict["test_acc"] = test_acc
        dict["best_val_acc"] = best_validation_accuracy
        with open('./part_3_results/{}.pickle'.format(arch), 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if part == 4:
    """That block conducts experiment for part-4 of the homework:
    optimizer: SGD
    activation: relu, sigmoid
    Epoch number: 15
    It records these information: loss_curve, magnitude curve of gradients w.r.t. first layer

    """
    list_of_arch = ["mlp_1", "cnn_5"]
    optimizer = "SGD"
    epoch_num = 15
    activation_types = ["relu", "sigmoid"]
    scheduled = False


    for arch in list_of_arch:  # iteration through the architecture types
        #  dictionary initialization:
        dict = {}
        dict["name"] = arch
        train_generator, validation_generator, test_generator = dataset_creater(batchsize, False)  # created at the
        # start of each run because it is random
        for activation_type in activation_types:  # iteration through the activation types
            a = learn(arch, activation_type, device, learning_rate, momentum, part, optimizer, scheduled)

            train_accuracy, train_loss, validation_accuracy, best_validation_accuracy_h, best_model_h, grad_curve = a.train(epoch_num,                                                                                                     train_generator,
                                                                                                                validation_generator)
            dict["{}_loss_curve".format(activation_type)] = train_loss
            dict["{}_grad_curve".format(activation_type)] = grad_curve
        #  The final outputs are written on a dictionary and saved:
        with open('./part_4_results/{}.pickle'.format(arch), 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if part == 5:
    """That block conducts experiment for part-5 of the homework:
        optimizer: SGD
        activation: relu
        Epoch number: 30 ,40
        It records these information: loss_curve_1/01/001 and val_acc_curve_1/01/001 

        """
    list_of_arch = ["cnn_4"]
    epoch_num = 40
    learning_rates=[0.1 , 0.01, 0.001]
    activation_type = "relu"
    optimizer = "SGD"
    scheduled = False

    for arch in list_of_arch:  # iteration through the architecture types
        #  dictionary initialization:
        dict = {}
        dict["name"] = arch

        train_generator, validation_generator, test_generator = dataset_creater(batchsize, False) # created at the
        # start of each run because it is random
        for learning_rate in learning_rates:  # iteration through different learning rates
            a = learn(arch, activation_type, device, learning_rate, momentum, part, optimizer, scheduled)

            train_accuracy, train_loss, validation_accuracy, best_validation_accuracy_h, best_model_h, grad_curve = a.train(epoch_num, train_generator, validation_generator)

            # saving the recordings to a dictionary
            if learning_rate == 0.1:
                dict["loss_curve_1".format(activation_type)] = train_loss
                dict["val_acc_curve_1".format(activation_type)] = validation_accuracy

            if learning_rate == 0.01:
                dict["loss_curve_01".format(activation_type)] = train_loss
                dict["val_acc_curve_01".format(activation_type)] = validation_accuracy

            if learning_rate == 0.001:
                dict["loss_curve_001".format(activation_type)] = train_loss
                dict["val_acc_curve_001".format(activation_type)] = validation_accuracy

        #  The final outputs are written on a dictionary and saved:
        with open('./part_5_results/{}.pickle'.format(arch), 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # SCHEDULED EXPERIMENT SAME CODES OF PART 3 WITH RUN NUM = 1
    run_num = 1
    scheduled = True
    for arch in list_of_arch:
        #  dictionary initialization:
        dict = {}
        dict["name"] = arch

        for it in range(run_num):
            print("Iteration :" + str(it))
            train_generator, validation_generator, test_generator = dataset_creater(batchsize, False)  # dataset is
            # created at the start of each run because it is random
            a = learn(arch, activation_type, device, learning_rate, momentum, part, optimizer,
                      scheduled)  # class initialization with
            # parameters
            if it == 0:

                #  FIRST RUN : train and test functions sequentially outputs are recorded:
                train_accuracy_h, train_loss_h, validation_accuracy_h, best_validation_accuracy, best_model, grad_curve = a.train(
                    epoch_num, train_generator, validation_generator)
                test_acc = a.test(test_generator)

            else:
                #  OTHER THAN FIRST RUN : train and test functions sequentially outputs are recorded:
                train_accuracy, train_loss, validation_accuracy, best_validation_accuracy_h, best_model_h, grad_curve = a.train(
                    epoch_num, train_generator,
                    validation_generator)
                test_acc_h = a.test(test_generator)

                #  Summing up the outputs of each run:
                train_accuracy_h += train_accuracy
                train_loss_h += train_loss
                validation_accuracy_h += validation_accuracy

                #  Seeking for the best among all runs:
                if best_validation_accuracy_h > best_validation_accuracy:
                    best_validation_accuracy = best_validation_accuracy_h
                    best_model = best_model_h
                #  Seeking for the best among all runs:
                if test_acc_h > test_acc: test_acc = test_acc_h
        #  The final outputs are written on a dictionary and saved:
        dict["loss_curve"] = train_loss_h / run_num
        dict["train_acc_curve"] = train_accuracy_h / run_num
        dict["val_acc_curve"] = validation_accuracy_h / run_num
        dict["weights"] = best_model
        dict["test_acc"] = test_acc
        dict["best_val_acc"] = best_validation_accuracy
        with open('./part_5_1_results/{}5.1.pickle'.format(arch), 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




