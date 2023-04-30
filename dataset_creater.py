import torch
import torchvision
import torchvision.transforms as transforms
import random
from imshow import visualize


def dataset_creater(batchsize, bool_visualize):
    """creates a new dataset
    Train set: 45000 samples
    Validation set: 5000 samples generated from cifar dataset with equal number of samples from each class selected random.
    Test set: The default cifar test set.

    if bool_visualize is True: prints random samples from the dataset out.
    """
    # it is adjusted to determine the transformation process when creating dataset
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),  # transforms input imagess to tensor type to compute gradients easier.
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # adjust the distribution of the dataset statistically
        torchvision.transforms.Grayscale()  # convert the RGB images to greyscale
    ])
    # download train dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # number of samples in validation set from each class
    valid_from_class = len(trainset) // 100
    # initialize indice recording:
    indice_dict = {}
    # initialize index list:
    idxs = []
    for i in range(10):
        indice_dict[i] = [] # create an index list for each class
    indice = 0

    #  record the indices at which each class appeares:
    for data in trainset:
        indice_dict[data[1]].append(indice)
        indice += 1
    #  random sampling indexes from the indice lists to determine who are going to be selected as validation data
    for i in range(10):
        idxs = idxs + random.sample(indice_dict[i], valid_from_class) #  validation data indexes in train set
     # removing validation indexes from the train set indexes:
    train_list = list(range(len(trainset)))
    for idx in idxs: train_list.remove(idx)


    trainset_new = torch.utils.data.Subset(trainset, train_list)  # validation removed version of trainset
    valid_set = torch.utils.data.Subset(trainset, idxs)  # validation set is created with random samples

    # download train dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           transform=transform)

    #  creating the batches of train, validation and test sets as iterables:
    train_generator = torch.utils.data.DataLoader(trainset_new, batch_size=batchsize, shuffle=True)
    validation_generator = torch.utils.data.DataLoader(valid_set, batch_size=4096, shuffle=False)
    test_generator = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

    # if bool_visualize is True: prints random samples from the dataset out:
    if bool_visualize is True:
        visualize.imshow(train_generator)
    return train_generator, validation_generator, test_generator