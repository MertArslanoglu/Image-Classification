import matplotlib.pyplot as plt
import numpy as np
import torchvision

class visualize:
    def __init__(self):
        self.classes= ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(self, trainloader):
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
