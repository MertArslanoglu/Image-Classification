import torch
# example mlp classifier
class Net(torch.nn.Module):
    """Model definitions which are reuqired with the homework experiments.
        network and activation function is requested from the user as an input when initialization of the class.
        forward function handles the forward propagation process.
    """

    def __init__(self, network, act):
        super(Net, self).__init__()
        self.network = network
        if network == "default":
            self.input_size = 1024
            self.first_layer = torch.nn.Linear(1024, 128)
            self.fc2 = torch.nn.Linear(128, 10)
            if act == "relu":
                self.act = torch.nn.ReLU()
            if act == "sigmoid":
                self.act = torch.nn.Sigmoid()


        if network == "mlp_1":
            self.input_size = 1024
            self.first_layer = torch.nn.Linear(self.input_size, 32)
            self.fc2 = torch.nn.Linear(32, 10)
            if act == "relu":
                self.act = torch.nn.ReLU()
            if act == "sigmoid":
                self.act = torch.nn.Sigmoid()

        if network == "mlp_2":
            self.input_size = 1024
            self.first_layer = torch.nn.Linear(self.input_size, 32)
            self.fc2 = torch.nn.Linear(32, 64, bias=False)
            self.fc3 = torch.nn.Linear(64, 10)
            if act == "relu":
                self.act = torch.nn.ReLU()
            if act == "sigmoid":
                self.act = torch.nn.Sigmoid()

        if network == "cnn_3":
            self.first_layer = torch.nn.Conv2d(1, 16 ,(3,3), stride=1)
            self.conv2 = torch.nn.Conv2d(16, 8, (5, 5), stride=1)
            self.conv3 = torch.nn.Conv2d(8, 16, (7, 7), stride=1)
            self.pool = torch.nn.MaxPool2d((2, 2))
            self.input_size = 9*16
            self.fc1 = torch.nn.Linear(self.input_size, 10)
            if act == "relu":
                self.act = torch.nn.ReLU()
            if act == "sigmoid":
                self.act = torch.nn.Sigmoid()

        if network == "cnn_4":
            self.first_layer = torch.nn.Conv2d(1, 16, (3, 3), stride=1)
            self.conv2 = torch.nn.Conv2d(16, 8, (3, 3), stride=1)
            self.conv3 = torch.nn.Conv2d(8, 16, (5, 5), stride=1)
            self.conv4 = torch.nn.Conv2d(16, 16, (5, 5), stride=1)
            self.pool = torch.nn.MaxPool2d((2, 2))
            self.input_size = 16 * 16
            self.fc1 = torch.nn.Linear(self.input_size, 10)
            if act == "relu":
                self.act = torch.nn.ReLU()
            if act == "sigmoid":
                self.act = torch.nn.Sigmoid()

        if network == "cnn_5":
            self.first_layer = torch.nn.Conv2d(1, 8, (3, 3), stride=1)
            self.conv2 = torch.nn.Conv2d(8, 16, (3, 3), stride=1)
            self.conv3 = torch.nn.Conv2d(16, 8, (3, 3), stride=1)
            self.conv4 = torch.nn.Conv2d(8, 16, (3, 3), stride=1)
            self.conv5 = torch.nn.Conv2d(16, 16, (3, 3), stride=1)
            self.conv6 = torch.nn.Conv2d(16, 8, (3, 3), stride=1)
            self.pool = torch.nn.MaxPool2d((2, 2))
            self.input_size = 8*16
            self.fc1 = torch.nn.Linear(self.input_size, 10)
            if act == "relu":
                self.act = torch.nn.ReLU()
            if act == "sigmoid":
                self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.network == "default":
            x = x.view(-1, self.input_size)
            hidden = self.first_layer(x)
            relu = self.act(hidden)
            output = self.fc2(relu)

        if self.network == "mlp_1":
            x = x.view(-1, self.input_size)
            hidden = self.act(self.first_layer(x))
            output = self.fc2(hidden)

        if self.network == "mlp_2":
            x = x.view(-1, self.input_size)
            hidden = self.act(self.first_layer(x))
            hidden2 = self.fc2(hidden)
            output = self.fc3(hidden2)

        if self.network == "cnn_3":
            x = self.act(self.first_layer(x))
            x = self.pool(self.act(self.conv2(x)))
            x = self.pool(self.conv3(x))
            x = x.view(-1, self.input_size)
            output = self.fc1(x)

        if self.network == "cnn_4":
            x = self.act(self.first_layer(x))
            x = self.act(self.conv2(x))
            x = self.pool(self.act(self.conv3(x)))
            x = self.pool(self.act(self.conv4(x)))
            x = x.view(-1, self.input_size)
            output = self.fc1(x)

        if self.network == "cnn_5":
            x = self.act(self.first_layer(x))
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))
            x = self.pool(self.act(self.conv4(x)))
            x = self.act(self.conv5(x))
            x = self.pool(self.act(self.conv6(x)))
            x = x.view(-1, self.input_size)
            output = self.fc1(x)




        return output