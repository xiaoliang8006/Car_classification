import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import data_processing
import torch.optim.lr_scheduler

data = data_processing.load_data(download=False)
new_data = data_processing.convert2onehot(data)
#print(new_data)
new_data = new_data.sample(frac=1).reset_index(drop=True) # shuffle
print(new_data)

# prepare training data
new_data = torch.Tensor(np.array(new_data))       # change to tensor
sep = int(0.7*len(new_data))

train_data = new_data[:sep]                         # training data (70%)
X_test = new_data[sep:, :21]                        # testing data (30%)
y_test = new_data[sep:, 21:]

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=20, shuffle=True)

# build network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 128)   # hidden layer 1
        self.hidden2 = torch.nn.Linear(128, 128)   # hidden layer 2
        self.predict = torch.nn.Linear(128, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer 1
        x = F.relu(self.hidden2(x))      # activation function for hidden layer 2
        x = self.predict(x)             # linear output
        return x

net = Net(21, 4)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)   # optimize all cnn parameters
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.8)
loss_func = torch.nn.CrossEntropyLoss()

# training and testing
for epoch in range(100):
    scheduler.step()
    for step, td in enumerate(train_loader):        # gives batch data
        b_x = td[:, :21]
        b_y = td[:, 21:]
        output = net(b_x)    # output

        #CrossEntropyLoss target
        b_y = torch.argmax(b_y, dim=1)
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 20 == 0:
            output_y = net(X_test)
            _, y_pred = torch.max(output_y, 1)
            _, y_label = torch.max(y_test, 1)
            accuracy = (y_pred == y_label).sum().item() / y_label.size(0)
            print('Epoch: ', epoch, '| train loss: %.6f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
