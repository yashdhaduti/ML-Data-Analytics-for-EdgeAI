import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from thop import profile
import matplotlib.pyplot as plt
import csv
import time
from torchsummary import summary

# Argument parser
parser = argparse.ArgumentParser(description='EE397K HW1 - SimpleCNN')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The number of target classes, you have 10 digits to classify
num_classes = 10

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
#learning_rate = args.lr

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# MNIST Dataset (Images and Labels)
# TODO: Insert here the normalized MNIST dataset
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)]), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)]))

#batch_size = 2048
times = []

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(7 * 7 * 8, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


accuracies = []
learning_rates = [0.3, 0.1, 0.03, 0.01, 0.001, 0.0001]
learning_rate = 0.01
#for learning_rate in learning_rates:
model = SimpleCNN(num_classes)
model = model.to(torch.device('cuda'))
# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
sgdoptimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 5e-4)
rmspropoptimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)
adamoptimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps = 1e-8)

optimizer = adamoptimizer


train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []
train_time = 0;
for epoch in range(num_epochs):
    # Training phase loop
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    train_start = time.time()
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(torch.device('cuda'))
        labels = labels.to(torch.device('cuda'))
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    train_time += (time.time() - train_start)
    train_losses.append(train_loss/len(train_loader))
    train_accuracy.append(100. * train_correct / train_total)
    if epoch == 24:
        accuracies.append(100. * train_correct / train_total)
    # Testing phase loop
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    test_losses.append(test_loss/len(test_loader))
    test_accuracy.append(100. * test_correct / test_total)
macs, params = profile(model, inputs=(torch.randn(1, 1, 28, 28).to(torch.device('cuda')),))
summary(model,input_size=(1,28,28))
print("MACs" + str(macs))
print("Params" + str(params))
#torch.save(model.state_dict(), 'simplecnn_simple.pth')
plt.figure()
plt.grid(True)
plt.plot(train_losses, label='Training losses')
plt.plot(test_losses, label = 'Test losses')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.savefig('Q4_Loss_adam01.png')
wtr = csv.writer(open ('Q4_Loss_adam01.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(train_losses)
wtr.writerow(test_losses)
"""times.append(train_time)
fig = plt.figure()
ax = plt.gca()
plt.grid(True)
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Learning Rate')
plt.semilogx(learning_rates, accuracies)
plt.savefig('Q4_accuracy_learning_rate_2048.png')
wtr = csv.writer(open ('accuracy_learning_rate_2048.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(accuracies)"""


"""fig = plt.figure()
ax = plt.gca()
plt.grid(True)
ax.set_xscale('log')
plt.xlabel('Batch size')
plt.ylabel('Training time')
plt.title('Training Time vs Batch Size')
plt.plot(batches, times)
plt.savefig('Q4_train_time_batches.png')
wtr = csv.writer(open ('Q4_Train_Times.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(times)
"""