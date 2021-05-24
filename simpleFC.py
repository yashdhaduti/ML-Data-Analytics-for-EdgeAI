import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import csv
import time

# Argument parser
parser = argparse.ArgumentParser(description='EE397K HW1 - SimpleFC')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)]), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.8)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = F.relu(self.linear2(out))
        out = self.dropout(out)
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        out = self.linear4(out)
        return out


model = SimpleFC(input_size, num_classes)
model = model.to(torch.device('cuda'))

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

train_time = 0
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
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
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
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size)
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

plt.figure()
plt.grid(True)
plt.plot(train_losses, label='Training losses')
plt.plot(test_losses, label = 'Test losses')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.savefig('Q2_Loss_08.png')
plt.figure()
plt.grid(True)
plt.plot(train_accuracy, label='Training accuracies')
plt.plot(test_accuracy, label = 'Test accuracies')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('% Accuracy')
plt.title('% Accuracy vs Epoch')
plt.savefig('Q2_Accuracy_08.png')
wtr = csv.writer(open ('Q2_loss_08.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(train_losses)
wtr.writerow(test_losses)
wtr = csv.writer(open ('Q2_accuracy_08.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow(train_accuracy)
wtr.writerow(test_accuracy)
wtr = csv.writer(open ('Q2_train_time_08.csv', 'w'), delimiter=',', lineterminator='\n')
wtr.writerow([train_time])
