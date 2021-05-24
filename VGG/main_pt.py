import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import time
from torchsummary import summary
from models.pytorch.vgg_pt import VGG
from models.pytorch.mobilenet_pt import MobileNetv1

# Argument parser
parser = argparse.ArgumentParser(description='EE379K HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch to train')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size

# Each experiment you will do will have slightly different results due to the randomness
# of the initialization value for the weights of the model. In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = VGG()
model = model.to(torch.device('cuda'))
print(model)
summary(model,input_size=(3,32,32))

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

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
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))
    test_accuracy.append(100. * test_correct / test_total)
print("pytorch train time " + str(train_time))
print("pytorch train accuracy " + str(train_accuracy))
print("pytorch test accuracy " + str(test_accuracy))
file = open("out_pt.txt","w")
file.write("train time " + str(train_time) + '\n')
file.write("train accuracy " + str(train_accuracy) + '\n')
file.write("test accuracy " + str(test_accuracy))
file.close()
torch.save(model.state_dict(), 'vgg_pt.pt')