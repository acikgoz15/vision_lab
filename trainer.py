import torch
import torch.nn as nn
from torch._C import MobileOptimizerType
from loader import FERDataReader
from network import Net
import torch.optim as optim
import numpy as np

batch_size = 20

train_loader = torch.utils.data.DataLoader(FERDataReader(mode='train'), batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(FERDataReader(mode='valid'), batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(FERDataReader(mode='test'), batch_size=batch_size, shuffle=False, num_workers=2)


classes = ['0', '1', '2', '3', '4', '5', '6']
# model = Network()
model = Net()
print(model)

model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
n_epoch = 3
valid_loss_min = np.Inf

for i in range(n_epoch):
    train_loss = 0
    valid_loss = 0
    total_true = 0
    total_false = 0
    model.train()
    for data, label in train_loader:
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        _, prediction = torch.max(output.data, 1)
        # print(prediction)
        # print(label)
        total_true += torch.sum(prediction == label)
        total_false += torch.sum(prediction != label)
    print ("Train Accuracy: "+ str(total_true.item()*1.0/(total_true.item()+total_false.item())))
    total_true = 0
    total_false = 0
    model.eval()
    for data,label in valid_loader:
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, label)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        _, prediction = torch.max(output.data, 1)
        # print(prediction)
        # print(label)
        total_true += torch.sum(prediction == label)
        total_false += torch.sum(prediction != label)
    print ("Val Accuracy: "+ str(total_true.item()*1.0/(total_true.item()+total_false.item())))
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        i, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'drive/My Drive/Colab Notebooks/model_cifar.pt')
        valid_loss_min = valid_loss
