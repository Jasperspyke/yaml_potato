import torch

print('setting device,')
print('torch.cuda.is_available is')
torch.cuda.is_available()
device = torch.device("cuda:0")
print(device)


print('this code tests for what the current device is. This should be 1')

print('torch.cuda.current_device() ')
print(torch.cuda.current_device())


print('now we will run some tensor tests')
X_train = torch.FloatTensor([0., 1., 2.])
X_train = X_train.to(device)


print('we will now print device name')
torch.cuda.get_device_name(0)


