import random
import os
import h5py
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model import CNN
from model import LSTMModel
from test import test
import faulthandler




torch.set_printoptions(profile="full")
faulthandler.enable()
######################
# params             #
######################
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Random/Enrol-files/cap10/clean-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Locations/Location1/training-3000')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Locations/Location2/clean-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Wireless/Day1/clean-training')

'''
model_root = 'model_wifi_location'
model_path = '/wired1_epoch_full1_'
training_path = '/home/abdurrahman/WiFi-Wired/Unstable/Day1/clean-training'
'''

model_root = 'model_lstm'
model_path = '/wirelss1_epoch_'
training_path = '/home/abdurrahman/Phase Detection Data/Wireless-Day1/Training'

cuda = True
cudnn.benchmark = True
lr = 3e-4
batch_size = 128
image_size = 8192
num_test = 20
n_epoch = 20
step_decay_weight = 0.95
lr_decay_step = 3000
weight_decay = 1e-6
momentum = 0.9

training_loss = []
training_acc = []
step_training_acc = []

#generating random seeds 
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


# Dataset Class 
class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, window_size, transform=None):
        self.root_dir = root_dir
        self.window_size = window_size
        self.transform = transform
        self.samples = []
        for class_dir in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_dir)
            #print(class_dir)
            for hdf5_file in os.listdir(class_dir):
                #print(hdf5_file)
                hdf5_file = os.path.join(class_dir, hdf5_file)
                with h5py.File(hdf5_file, 'r') as f:
                    data = f['data'][:]
                    #data = data[0:52353600]
                for i in range(len(data)):
                    self.samples.append((np.concatenate((data[i, 0:window_size], data[i, 25170:25170+window_size])), class_dir))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        data = torch.tensor(data, dtype=torch.float32).view(2, 8192)
        if (self.transform):
            data = self.transform(data)
        label = os.path.basename(label)
        return data[:, 0:8192], int(label)
  
def normalize_amplitude(data):
    """
    Normalize the amplitude of the input data.
    """
    max_amp = torch.max(torch.abs(data))
    if max_amp > 0:
        data = data / max_amp
    return data
############################################################################################################################
#                                                   Data Loading                                                          #
###########################################################################################################################
#training_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Random/Enrol-files/cap10/clean-training', window_size=8390)
#training_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Locations/Location2/clean-training', window_size=8192)
training_dataset = HDF5Dataset(root_dir=training_path, window_size=8192)

dataloader_source = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=1
)




#  load model   
my_net = CNN()

total_params = sum(p.numel() for p in my_net.parameters())
print(f"Number of parameters: {total_params}")
# setup optimizer 
def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print ('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


# Defining the losses  
loss_classification = torch.nn.CrossEntropyLoss()

# Moving Objects to cuda 
if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()


for p in my_net.parameters():
    p.requires_grad = True


#######################################################################################################
#                                          training network                                           #
#######################################################################################################


len_dataloader = len(dataloader_source)


current_step = 0
for epoch in range(n_epoch):

    ################################################################################################
    #                                      data training                                           #
    ################################################################################################

    data_source_iter = iter(dataloader_source)
    i = 0
    n_total = 0
    n_correct = 0
    
    while (i<len_dataloader):

        data_source = data_source_iter.__next__()
        
        s_img, s_label = data_source
        s_img = s_img.unsqueeze(1)
        
        #There is no need => 1 stable, 0 unstable
        #s_label = s_label - 1
    
     
        #print('source:', t_img.shape)
        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 1, 2, image_size)
        class_label = torch.LongTensor(batch_size)
       

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
          
        input_img.resize_as_(input_img).copy_(s_img)
        class_label.resize_as_(class_label).copy_(s_label)
        source_inputv_img = Variable(input_img)
        source_classv_label = Variable(class_label)

   
        
        result = my_net(input_data=source_inputv_img)
        
        #print("Source Samples:", source_inputv_img)
        source_class_label = result
        #print("Class labels:", source_class_label.shape)
        #print("Class labels", source_class_label)
        source_classification = loss_classification(source_class_label, source_classv_label)
        #print("Predicted label:", source_class_label)
        #print("Correct label:", source_classv_label-1)

        loss = source_classification

       
        loss.backward()
      
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()
        pred = result.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(source_classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1
        current_step += 1
        
        accu = (n_correct * 1.0 / n_total)*100
        step_training_acc.append(accu)
    #fig,ax = plt.subplots()
    #ax.plot(np.arange(len(step_training_acc)), step_training_acc)
    #ax.set_xlabel('Epochs')
    #ax.set_ylabel('Losses')
    #ax.set_title("Training Classification")
    #plt.savefig('/home/abdurrahman/step_training_indoor2.png', dpi=300, bbox_inches='tight') 
      
        #new
    accu = (n_correct * 1.0 / n_total)*100
    print ('Epoch: %d, source_classification: %f, calssification_Acc: %f%%' % (epoch, source_classification.data.cpu().numpy(), accu))
    
    
    training_loss.append(source_classification.data.cpu().numpy())
    training_acc.append(accu)
        # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    
    torch.save(my_net.state_dict(), model_root + model_path + str(epoch) + '_'+ str(int(accu))+'.pth')
    test(epoch=epoch, acc = int(accu))


fig,ax = plt.subplots()
ax.plot(np.arange(len(training_acc)), training_acc)
ax.set_xlabel('Epochs')
ax.set_ylabel('Losses')
ax.set_title("Training Classification")
plt.savefig('/home/abdurrahman/indoor_training2.png', dpi=300, bbox_inches='tight')    

print ('done')












'''
import random
import os
import h5py
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model import CNN
from model import LSTMModel
from test import test
import faulthandler




torch.set_printoptions(profile="full")
faulthandler.enable()
######################
# params             #
######################
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Random/Enrol-files/cap10/clean-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Locations/Location1/training-3000')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Locations/Location2/clean-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Wireless/Day1/clean-training')

'''
model_root = 'model_wifi_location'
model_path = '/wired1_epoch_full1_'
training_path = '/home/abdurrahman/WiFi-Wired/Unstable/Day1/clean-training'
'''

model_root = 'model_lstm'
model_path = '/wirelss1_epoch_'
training_path = '/home/abdurrahman/Phase Detection Data/Wireless-Day1/Training'

cuda = True
cudnn.benchmark = True
lr = 3e-4
batch_size = 128
image_size = 8192
num_test = 20
n_epoch = 20
step_decay_weight = 0.95
lr_decay_step = 3000
weight_decay = 1e-6
momentum = 0.9

training_loss = []
training_acc = []
step_training_acc = []

#generating random seeds 
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


# Dataset Class 
class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, window_size, transform=None):
        self.root_dir = root_dir
        self.window_size = window_size
        self.transform = transform
        self.samples = []
        for class_dir in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_dir)
            #print(class_dir)
            for hdf5_file in os.listdir(class_dir):
                #print(hdf5_file)
                hdf5_file = os.path.join(class_dir, hdf5_file)
                with h5py.File(hdf5_file, 'r') as f:
                    data = f['data'][:]
                    #data = data[0:52353600]
                for i in range(len(data)):
                    self.samples.append((np.concatenate((data[i, 0:window_size], data[i, 25170:25170+window_size])), class_dir))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        data = torch.tensor(data, dtype=torch.float32).view(2, 8192)
        if (self.transform):
            data = self.transform(data)
        label = os.path.basename(label)
        return data[:, 0:8192], int(label)
  
def normalize_amplitude(data):
    """
    Normalize the amplitude of the input data.
    """
    max_amp = torch.max(torch.abs(data))
    if max_amp > 0:
        data = data / max_amp
    return data
############################################################################################################################
#                                                   Data Loading                                                          #
###########################################################################################################################
#training_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Random/Enrol-files/cap10/clean-training', window_size=8390)
#training_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Locations/Location2/clean-training', window_size=8192)
training_dataset = HDF5Dataset(root_dir=training_path, window_size=8192)

dataloader_source = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=1
)




#  load model   
my_net = CNN()

total_params = sum(p.numel() for p in my_net.parameters())
print(f"Number of parameters: {total_params}")
# setup optimizer 
def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print ('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


# Defining the losses  
loss_classification = torch.nn.CrossEntropyLoss()

# Moving Objects to cuda 
if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()


for p in my_net.parameters():
    p.requires_grad = True


#######################################################################################################
#                                          training network                                           #
#######################################################################################################


len_dataloader = len(dataloader_source)


current_step = 0
for epoch in range(n_epoch):

    ################################################################################################
    #                                      data training                                           #
    ################################################################################################

    data_source_iter = iter(dataloader_source)
    i = 0
    n_total = 0
    n_correct = 0
    
    while (i<len_dataloader):

        data_source = data_source_iter.__next__()
        
        s_img, s_label = data_source
        s_img = s_img.unsqueeze(1)
        
        #There is no need => 1 stable, 0 unstable
        #s_label = s_label - 1
    
     
        #print('source:', t_img.shape)
        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 1, 2, image_size)
        class_label = torch.LongTensor(batch_size)
       

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
          
        input_img.resize_as_(input_img).copy_(s_img)
        class_label.resize_as_(class_label).copy_(s_label)
        source_inputv_img = Variable(input_img)
        source_classv_label = Variable(class_label)

   
        
        result = my_net(input_data=source_inputv_img)
        
        #print("Source Samples:", source_inputv_img)
        source_class_label = result
        #print("Class labels:", source_class_label.shape)
        #print("Class labels", source_class_label)
        source_classification = loss_classification(source_class_label, source_classv_label)
        #print("Predicted label:", source_class_label)
        #print("Correct label:", source_classv_label-1)

        loss = source_classification

       
        loss.backward()
      
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()
        pred = result.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(source_classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1
        current_step += 1
        
        accu = (n_correct * 1.0 / n_total)*100
        step_training_acc.append(accu)
    #fig,ax = plt.subplots()
    #ax.plot(np.arange(len(step_training_acc)), step_training_acc)
    #ax.set_xlabel('Epochs')
    #ax.set_ylabel('Losses')
    #ax.set_title("Training Classification")
    #plt.savefig('/home/abdurrahman/step_training_indoor2.png', dpi=300, bbox_inches='tight') 
      
        #new
    accu = (n_correct * 1.0 / n_total)*100
    print ('Epoch: %d, source_classification: %f, calssification_Acc: %f%%' % (epoch, source_classification.data.cpu().numpy(), accu))
    
    
    training_loss.append(source_classification.data.cpu().numpy())
    training_acc.append(accu)
        # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    
    torch.save(my_net.state_dict(), model_root + model_path + str(epoch) + '_'+ str(int(accu))+'.pth')
    test(epoch=epoch, acc = int(accu))


fig,ax = plt.subplots()
ax.plot(np.arange(len(training_acc)), training_acc)
ax.set_xlabel('Epochs')
ax.set_ylabel('Losses')
ax.set_title("Training Classification")
plt.savefig('/home/abdurrahman/indoor_training2.png', dpi=300, bbox_inches='tight')    

print ('done')
'''
