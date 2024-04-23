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
from model import CNN_env
from env_test import env_test
import faulthandler
torch.set_printoptions(profile="full")
faulthandler.enable()
######################
# params             #
######################
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Locations/Location3/full-env-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Locations/Location3/final-env-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Locations/Location1/norm-training-env')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Random/Enrol-files/cap10/full-env-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Wireless/Day3/full-env-training')
#source_file_root = os.path.join('/home/abdurrahman/WiFi-Wired/Day3/full-env-training')
#source_file_root = os.path.join('/media/abdurrahman/Elements/WiFi-Day2/cap10/final-env-training')
#model_root = 'model_wifi_random'
model_root = 'model_wifi_location'
model_path ='/wired3_epoch_env_full3'
source_file_root = '/home/abdurrahman/WiFi-Wired/Stable/Day3/full-env-training'
#source_file_root = '/home/abdurrahman/WiFi-Wireless/Day3/full-env-training'
cuda = True
cudnn.benchmark = True
lr = 3e-4
batch_size = 128
#image_size = 2048
image_size = 4096
num_test = 20
n_epoch = 30
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
   


#Envelope Data
class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
       
        self.transform = transform
        self.samples = []
        for class_dir in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_dir)
            #print(class_dir)
            for hdf5_file in os.listdir(class_dir):
                #print(hdf5_file)
                hdf5_file = os.path.join(class_dir, hdf5_file)
                with h5py.File(hdf5_file, 'r') as f:
                    data = (f['data'][:])
                   
                    #data = data[0:52353600]
                    
                for i in range(data.shape[0]):
                    #self.samples.append((np.concatenate((np.array(data[i,0:2048]),np.array(data[i, 4096:6144]))) , class_dir))
                    self.samples.append((data[i, :], class_dir))
                
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        #data = torch.tensor(data, dtype=torch.float32).view(2, 2048)
        data = torch.tensor(data, dtype=torch.float32).view(2, 4096)
        label = os.path.basename(label)

        if self.transform:
            data = self.transform(data, label)

        
        return data, int(label)
  
def normalize_amplitude(data, label):
    """
    Normalize the amplitude of the input data.
    """
    if label == '8' or label == '5':
        max_amp = torch.max(torch.abs(data))
        if max_amp > 0:
            data = data / max_amp
        return data
    
    else:
        part1 = 0.5*data[:, 0:1900]/np.max(data[0, 0:1900])
        part2 = data[:, 1900:2150]/np.max(data[0, 1900:2150])
        part3 = 0.5*data[:, 2150:4096]/np.max(data[0, 2150:4096])

        data = np.concatenate((part1, part2, part3))
        '''
        part1 = 0.5*data[1, 0:1900]/np.max(data[1, 0:1900])
        part2 = data[1, 1900:2150]/np.max(data[1, 1900:2150])
        part3 = 0.5*data[1, 2150:4096]/np.max(data[1, 2150:4096])
        data[1, :] = np.concatenate((part1, part2, part3))
        '''
        print(data.shape)
        return data
    
##############################
#        Data Loading        #
##############################
training_dataset = HDF5Dataset(root_dir= source_file_root, transform=None)
#training_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Random/Enrol-files/cap10/full-env-training', transform=None)
#training_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Locations/Location3/final-env-training', transform=None)
#training_dataset = HDF5Dataset(root_dir='/media/abdurrahman/Elements/WiFi-Day2/cap10/final-env-training', transform=None)
#training_dataset = HDF5Dataset(root_dir= '/home/abdurrahman/WiFi-Locations/Location2/full-env-training', transform=None)



dataloader_source = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=1
)




#  load model   
my_net = CNN_env()
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


#  Defining the losses  
loss_classification = torch.nn.CrossEntropyLoss()

# moving the objects to cuda 
if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()


for p in my_net.parameters():
    p.requires_grad = True

#############################
#     training network      #
#############################


len_dataloader = len(dataloader_source)


current_step = 0
for epoch in range(n_epoch):

    ###################################
    #          data training          #
    ###################################
    data_source_iter = iter(dataloader_source)
    i = 0
    n_total = 0
    n_correct = 0
    
    while (i<len_dataloader):

        data_source = data_source_iter.__next__()
        
        s_img, s_label = data_source
        s_img = s_img.unsqueeze(1)
        
        s_label = s_label - 1
    
     
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
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(step_training_acc)), step_training_acc)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title("Training Classification")
    plt.savefig('/home/abdurrahman/step_training_indoor2.png', dpi=300, bbox_inches='tight') 
      
        #new
    accu = (n_correct * 1.0 / n_total)*100
    print ('Epoch: %d, Training Loss: %f, Training_Acc: %f%%' % (epoch, source_classification.data.cpu().numpy(), accu))
    
    
    training_loss.append(source_classification.data.cpu().numpy())
    training_acc.append(accu)
        # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    
    torch.save(my_net.state_dict(), model_root + model_path + str(epoch) + '_'+ str(int(accu))+'.pth')
    env_test(epoch=epoch, acc = int(accu))


   

print ('done')
