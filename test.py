import os
import h5py
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from model import CNN
from model import CNN_env
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix

testing_acc = []
def test(epoch, acc):
    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 8192
    #image_size = 4096
    testing_path = '/home/abdurrahman/WiFi-Wireless/Unstable/Day2/clean-testing'
    model_path = 'wireless1_epoch_full1_'
    conf_name = 'wired1_fold1_conf_unstable1.txt'
    flag = 1
    ############################
    #      Dataset Class       #
    ############################

    
    class HDF5Dataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, window_size, transform=None):
            self.root_dir = root_dir
            self.window_size = window_size
            self.transform = transform
            self.samples = []
            for class_dir in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, class_dir)
                for hdf5_file in os.listdir(class_dir):
                    hdf5_file = os.path.join(class_dir, hdf5_file)
                    with h5py.File(hdf5_file, 'r') as f:
                        #print(hdf5_file)
                        data = f['data'][:]
                        #data = data[0:3272100]
                    for i in range(len(data)):
                        self.samples.append((np.concatenate((data[i, 0:window_size], data[i, 25170:25170+window_size])) , class_dir))
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
    
    ##############################
    #        Data Loading        #
    ##############################
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Random/Test-files/cap10/clean-testing', window_size=8390)
    testing_dataset = HDF5Dataset(root_dir= testing_path, window_size=8192)
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Wired/Day3/clean-testing', window_size=8192)

    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Day3/cap10/testing-3000', window_size=8390)

    testing_dataloader = torch.utils.data.DataLoader(
        dataset=testing_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    #model_root = 'model_wifi_random'
    #model_root = 'model_wifi_location'
    #model_root = 'model_wifi_day'
  
    ####################
    # load model       #
    ####################

    my_net = CNN()
    model_root = 'model_wifi_location'
    #checkpoint = torch.load(os.path.join(model_root, 'rand1_epoch_3000_' + str(epoch) + '_' + str(acc) + '.pth'))
    #checkpoint = torch.load(os.path.join(model_root, 'cap10_epoch_' + str(epoch) + '_' + str(acc) + '.pth'))
    #checkpoint = torch.load(os.path.join(model_root, 'rand_epoch_full1_' + str(epoch) + '_' + str(acc) + '.pth'))
    checkpoint = torch.load(os.path.join(model_root, model_path + str(epoch) + '_' + str(acc) + '.pth'))
    #print(checkpoint)
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    if cuda:
        my_net = my_net.cuda()


    len_dataloader = len(testing_dataloader)
    
    #print("number of testing batches:", len_dataloader)
    data_iter = iter(testing_dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    cm = 0
    all_cm = np.zeros((15, 15))

    while i < len_dataloader:
        data_source = data_iter.__next__()
        
        
        s_img, s_label = data_source
      
        s_img = s_img.unsqueeze(1)
        #print(s_label)
        #print(s_img.shape)
        #print(i)
        s_label = s_label - 1
        #print(s_label)
        #print("batch:", s_img)
        #print("batch labels:", s_label)

        #print('source:', t_img.shape)
        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 1, 2, image_size) #change image_size=>2
        class_label = torch.LongTensor(batch_size)
       

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        #print(s_img.shape)
        #print(s_label.shape)  
        #print(input_img.shape)
        #print("before resizing input")
        input_img.resize_as_(input_img).copy_(s_img)
        class_label.resize_as_(class_label).copy_(s_label)
        inputv_img = Variable(input_img)
        #print("after variabling the input")
        classv_label = Variable(class_label)
        #print("after variabling the labels")

        result = my_net(input_data=inputv_img)
        #print("after results")
        #print(result)
        #print(len(result))
        
        pred = result.data.max(1, keepdim=True)[1]
        cm+= confusion_matrix(classv_label.data.view_as(pred).cpu(), pred.cpu(), labels=range(15))
    
        #print("after pred")
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1
    
    accu = (n_correct * 1.0 / n_total)*100
    testing_acc.append(accu)
    print ('epoch: %d, accuracy: %f%%' % (epoch, accu))

    fig,ax = plt.subplots()
    ax.plot(np.arange(len(testing_acc)), testing_acc)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title("Testing Classification")
    plt.savefig('/home/abdurrahman/indoor_testing2.png', dpi=300, bbox_inches='tight')
    
    #print(cm)
    if flag == 1:
        np.savetxt(conf_name, cm)
    



