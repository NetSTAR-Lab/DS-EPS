import os
import h5py
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from model import CNN_env
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix
import seaborn as sns

testing_acc = []

################################################################################################################
#
# Test function for envelope data
#
#################################################################################################################
def env_test(epoch, acc):
    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 4096
    model_path = 'wireless2_epoch_env_full1'
    #testing_path = '/home/abdurrahman/WiFi-Random/Test3/full-env-testing'
    #testing_path = '/home/abdurrahman/WiFi-Locations/Location1/full-env-testing'
    testing_path = '/home/abdurrahman/WiFi-Wireless/Unstable/Day1/full-env-testing'

    conf_name = 'wireless1_env_fold1_conf1_min1.txt'
    flag = 1
    #image_size = 2048

    ############################
    #      Dataset Class       #
    ############################

    
    #Envelope Data
    class HDF5Dataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, transform=None):
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
            if self.transform:
                data = self.transform(data)

            label = os.path.basename(label)
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
            return data
    
    ##############################
    #        Data Loading        #
    ##############################
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Day2/cap10/testing-Envelope', transform=normalize_amplitude)
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Random/Test-files/cap10/full-env-testing', transform=None)
    testing_dataset = HDF5Dataset(root_dir= testing_path, transform=None)
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Locations/Location1/full-env-testing', transform=None)
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Locations/Location3/testing-Envelope', transform=None)
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/WiFi-Wired/Day3/full-env-testing', transform=None)
    
    
    #testing_dataset = HDF5Dataset(root_dir='/home/abdurrahman/Dir2/testing', window_size=8390)

    testing_dataloader = torch.utils.data.DataLoader(
        dataset=testing_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    #model_root = 'model_wifi_random'
    model_root = 'model_wifi_location'
    #model_root = 'model_wifi_random'
    labels = ['29', '2', '30', '40', '15', '6', '7', '8', '28', '10', '11', '36', '37', '39', '46']
    #labels = ['29', '2', '40', '15', '6', '7', '8', '10', '11', '28', '36', '37', '39', '46']
    #labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    ####################
    # load model       #
    ####################

    my_net = CNN_env()
    #checkpoint = torch.load(os.path.join(model_root, 'day2_epoch_env_' + str(epoch) + '_' + str(acc) + '.pth'))
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
        #print("result", result)
        #print(len(result))
        pred = result.data.max(1, keepdim=True)[1]
        #print("pred", pred)
        #print(classv_label.data.view_as(pred))
        # Create confusion matrix
        cm+= confusion_matrix(classv_label.data.view_as(pred).cpu(), pred.cpu(), labels=range(15))

        #all_cm += cm
        #print(classv_label.data.view_as(pred).shape)
        #print(pred.shape)
        #print(cm.shape)
        #print("after pred")
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = (n_correct * 1.0 / n_total)*100
    testing_acc.append(accu)
    print ('epoch: %d, accuracy: %f%%' % (epoch, accu))

    if flag == 1:
        np.savetxt(conf_name, cm)
    '''
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(testing_acc)), testing_acc)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title("Testing Classification")
    plt.savefig('/home/abdurrahman/indoor_testing2.png', dpi=300, bbox_inches='tight')

    '''


'''
    # Plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    #print(cm.shape)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        xlabel='Predicted label', ylabel='True label',
        title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',  xticklabels=labels, yticklabels=labels, ax=ax)

    plt.tight_layout()
    ax.set_ylabel('True Lables', fontsize=13)
    ax.set_xlabel('Predicted Labels', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=11)
    plt.xticks(rotation=0)
    plt.savefig("acc_enrol-enrol.pdf", bbox_inches='tight')
    plt.show()
'''
'''
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    print(cm.shape)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        xlabel='Predicted label', ylabel='True label',
        title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm*100
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, ax=ax)

    plt.tight_layout()
    plt.show()
'''
