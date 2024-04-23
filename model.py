import torch.nn as nn
import random


class CNN(nn.Module):

    def __init__(self, code_size=100, n_class=15): #10=>25
        super(CNN, self).__init__()
        self.code_size = code_size
        self.pool_size = (1,2)
        self.stride_size = (1,2)
        self.filter_size = (1,65)
        self.padding = int((self.filter_size[1] - 1)/2)

        
        self.RFFP_extractor = nn.Sequential()
        #dim = 32x1x2x8192
        self.RFFP_extractor.add_module('conv_se1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se1', nn.BatchNorm2d(32, affine=False))
        self.RFFP_extractor.add_module('ac_se1', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se1', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 2, 32x32x2x4096
        self.RFFP_extractor.add_module('conv_se2', nn.Conv2d(in_channels=32, out_channels=48, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se2', nn.BatchNorm2d(48, affine=False))
        self.RFFP_extractor.add_module('ac_se2', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se2', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 3, 32x48x2x2048
        self.RFFP_extractor.add_module('conv_se3', nn.Conv2d(in_channels=48, out_channels=64, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se3', nn.BatchNorm2d(64, affine=False))
        self.RFFP_extractor.add_module('ac_se3', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se3', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 4, 32x64x2x1024
        self.RFFP_extractor.add_module('conv_se4', nn.Conv2d(in_channels=64, out_channels=76, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se4', nn.BatchNorm2d(76, affine=False))
        self.RFFP_extractor.add_module('ac_se4', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se4', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 5, 32x76x2x512
        self.RFFP_extractor.add_module('conv_se5', nn.Conv2d(in_channels=76, out_channels=96, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se5', nn.BatchNorm2d(96, affine=False))
        self.RFFP_extractor.add_module('ac_se5', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se5', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 6, 32x96x2x256
        self.RFFP_extractor.add_module('conv_se6', nn.Conv2d(in_channels=96, out_channels=110, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se6', nn.BatchNorm2d(110, affine=False))
        self.RFFP_extractor.add_module('ac_se6', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se6', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))


        #FC layers, 32x110x2x128
        self.fc = nn.Sequential()
        #self.fc.add_module('fc_se3', nn.Linear(in_features=64 * 2 * 110, out_features=code_size))
        self.fc.add_module('fc_se3', nn.Linear(in_features=128 * 2 * 110, out_features=code_size))
        self.fc.add_module('Dropout1', nn.Dropout(0.5))
        self.fc.add_module('ac_se3', nn.LeakyReLU(True))

        # classify 16 numbers
        self.RFFP_classifier = nn.Sequential()
        self.RFFP_classifier.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        self.RFFP_classifier.add_module('Dropout2', nn.Dropout(0.5))
        self.RFFP_classifier.add_module('relu_se4', nn.LeakyReLU(True))
        self.RFFP_classifier.add_module('fc_se5', nn.Linear(in_features=100, out_features=64))
        self.RFFP_classifier.add_module('Dropout3', nn.Dropout(0.5))
        self.RFFP_classifier.add_module('relu_se5', nn.LeakyReLU(True))
        self.RFFP_classifier.add_module('fc_se6', nn.Linear(in_features=64, out_features=n_class))



    def forward(self, input_data):

        # shared encoder
        RFFP_feat = self.RFFP_extractor(input_data)
        #print("RFFP_feat:", RFFP_feat.shape)
        RFFP_feat = RFFP_feat.view(-1, 110 * 128 * 2)
        #RFFP_feat = RFFP_feat.view(-1, 110 * 64 * 2)
        #print("RFFP_feat:", RFFP_feat.shape)
        shared_code = self.fc(RFFP_feat)
        return self.RFFP_classifier(shared_code)
    

class CNN_env(nn.Module):

    def __init__(self, code_size=100, n_class=15): #10=>25
        super(CNN_env, self).__init__()
        self.code_size = code_size
        self.pool_size = (1,2)
        self.stride_size = (1,2)
        self.filter_size = (1,65)
        self.padding = int((self.filter_size[1] - 1)/2)
       

        
        self.RFFP_extractor = nn.Sequential()
        #dim = 32x1x2x4096
        self.RFFP_extractor.add_module('conv_se1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se1', nn.BatchNorm2d(32, affine=False))
        self.RFFP_extractor.add_module('ac_se1', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se1', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 2, 32x32x2x2048
        self.RFFP_extractor.add_module('conv_se2', nn.Conv2d(in_channels=32, out_channels=48, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se2', nn.BatchNorm2d(48, affine=False))
        self.RFFP_extractor.add_module('ac_se2', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se2', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 3, 32x48x2x1024
        self.RFFP_extractor.add_module('conv_se3', nn.Conv2d(in_channels=48, out_channels=64, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se3', nn.BatchNorm2d(64, affine=False))
        self.RFFP_extractor.add_module('ac_se3', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se3', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 4, 32x64x2x512
        self.RFFP_extractor.add_module('conv_se4', nn.Conv2d(in_channels=64, out_channels=76, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se4', nn.BatchNorm2d(76, affine=False))
        self.RFFP_extractor.add_module('ac_se4', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se4', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 5, 32x76x2x256
        self.RFFP_extractor.add_module('conv_se5', nn.Conv2d(in_channels=76, out_channels=96, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se5', nn.BatchNorm2d(96, affine=False))
        self.RFFP_extractor.add_module('ac_se5', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se5', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))
        
        #layer 6, 32x96x2x128
        self.RFFP_extractor.add_module('conv_se6', nn.Conv2d(in_channels=96, out_channels=110, kernel_size=self.filter_size,
                                                                  padding=(0,self.padding)))
        self.RFFP_extractor.add_module('batch_norm_se6', nn.BatchNorm2d(110, affine=False))
        self.RFFP_extractor.add_module('ac_se6', nn.LeakyReLU(True))
        self.RFFP_extractor.add_module('pool_se6', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.stride_size))


        #FC layers, 32x110x2x64
        self.fc = nn.Sequential()
        self.fc.add_module('fc_se3', nn.Linear(in_features=64 * 2 * 110, out_features=code_size))
        self.fc.add_module('Dropout1', nn.Dropout(0.5))
        self.fc.add_module('ac_se3', nn.LeakyReLU(True))

        # classify 16 numbers
        self.RFFP_classifier = nn.Sequential()
        self.RFFP_classifier.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        self.RFFP_classifier.add_module('Dropout2', nn.Dropout(0.5))
        self.RFFP_classifier.add_module('relu_se4', nn.LeakyReLU(True))
        self.RFFP_classifier.add_module('fc_se5', nn.Linear(in_features=100, out_features=64))
        self.RFFP_classifier.add_module('Dropout3', nn.Dropout(0.5))
        self.RFFP_classifier.add_module('relu_se5', nn.LeakyReLU(True))
        self.RFFP_classifier.add_module('fc_se6', nn.Linear(in_features=64, out_features=n_class))



    def forward(self, input_data):

        # shared encoder
        RFFP_feat = self.RFFP_extractor(input_data)
        RFFP_feat = RFFP_feat.view(-1, 110 * 64 * 2)
        shared_code = self.fc(RFFP_feat)
        return self.RFFP_classifier(shared_code)
    