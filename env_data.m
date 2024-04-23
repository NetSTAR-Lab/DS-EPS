
files = [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];

for i=1:length(files)

filename = "/media/abdurrahman/Crucial X6/Wifib_day1/Loc3/IQ-files/dev"+files(i)+".h5";
file_id = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');

% Specify the name of the dataset you want to read
dataset_name = '/data';

% Open the dataset and obtain a dataset identifier
dataset_id = H5D.open(file_id, dataset_name);

% Get the dataspace of the dataset
dataspace_id = H5D.get_space(dataset_id);

% Get the dimensions of the dataset
dims = fliplr(H5S.get_simple_extent_dims(dataspace_id));

% Read the dataset into a variable
data = H5D.read(dataset_id, 'H5ML_DEFAULT', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');
%I = complex(data(1:25170, 1), data(25171:50340, 1));
I = complex(data(1:10000, 1), data(25171:35170, 1));
figure
plot(angle(I))
xlim([0 25170])
xlim([0 10000])


DownsampleFactor = 15;


lp1 = dsp.FIRFilter(Numerator=firpm(20,[0 0.03 0.1 1],[1 1 0 0]));
size(data)
sig1 = data(1:25170, :); 
sig2 = data(25171:50340 ,:);


N = 60; % Filter order
hilbertTransformer = dsp.FIRFilter( ...
        Numerator=firpm(N,[0.01 .95],[1 1],"hilbert"));
delay = dsp.Delay(Length=N/2);
lp2 = dsp.FIRFilter(Numerator=firpm(20,[0 0.03 0.1 1],[1 1 0 0]));


% Envelope detector using the Hilbert transform in the time domain
sige = abs(complex(0, hilbertTransformer(sig1)) + delay(sig1));
sigenv1 =lp2(downsample(sige,DownsampleFactor));
sigenv1 = sigenv1 - mean(sigenv1);


% Envelope detector using the Hilbert transform in the time domain
sige = abs(complex(0, hilbertTransformer(sig2)) + delay(sig2));
sigenv2 =lp2(downsample(sige,DownsampleFactor));

sigenv2 = sigenv2 - mean(sigenv2);

sigenv = complex(sigenv1, sigenv2);

frequencyLimits = [-0.1 0.1]*pi; % full-env Normalized frequency (rad/sample)
%frequencyLimits = [0 0.1]*pi; % full-env Normalized frequency (rad/sample)

[Pd1, Fd1] = pspectrum((sigenv1), ...
    'FrequencyLimits',frequencyLimits, 'TwoSided',true);
 
[Pd2, Fd2] = pspectrum((sigenv2), ...
    'FrequencyLimits',frequencyLimits, 'TwoSided',true);


frames = [Pd1 ; Pd2];
% Close the dataspace, dataset, and file identifiers
H5S.close(dataspace_id);
H5D.close(dataset_id);
H5F.close(file_id);

%--------------------------------------------------------------------------
%Writing the new one. 

%Define the file name and dataset name
filename = "/media/abdurrahman/Crucial X6/Wifib_day1/Loc3/Envelope-files/dev"+files(i)+".h5";
datasetname = '/data';

% Create the HDF5 file
h5create(filename, datasetname, [8192 size(frames,2)]);

% Write the matrix to the HDF5 file
h5write(filename, datasetname, frames);

close all
end
