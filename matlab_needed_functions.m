% Comment (Ctrl+R)

% Clear the workspace
clear;

% To load annotation data (essentially an excel sheet) from provided
% dataset
global annotations;
annotations = load("car_ims/cars_annos.mat", "annotations").annotations;
classes = string(load("car_ims/cars_annos.mat", "class_names").class_names);
labels = categorical(string({annotations.class}));

% For each loop over the structure array and edit relative path of all
% images
for car = annotations
    car.relative_im_path = "car_ims/" + car.relative_im_path;
end

% Load the resnet50 CNN
net = alexnet;

% Figure out how to fine-tune
img_location = "car_ims/car_ims";
imds = imageDatastore(img_location, "Labels", labels, "ReadFcn", @loadCroppedImage);
% imds.ReadFcn = @loadCroppedImage;
% disp(imds);
% preprocess images by their bbox

% To display some images:
visImds = splitEachLabel(imds, 1, 'randomize');

% for img = 1:6
%     subplot(2,3,img);
%     imshow(visImds.readimage(img));
%     title(classes(img));
% end

% Now to fine-tune resnet on our images:
% use 70% of imds for training and other 30% of imds for validation to test
% for accuracy
[trainDS, testDS] = splitEachLabel(imds, 0.7, 'randomized');

% Now check how many images from each class there are in the train
% datastore
% tbl = countEachLabel(trainDS);

% get the layers of resnet50
layers = net.Layers;

% Alter Layer 175 and 177
% Layer 175: Fully Connected Layer Needs to Look at 196 categories instead
% of the default 1000
% Layer 177: Classification Layer Needs to Output 1 of 196 numbers for the
% car classes

num_objects = height(trainDS.countEachLabel);
disp("Number of categories: " + num_objects);
layers(23) = fullyConnectedLayer(num_objects, "Name", "fc8");
layers(25) = classificationLayer("Name", "ClassificationLayer_fc8");

% Increase the Learning Rate Factor for the last couple layers so they
% change faster than the rest of the network
layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).BiasLearnRateFactor = 20;

% Let's visualize the training in progress by plotting its accuracy as we
% are training the model
% Also, stop the training early if validation accuracy doesn't increase
% much

functions = {
%     @plotTrainingAccuracy, ...
    @(info) stopTrainingatThreshold(info, 99.5)
};

% Finally we will FINE-TUNE the resnet50!

miniBatchSize = 16;
maxEpochs = 10;
opts = trainingOptions("sgdm", ...
    "Verbose", true, ...
    "LearnRateSchedule", "none", ...
    "InitialLearnRate", 0.0001, ...
    "MaxEpochs", maxEpochs, ...
    "MiniBatchSize", miniBatchSize);

% figure;
tic
disp("Initialization may take up to a minute before training begins")
train_net = trainNetwork(trainDS, layers, opts);
toc

% Test the network (may take some time)
testNetwork = false;

if testNetwork
    tic
    [label_preds, err_test] = classify(train_net, testDS, "MiniBatchSize", 64);
    toc
    confMat = confusionmat(testDS.Labels, label_preds);
    confMat = confMat./sum(confMat,2);
    mean(diag(confMat));
% else % This is reached when all test images have been classified and saved in a mat file
%     load savedLabels
end

function returnCrop = loadCroppedImage(filename)
    global annotations;
    I = imread(filename);
    if ismatrix(I)
        I = cat(3, I, I, I);
    end
    x = split(filename, "\");
    x = x(end);
    x = split(x, ".");
    x = x(1);
    index = str2double(x);
    data = annotations(index);
    cropper = imcrop(I, [data.bbox_x1, data.bbox_y1, data.bbox_x2-data.bbox_x1, data.bbox_y2-data.bbox_y1]);
    returnCrop = imresize(cropper, [227, 227]);
end

% Oh, cool! Mathworks has this universal CNN function to predict item's
% class

% YPred = classify(net, imds);

% Where net is the neural network
%   and imds is the image data store (which holds all the images to
%   classify on)