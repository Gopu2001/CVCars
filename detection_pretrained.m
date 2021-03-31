% Detector made using tutorial from:
% https://www.mathworks.com/help/vision/ug/object-detection-using-faster-r-cnn-deep-learning.html

% download the example pretrained detector
% just for kicks, coincidentally, this was made to detect cars!

doTrainingAndEval = false;

% if you don't want to train and evaluate yourself, though I want to
% but first, let's try doing this the tutorial's way
if ~doTrainingAndEval && ~exist('fasterRCNNResNet50EndToEndVehicleExample.mat',...
    'file')
    disp("Downloading the pretrained detector ~2 min");
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/fasterRCNNResNet50EndToEndVehicleExample.mat';
    websave('fasterRCNNResNet50EndToEndVehicleExample.mat', pretrainedURL);
end

% this is the dataset that comes with the tutorial
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

% distribute the dataset with a 60-10-30 train-validation-test split
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6*height(vehicleDataset));

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx+1 + floor( 0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

% next step is to create image datastores for our databases
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

trainingData = combine(imdsTrain, bldsTrain);
validationData = combine(imdsValidation, bldsValidation);
testData = combine(imdsTest, bldsTest);

% display one of the training images with provided box labels
% data1 = read(trainingData);
% I = data1{1};
% bbox = data1{2};
% annotatedImage = insertShape(I, 'Rectangle', bbox);
% annotatedImage = imresize(annotatedImage, 2);
% figure;
% imshow(annotatedImage);

% now to create a Faster R-CNN Detection algorithm
% specify the input size of the image to resize all images to
inputSize = [224 224 3];

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(...
    data,inputSize));
preprocessedValidationData = transform(validationData, @(data)preprocessData(...
    data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData, numAnchors);

% use resnet50 for the specific CNN
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
numClasses = width(vehicleDataset)-1;
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,...
    featureExtractionNetwork,featureLayer);

% data augmentation
% essentially expand the dataset without increasing the number of images
% will implement later

% read preprocessed data
data = read(preprocessedTrainingData);

% setup the options for training the faster rcnn
options = trainingOptions("sgdm",...
    "MaxEpochs",10,...
    "MiniBatchSize",2,...
    "InitialLearnRate",1e-3,...
    "CheckpointPath",tempdir,...
    "ValidationData",validationData);

% only do this if you want to train yourself,
% which we opted to not do right away
if doTrainingAndEval
    % train the rcnn detector
    % adjust the negative overlap range and the positive overlap range to
    % make sure that the training samples tightly overlap with the ground
    % truth
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,...
        options,...
        "NegativeOverlapRange", [0 0.3],...
        "PositiveOverlapRange", [0.6 0]);
else
    % load the pretrained detector for the sample
    % which is what I am starting off with doing
    pretrained = load("fasterRCNNResNet50EndToEndVehicleExample.mat");
    detector = pretrained.detector;
end

% define the preprocessData function
function data = preprocessData(data,targetSize)
    % Resize image and bounding boxes to targetSize.
    scale = targetSize(1:2)./size(data{1},[1 2]);
    data{1} = imresize(data{1},targetSize(1:2));
    data{2} = bboxresize(data{2},scale);
end