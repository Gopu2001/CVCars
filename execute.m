% Make the file that runs both the detector and recognizer
clear;

% Load the classifier network
net = load("alexnetClassifier.mat");
classes = net.classes;
classifier = net.train_net;

% Load the detector network
global detector;
detector = load("fasterRCNNResNet50EndToEndVehicleExample.mat").detector;

file = fopen("filenames.txt");
data = textscan(file,"%s", 'Delimiter', '\n');
filenames = transpose(string(data{:}));
fclose(file);
disp("Successfully retrieved the filenames of images");

tic
for filename = filenames
    [img, bboxes] = editImage(filename);
    numBoxes = size(bboxes);
    numBoxes = numBoxes(2);
    label = -1;
    for box = 1:numBoxes
        cropper = imcrop(img, bboxes(4*box-3:4*box));
        cropper = imresize(cropper, [227,227]);
        label = classify(classifier, cropper);
    end
    figure;
    imshow(img);
    if ~(class(label) == "categorical")
        title("No Car Found!");
    else
        title(classes(label));
    end
end
toc
% testing time ~17 seconds

function [img,bboxes] = editImage(filename)
    global detector;
    inputSize = [224 224];
    img = imresize(imread(fullfile(filename)),inputSize);
    if ismatrix(img)
        img = cat(3, img, img, img);
    end
    bboxes = transpose(detect(detector,img));
end