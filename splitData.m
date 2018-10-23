function [class, tstClass, trnClass] = splitData(M, spltPerc, seed)
r = size(M.C,3); c = size(M.C,4); % # of rows and columns
linC = reshape(M.C,3,3,r*c,1); % "one" row of pixels, ie. 3x3xr*c
%Create cell array for all classes, test set and training set
class = cell(1,16);
tstClass = cell(1,16);
trnClass = cell(1,16);
rng(seed); %set random seed for reproduceability
for i = 1 : 16
    % seperate data into classes
    linIdx = find(M.gtruth == i - 1);
    class{i} = linC(:,:,linIdx);
    % randomly split data
    classLength = size(class{i},3);
    rndIdx = randperm(classLength);
    spltIdx = round(spltPerc * classLength);
    tstClass{i} = class{i}(:,:,rndIdx(1:spltIdx)); 
    trnClass{i} = class{i}(:,:,rndIdx(spltIdx + 1 : end));
end
end