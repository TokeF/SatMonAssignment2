function [class, tstClass, trnClass] = splitData(M, spltPerc, seed)
r = size(M.C,3); c = size(M.C,4); % # of rows and columns
linC = reshape(M.C,3,3,r*c,1); % "one" row of pixels, ie. 3x3xr*c
%Create cell array for all classes, test set and training set
numClasses = 15;
class = cell(1,numClasses);
tstClass = cell(1,numClasses);
trnClass = cell(1,numClasses);
rng(seed); %set random seed for reproduceability
for i = 1 : numClasses
    % seperate data into classes
    linIdx = find(M.gtruth == i );
    class{i} = linC(:,:,linIdx);
%     [ridx,cidx] = ind2sub([r,c], linIdx);
    % randomly split data
    classLength = size(class{i},3);
    rndIdx = randperm(classLength);
    spltIdx = round(spltPerc * classLength);
    tstClass{i} = class{i}(:,:,rndIdx(1:spltIdx)); 
    trnClass{i} = class{i}(:,:,rndIdx(spltIdx + 1 : end));
end
end