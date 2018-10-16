% function assignment2()
% close all
% M = loadData();
% plot1a(M)
% [tstData, trnData] = testTrainSplit(M);
% end
%% Load data
M = load('flevoland.mat');

%% part 1a
R = M.C(1,1,:,:); G = M.C(2,2,:,:); B = M.C(3,3,:,:);
R = squeeze(R); G = squeeze(G); B = squeeze(B);
R = 10*log10(4*pi*R); 
G = 10*log10(4*pi*G); 
B = 10*log10(4*pi*B);
% RGB = cat(3,R,G,B);
RGB(:,:,1) = mat2gray(R,[min(min(R)) max(max(R))]);
RGB(:,:,2) = mat2gray(G,[min(min(G)) max(max(G))]);
RGB(:,:,3) = mat2gray(B,[min(min(B)) max(max(B))]);
figure(1); 
subplot(1,3,1); imshow(R,[]); title('Red')
subplot(1,3,2); imshow(G,[]); title('Green')
subplot(1,3,3); imshow(B,[]); title('Blue')

% for i = 1:3
%     oldmax = max(max(RGB(:,:,i)))
%     oldmin = min(min(RGB(:,:,i)))
%     RGB(:,:,i) = (((RGB(:,:,i) - oldmin)*2) ./ (oldmax - oldmin)) -1;
% end

figure(2);
imagesc(RGB)
title('RGB image')

figure(3)
imagesc(M.gtruth)
colormap(M.cmap)
title('Ground truth')

%% 1.b split in training and test

r = size(M.C,3); c = size(M.C,4);
linC = reshape(M.C,3,3,r*c,1);
class = cell(1,16);
tstClass = cell(1,16);
trnClass = cell(1,16);
spltPerc = 0.6;
for i = 1 : 16
    % seperate data in to classes
    linIdx = find(M.gtruth == i - 1);
    class{i} = linC(:,:,linIdx);
    % randomly split data
    classLength = size(class{i},3);
    rndIdx = randperm(classLength);
    spltIdx = round(spltPerc * classLength);
    tstClass{i} = class{i}(:,:,rndIdx(1:spltIdx)); 
    trnClass{i} = class{i}(:,:,rndIdx(spltIdx + 1 : end));
end

%% 1.c histogram
s = size(class{16},3);
b = zeros(0,0);
for i = 1 : size(class{16},3)
    b = [b; diag(abs(class{16}(:,:,i)))];
end
