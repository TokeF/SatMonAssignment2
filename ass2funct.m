function ass2funct()

end
function M = loadData()
M = load('flevoland.mat');
end
function plot1a(M)
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
end

function [trnData, tstData] = testTrainSplit(M)
%% 1.b split in training and test
rng(100);
r = size(M.C,3);
c = size(M.C,4);
rr = randperm(r);
rc = randperm(c);
percentileRow = round(0.6*length(rr));
percentileCol = round(0.6*length(rc));
trnData = M.C(:,:,rr(1:percentileRow), rc(1:percentileCol));
tstData = M.C(:,:,rr(percentileRow + 1 : end), rc(percentileCol + 1 : end));
end


