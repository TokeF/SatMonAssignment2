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

figure(2);
imagesc(RGB)
title('RGB image')

figure(3)
imagesc(M.gtruth)
colormap(M.cmap)
title('Ground truth')

%% 1.b split in training and test
[class, tstClass, trnClass] = splitData(M, 0.6, 2);

%% 1.c histogram
cn = 5;
s = size(class{cn},3); %number of samples in a class
b = zeros(s*3,1); %times 3 because we have 3 values in diagonal
for i = 0 : s - 1
    idx = i*3+1; %to go trough 3 indexes in b
    b(idx:idx+2) = diag(abs(class{cn}(:,:,i+1)));
end
histogram(b, 'Normalization', 'probability');
alpha = 27;
mu = mean(b);
p = @(I) (alpha / mu)^alpha .* I.^(alpha - 1) / factorial(alpha - 1)...
    .* exp(-alpha / mu .* I);
Imodel = linspace(0,max(b),100);
pdist = p(Imodel);
hold on
plot(Imodel,pdist./sum(pdist))

%% 1.d 