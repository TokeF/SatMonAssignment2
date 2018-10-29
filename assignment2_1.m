%% Load data
M = load('flevoland.mat');

%% part 1a
% Use HH as red, XX as green and VV as blue
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

%% 1.b split datal in training and test
[class, tstClass, trnClass] = splitData(M, 0.6, 2);

%% 1.c histogram
cn = 5;
s = size(trnClass{cn},3); %number of samples in a class
Ihh = zeros(s,1); %times 3 because we have 3 values in diagonal
Ihv = zeros(s,1);
Ivv = zeros(s,1);
for i = 1 : s
    %idx = i*3+1; %to go trough 3 indexes in b
    Ihh(i) = abs(trnClass{cn}(1,1,i));
    Ihv(i) = abs(trnClass{cn}(2,2,i));
    Ivv(i) = abs(trnClass{cn}(3,3,i));
end

alpha = 27;
p = @(I, mu) (alpha / mu)^alpha .* I.^(alpha - 1) / factorial(alpha - 1)...
    .* exp(-alpha / mu .* I);

subplot(1,3,1); 
histogram(Ihh, 'Normalization', 'pdf');
hold on
Imodel = linspace(0,max(Ihh),length(Ihh));
plot(Imodel,p(Imodel,mean(Ihh)))
title('Ihh')

subplot(1,3,2);
histogram(Ihv, 'Normalization', 'pdf');
hold on
Imodel = linspace(0,max(Ihv),length(Ihv));
plot(Imodel,p(Imodel,mean(Ihv)))
title('Ihv')

subplot(1,3,3); 
histogram(Ivv, 'Normalization', 'pdf');
hold on
Imodel = linspace(0,max(Ivv),length(Ivv));
plot(Imodel,p(Imodel,mean(Ivv)))
title('Ivv')


%% 1.d 
% The gamma function is used as decision function. The prior P(w_j) = 1.
% The evidence p(x) is not nescessary.
classNum = 15;
clsFication = cell(1,classNum); % store classification result
userAcc = zeros(1,classNum);
conf = zeros(classNum,classNum); % confusion matrix
%iterate trough each class
for i = 1 : classNum
    % to keep result of gamma function
    b = zeros(size(tstClass{i},3), classNum);
    %calc gamme function for each class for each gamma function
    for j = 1 : classNum
        b(:,j) = squeeze(p(tstClass{i}(1,1,:), mean(tstClass{j}(1,1,:))));
    end
    %assign labels to a pixel according to highest valued gamma function
    [~, label] = max(b,[],2);
    clsFication{i} = label;
    %store the user accuracy
    userAcc(i) = sum(label == i) / length(label);
    
    %calcualte how many of each class a test set predicted
    for k = 1 : classNum
        conf(i,k) = sum(label == k);
    end
end
edges = 0.5:1:15.5;
%histogram(label,edges,'normalization', 'probability')

%% 1.e
%user accuracy:
userAcc;
prodAcc = diag(conf)' ./ sum(conf);




