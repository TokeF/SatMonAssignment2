%% Load data
M = load('flevoland.mat');

%% part 1a
% Use HH as red, XX as green and VV as blue
R = M.C(1,1,:,:); G = M.C(2,2,:,:); B = M.C(3,3,:,:);
R = squeeze(R); G = squeeze(G); B = squeeze(B);
%convert to dB
R = 10*log10(4*pi*R);
G = 10*log10(4*pi*G);
B = 10*log10(4*pi*B);
%stretch values 
RGB(:,:,1) = mat2gray(R,[min(min(R)) max(max(R))]);
RGB(:,:,2) = mat2gray(G,[min(min(G)) max(max(G))]);
RGB(:,:,3) = mat2gray(B,[min(min(B)) max(max(B))]);
%plot each channel to check they are correct
figure(1);
subplot(1,3,1); imshow(R,[]); title('Red')
subplot(1,3,2); imshow(G,[]); title('Green')
subplot(1,3,3); imshow(B,[]); title('Blue')

%rgb image
figure(2);
imagesc(RGB)
title('RGB image')

%gtruth
figure(3)
imagesc(M.gtruth)
colormap(M.cmap)
title('Ground truth')

%% 1.b split datal in training and test
[class, tstClass, trnClass] = splitData(M, 0.6, 2);

%% 1.c histogram
%define the gamma distribution
alpha = 27;
p = @(I, mu) (alpha / mu)^alpha .* I.^(alpha - 1) / factorial(alpha - 1)...
    .* exp(-alpha / mu .* I);

%plot histogram for HH, HV and VV cahnnel for the three classes
for c = 1 : 3
    cn = [3 5 15];
    s = size(trnClass{cn(c)},3); %number of samples in a class
    Ihh = zeros(s,1);
    Ihv = zeros(s,1);
    Ivv = zeros(s,1);
    for i = 1 : s
        Ihh(i) = abs(trnClass{cn(c)}(1,1,i));
        Ihv(i) = abs(trnClass{cn(c)}(2,2,i));
        Ivv(i) = abs(trnClass{cn(c)}(3,3,i));
    end
    
    figure
    subplot(1,3,1);
    histogram(Ihh, 'Normalization', 'pdf', 'linewidth', 0.01, 'edgealpha', 0.4);
    hold on
    Imodel = linspace(0,max(Ihh),length(Ihh));
    plot(Imodel,p(Imodel,mean(Ihh)), 'linewidth', 1.3)
    title('Ihh')
    
    subplot(1,3,2);
    histogram(Ihv, 'Normalization', 'pdf', 'linewidth', 0.01, 'edgealpha', 0.4);
    hold on
    Imodel = linspace(0,max(Ihv),length(Ihv));
    plot(Imodel,p(Imodel,mean(Ihv)), 'linewidth', 1.3)
    title('Ihv')
    
    subplot(1,3,3);
    histogram(Ivv, 'Normalization', 'pdf', 'linewidth', 0.01, 'edgealpha', 0.4);
    hold on
    Imodel = linspace(0,max(Ivv),length(Ivv));
    plot(Imodel,p(Imodel,mean(Ivv)), 'linewidth', 1.3)
    title('Ivv')
end


%% 1.d
% The gamma function is used as decision function. The prior P(w_j) = 1.
% The evidence p(x) is not nescessary.
classNum = 15;
clsFication = cell(1,classNum); % store classification result
userAcc = zeros(1,classNum);
conf = zeros(classNum,classNum); % confusion matrix
%iterate trough each class
chan = 1; % cahnnel: 1 for HH, 2 for HV, 3 for VV
for i = 1 : classNum
    % to keep result of gamma function
    b = zeros(size(tstClass{i},3), classNum);
    %calc gamme function for each class for each gamma function
    for j = 1 : classNum
        b(:,j) = squeeze(p(tstClass{i}(chan,chan,:), mean(tstClass{j}(chan,chan,:))));
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
histogram(label,edges,'normalization', 'probability')

%% 1.e
%Accuracy:
totSamp = sum(sum(conf,2));
userAcc;
prodAcc = diag(conf)' ./ sum(conf);
totAcc = sum(diag(conf)) / totSamp;
pe = sum(conf)/ totSamp * (sum(conf,2) / totSamp);
p0 = totAcc;
kappa = (p0 - pe) / (1 - pe);



