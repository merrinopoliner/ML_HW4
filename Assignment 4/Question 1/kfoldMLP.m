function bestM = kfoldMLP(data, labels, K)
PERCEPTRONS = [1:30];

% Maximum likelihood training of a 2-layer MLP
% assuming additive (white) Gaussian noise

N = length(labels);


% Input N specifies number of training samples

% Divide the data set into K approximately-equal-sized partitions
dummy = ceil(linspace(0,N,K+1));
indPartitionLimits = zeros(K, 2);
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end


% Try several numbers-of-perceptrons
rng('default');
index = 1;
MSEvalidate = zeros(K,length(PERCEPTRONS));
AverageMSEvalidate = zeros(1,length(PERCEPTRONS));
for M = PERCEPTRONS
    disp("          " + M + " perceptrons...");
    % K-fold cross validation
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = data(:, indValidate); % Using folk k as validation set
        yValidate = labels(:, indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k+1,1):N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k-1,2)];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
        end
        xTrain = data(:, indTrain); % using all other folds as training set
        yTrain = labels(:, indTrain);
        Nvalidate = length(indValidate);
        % Train model parameters
        % Initialize model
        net = feedforwardnet(M);
        net.layers{1}.transferFcn = 'poslin';
        net.layers{2}.transferFcn = 'purelin';
        
        
        % Optimize model
        net = train(net, xTrain, yTrain);
        
        %run trained network on validation set
        hValidate = net(xValidate);

        
        MSEvalidate(k,M) = sum(sum((yValidate-hValidate).*(yValidate-hValidate),1),2)/N;
    end
    AverageMSEvalidate(index) = mean(MSEvalidate(:,M));
    index = index + 1;
end

[bestErr,bestMIndex] = min(AverageMSEvalidate);
bestM = PERCEPTRONS(bestMIndex);
disp("          Selected #: " + bestM + ", MSE = " + bestErr);
rng('shuffle');
end




