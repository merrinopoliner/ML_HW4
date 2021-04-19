function Question1Main

%generate training and validation data
[training, validate] = generateQ1Data();


%now perform 10-fold cross validation for number of perceptron selection

disp("Performing 10-fold cross validation...");

numPerceptrons = kfoldMLP(training.x, training.y, 10);
    
%train the network using the training sets. Train multiple networks and
%choose the best one to avoid stopping at a local minimum
disp("Training networks...");

trainedNet.MSE = Inf;
disp("          min(Perror):");
for j = 1 : 10

    net = feedforwardnet(numPerceptrons);
    net.layers{1}.transferFcn = 'poslin';
    net.layers{2}.transferFcn = 'purelin';
    
    net = train(net, training.x, training.y);
    %evaluate the network
    hValidate = net(validate.x);
    MSEvalidate = sum(sum((validate.y-hValidate).*(validate.y-hValidate),1),2)/length(validate.y);

    if MSEvalidate < trainedNet.MSE
        disp("          " + MSEvalidate);
        trainedNet.network = net;
        trainedNet.MSE = MSEvalidate;
    end

end
disp("          Trained network with MSE = " + trainedNet.MSE);

end