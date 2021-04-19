function [trainData, validateData] = generateQ1Data()

%Generate training data
[trainData.x, trainData.y, validateData.x, validateData.y] = hw2q1(1000, 10000);


end