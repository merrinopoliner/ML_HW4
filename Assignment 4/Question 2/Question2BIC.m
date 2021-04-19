 close all,

filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';

for imageCounter = 1:size(filenames,2)
    imdata = imread(filenames{1,imageCounter}); 
    figure();
    if length(size(imdata))==3 % color image with RGB color values
        [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
        rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
        features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
        for d = 1:D
            imdatad = imdata(:,:,d); % pick one color at a time
            features = [features;imdatad(:)'];
        end
        minf = min(features,[],2); maxf = max(features,[],2);
        ranges = maxf-minf;
        x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
    end
    [d, N] = size(x); % feature dimensionality
    
    nSamples = d*N;
    maxM = 20;
    for M = 1:maxM
        M,
        nParams(1,M) = (M-1) + d*M + M*(d+nchoosek(d,2));
        % (M-1) is the degrees of freedomg for alpha parameters
        % d*M is the derees of freedomg for mean vectors of M Gaussians
        % M*(d+nchoosek(d,2)) is the degrees of freedom in cov matrices
        % For cov matrices, due to symmetry, only count diagonal and half of
        % off-diagonal entries.
        options = statset('MaxIter',1000); % Specify max allowed number of iterations for EM
        % Run EM 'Replicates' many times and pickt the best solution
        % This is a brute force attempt to catch the globak maximum of
        % log-likelihood function during EM based optimization
        gm{M} = fitgmdist(x',M,'Replicates',10,'start', 'plus', 'Options',options); 
        neg2logLikelihood(1,M) = -2*sum(log(pdf(gm{M},x')));
        BIC(1,M) = neg2logLikelihood(1,M) + nParams(1,M)*log(nSamples);
        figure(1), plot([1:M],BIC(1:M),'.'), 
        xlabel('Number of Gaussian Components in GMM'),
        ylabel('BIC'),
        drawnow,
    end
    [~,bestM] = min(BIC),
    bestGMM = gm{bestM},
    
    
    %Evaluate which component each feature vector belongs to
    
    prob = ones(bestM, N);
    for m = 1 : bestM
        prob(m, :) = bestGMM.ComponentProportion(m) * evalGaussian(x, bestGMM.mu(m,:)', bestGMM.Sigma(:,:,m));
    end
    [~,labels] = max(prob, [], 1);
    labelImage = reshape(labels, R, C);
    figure, imshow(uint8(labelImage*255/bestM));
        title(strcat({'Clustering with K = '},num2str(bestM)));
    
end

%%%
function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),

end

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end