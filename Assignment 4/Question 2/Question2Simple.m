clear all, close all,

filenames{1,1} = '3096_color.jpg';
filenames{1,2} = '42049_color.jpg';

K = 2; % desired numbers of clusters
delta = 1e-2; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates


for imageCounter = 1:size(filenames,2)
    imdata = imread(filenames{1,imageCounter}); 
    figure(1), subplot(size(filenames,2),length(K)+1,(imageCounter-1)*(length(K)+1)+1), imshow(imdata);
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
    d = size(x,1); % feature dimensionality
    
    M = 2;
    % Initialize the GMM to randomly selected samples
    alpha = ones(1,M)/M;
    shuffledIndices = randperm(N);
    mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
    for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
        %alpha(1,m) = find(assignedCentroidLabels==m)/N;
        Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
    end
    t = 0; %displayProgress(t,x,alpha,mu,Sigma);

    Converged = 0; % Not converged at the beginning
    while ~Converged
        for l = 1:M
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        muNew = x*w';
        for l = 1:M
            v = x-repmat(muNew(:,l),1,N);
            u = repmat(w(l,:),d,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        Dmu = sum(sum(abs(muNew-mu)));
        DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
        t = t+1; 
        displayProgress(t,x,alpha,mu,Sigma);
    end
    
    
    %Evaluate which component each feature vector belongs to
    
    prob = ones(M, size(features, 2));
    for m = 1 : M
        prob(m, :) = alpha(m) * evalGaussian(x, mu(:,m), Sigma(:,:,m));
    end
    [~,labels] = min(prob, [], 1);
    labelImage = reshape(labels, R, C);
    figure, imshow(uint8(labelImage*255/K));
        title(strcat({'Clustering with K = '},num2str(K)));
    
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