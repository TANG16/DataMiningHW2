function IDX =spectral(data,k, sigma)
% Ng, A., Jordan, M., and Weiss, Y. (2000). On spectral clustering: analysis and an algorithm. In T. Dietterich,
% S. Becker, and Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14 
% (pp. 849 – 856). MIT Press.

% Asad Ali
% GIK Institute of Engineering Sciences & Technology, Pakistan
% Email: asad_82@yahoo.com

% CONCEPT: Introduced the normalization process of affinity matrix(D-1/2 A D-1/2), 
% eigenvectors orthonormal conversion and clustering by kmeans 

% data: data matrix (nxd)
% k: number of clusters
% sigma: width of RBF kernel

% data normalization
Y=data;
meanY = mean(Y);
stdY  = std(Y);
Y = Y - repmat(meanY,size(Y,1),1);
Y = Y ./ repmat(stdY,size(Y,1),1);
data=Y; clear Y;

% calculate the affinity / similarity matrix (RBF kernel is assumed)
DIST = squareform(pdist(data,'euclidean'));
Affinity=exp(-DIST/(sigma^2));

% compute the degree matrix
for i=1:size(Affinity,1)
    D(i) = sum(Affinity(i,:));
end

% compute the normalized laplacian / affinity matrix (method 1)
for i=1:size(Affinity,1)
    for j=1:size(Affinity,2)
        NL1(i,j) = Affinity(i,j) / (sqrt(D(i)) * sqrt(D(j)));  
    end
end

% perform the eigen value decomposition
[eigVectors,eigValues] = eig(NL1);
%[eigVectors,eigValues]=eigs(NL1,k);

% select k top eigen vectors
nEigVec = eigVectors(:,(size(eigVectors,1)-(k-1)): size(eigVectors,1));

%nEigVec=eigVectors;

% construct the normalized matrix U from the obtained eigen vectors
for i=1:size(nEigVec,1)
    n = sqrt(sum(nEigVec(i,:).^2));    
    U(i,:) = nEigVec(i,:)./ n; 
end

% perform kmeans clustering on the matrix U
IDX = kmeans(U,k,'replicates',20); 


plot_max10_clusters(data,IDX);

