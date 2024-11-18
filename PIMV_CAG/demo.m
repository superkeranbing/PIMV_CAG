clear;
addpath('./fun');
addpath('D:/multiview-dataset');
addpath('./MIndex');

Dataname='MSRCv1';
load(Dataname);

numClust=length(unique(Y));
numSample=length(Y);
numView=length(X);

del= 0.5;
max_iter = 100;

Datafold=[Dataname,'_del_',num2str(del),'.mat'];
load(Datafold);

lambda=10^(0);alpha=10^(-1);dim=numClust*3;k=3;
m=dim;
for f=1:10 
    fold = folds{f};
    linshi_GG = 0;
    linshi_LS = 0;
    for iv = 1:length(X) 
        X1{iv}= NormalizeFea(X{iv}',0); 
        ind_1 = find(fold(:,iv) == 1);
        ind_0 = find(fold(:,iv) == 0);
        X1{iv}(:,ind_0) = []; 
        n_v(iv)=length(ind_0);
        W{iv}=zeros(n_v(iv),numSample);
        for i = 1:length(ind_0)
            j = ind_0(i);
            W{iv}(i, j) = 1;
        end
        linshi_W = diag(fold(:,iv));
        linshi_W(:,ind_0) = []; 
        G{iv} = linshi_W; 
        
        X1{iv} = X1{iv}*G{iv}'; 
        linshi_St = X1{iv}*X1{iv}'+lambda*eye(size(X1{iv},1));
        St2{iv} = mpower(linshi_St,-0.5); 
    end
   
    [Z,obj] = PPAU(X1,W,St2,n_v,dim,m,k,lambda,alpha,max_iter,numClust);
    pre_labels=kmeans(real(Z'),numClust,'emptyaction','singleton','replicates',10,'display','off');
    res(f,:)=Clustering8Measure(Y, pre_labels)*100;
end

Metrics = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'}; 
numMetrics = length(Metrics);
meanMetrics = zeros(1, numMetrics);
stdMetrics = zeros(1, numMetrics);

for i = 1:numMetrics
    meanMetrics(i) = mean(res(:,i));
    stdMetrics(i) = std(res(:,i));
    result{i}=strcat(num2str(meanMetrics(i), '%.2f'),'±',num2str(stdMetrics(i), '%.2f'));
end
for i=1:3
    fprintf(['---',Metrics{i},'=',result{i},'---']);
end
fprintf('\n');
