function [predYen,result,varargout]=prior_MA(X1,X2,truthmap,truthmap2,par,graph,varargin)
% prior_MA for bi-temporal hyperspectral image
% input parameter
% X1 and X2 are Nxdim matrix;X1 is source image and X2 is target image
% truthmap and truthmap2 is label of X1 and X2 respectivily
% par : k is neighbor number of knnsearch 
%       k_ali is neighbor number of knnclassify
%       n is neighbor number of manifold algorithom
%       band_num is the reduced dim of manifold algorithom
%       spa is the  spatial gauss parameter  
%       delta is apectral gauss parameter
%       weightmethod :spa is spatial and apectral weight method  else only consider spectral weight
%       when weightmethod is spa, par.a is the spatial weight 
% graph :'01' Laplacian graph is 01 graph
%        "all"  
%varargin: if  weightmethod is spa then varargin{1} and varargin{2} is
%location of X1 and X2 
% output
% predYen is label by using Ms (aligned X1) to classify Ft (aligned X2)
% result: result of classification concluding OA /KAPPA and confusion matrix
% %[Author Notes]% 
%Author:        Meiling Zhang
%Email:         921783424@qq.com
%Affiliation:   Harbin Instratute of Techonolgy (HIT)
%date:          2016-11-06
% %[Reference]%:
%[1] C. Wang and S. Mahadevan, “Manifold Alignment without Correspondence” in Proc. IJCAI., pp.1273-1278, Jul. 2009.
%[2] H. L. Yang and M. Crawford. “Spectral and spatial proximity-based manifold alignment for multitemporal hyperspectral image classification,”
% IEEE Trans. Geosci. Remote Sens., vol. 54, no. 1, pp. 51-64, Jan. 2016.
%[3] drtoolbox
N1=size(X1,1);
M=size(X2,1);
band_num=par.band_num;
k=par.k;
k_maj=par.k_maj;
delta=par.delta;
n=par.n;
% get the prior manifold of source image X1
[Ms,mapping] = compute_mapping(X1,'Laplacian',band_num,n);
X1=X1(mapping.conn_comp,:);
truthmap=truthmap(mapping.conn_comp,:);
% alignment
if isfield(par,'weightmethod')&&(strcmp(par.weightmethod,'spa'))
    
    loc1=varargin{1};
    loc2=varargin{2};
    a=par.a;
    spa=par.spa_weight;
    fprintf('空谱权重，空间比重为%d\n',a);
    switch graph
        case '01'
            [idx1,D1]=knnsearchII(X1, [], k);
            [idx2,D2]=knnsearchII(X2, [], k);
            W1=createKnnGraph(idx1);
            W2=createKnnGraph(idx2);
        case 'all'
            W1=createAllConnectedGraph(X1', delta);
            W2=createAllConnectedGraph(X2',delta);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %加权空间坐标
    dist1=L2_distance(loc1',loc1',0);
    dist1=exp(-dist1.^2./(2*spa*spa));
    W1=W1.*dist1+W1;
    dist2=L2_distance(loc2',loc2',0);
    dist2=exp(-dist2.^2./(2*spa*spa));
    W2=W2.*dist2+W2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 建立W12
    W21=zeros(M,N1);
    D=L2_distance(X2',X1');
    dist=L2_distance(loc2',loc1');
    dist=exp(-dist.^2./(2*spa*spa));
    D=exp(-D.^2./(2*delta*delta));
    D=a.*dist+D;
    [D,idx_k]=sort(D,2,'descend');
    D=D(:,1:k_maj);
    idx_k=idx_k(:,1:k_maj);
    tr_idx=truthmap(idx_k);
    for i=1:size(tr_idx,1)
        
        tj=tabulate(tr_idx(i,:));
        [max_v,dqwz]=max(tj(:,2));
        max_n=length(find(tj(:,2)==max_v));%最大的个数出现了几次
        if max_n==1
            j_loc=find(tr_idx(i,:)==tj(dqwz,1));
            %             disp('only one maxminum');
            for j=1:length(j_loc)
                jloc=j_loc(j);
                W21(i,idx_k(i,jloc))=D(i,jloc);
                %  W21(i,idx_k(i,jloc))=1;
            end
        end
    end
else
    fprintf('only spectral weight!');
    switch graph
        case '01'
            fprintf('------------01 graph------------------');
            [idx1,D1]=knnsearchII(X1, [], k);
            [idx2,D2]=knnsearchII(X2, [], k);
            W1=createKnnGraph(idx1);
            W2=createKnnGraph(idx2);
        case 'all'
            W1=createAllConnectedGraph(X1', delta);
            W2=createAllConnectedGraph(X2',delta);
    end
    % 建立W12
    W21=zeros(M,N1);
    D=L2_distance(X2',X1');
    D=exp(-D.^2./(2*delta*delta));
    [D,idx_k]=sort(D,2,'descend');
    D=D(:,1:k_maj);
    idx_k=idx_k(:,1:k_maj);
    tr_idx=truthmap(idx_k);
    for i=1:size(tr_idx,1)
        
        tj=tabulate(tr_idx(i,:));
        [max_v,dqwz]=max(tj(:,2));
        max_n=length(find(tj(:,2)==max_v));%最大的个数出现了几次
        if max_n==1
            j_loc=find(tr_idx(i,:)==tj(dqwz,1));
            for j=1:length(j_loc)
                jloc=j_loc(j);
                W21(i,idx_k(i,jloc))=D(i,jloc);
                %  W21(i,idx_k(i,jloc))=1;
            end
        end
    end
end
% laplacian graph
D = sqrt(sum(W2))';
% D(D==0) = 1;
D = spdiags(1./D,0,M,M);
% Compute Laplacian
Lt = speye(M) - D*W2*D;
Lt(isnan(Lt)) = 0;
Lt(isinf(Lt)) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lt=full(Lt);
Fu=pinv(Lt)*(W21)*Ms;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(par,'k_ali')
    par.k_ali=1;
end
predYen = knnclassify(Fu,Ms,truthmap,par.k_ali);
result=assessment(truthmap2,predYen);
fprintf('对准后的分类精度： %.3f',result.OA);
if nargout>2
    varargout{1}=Ms;
    varargout{2}=Fu;
end