 clear all
load('qingdaohyper.mat')
 dims=60;
 n=5;
 X1=double(qingdao11_03)';

 X2=double(qingdao11_05)';

  X1=[X1(1:49,:);X1(51:93,:);X1(101:145,:);X1(148:149,:);X1(153:197,:)];
  X2=[X2(1:49,:);X2(51:93,:);X2(101:145,:);X2(148:149,:);X2(153:197,:)];
  X1=Normalization(X1',1);
X2=Normalization(X2',1); 
X1=X1(2:end,:);

truthmap=qingdao14_truthmap03;
truthmap2=qingdao14_truthmap05;
truthmap=truthmap(2:end);
loc03=loc03(2:end,:);
 N=size(X1,1);
M=size(X2,1);
 [Ms,mapping] = compute_mapping(X1,'Laplacian',dims,n);
X1=X1(mapping.conn_comp,:);
truthmap=truthmap(mapping.conn_comp,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%找权重
sigma=2;
k=15;
[idx1,D1]=knnsearchII(X1, [], k);
[idx2,D2]=knnsearchII(X2, [], k);
W1=createKnnGraph(idx1,D1,sigma);
W2=createKnnGraph(idx2,D2,sigma);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 监督的拉普拉斯图
% [w_i,w_j,~]=find(W1~=0);
% [wi_idx,wj_idx]=find(truthmap(w_i)~=truthmap(w_j));
% W1(w_i(wi_idx),w_j(wj_idx))=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 加上空间位置的权重

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% method='knn';
% [idx_k,D]=getW21_idx(X1',X2',method,k);
% WA12=zeros(N, M, k);
% for i=1:k
%     idx=idx_k(:,i);
%     for j=1:size(idx,1)
%         WA12(idx(j),j,i)=exp(D(j,i)^2/-2*sigma^2);
%     end
% end
% W21=reshape(WA12(:,:,1),N*M,1);
% WA12=reshape(WA12,N*M,k);fc=std(WA12');wz=find(fc);
% for i=1:length(wz)
%     tj=tabulate(WA12(wz(i),:));
%     [~,dqwz]=max(tj(:,2));
%     if tj(dqwz,1)
%         W21(wz(i))=tj(dqwz,1);
%     else
%         tj(dqwz,2)=0;[~,dqwz]=max(tj(:,2));W21(wz(i))=tj(dqwz,1);
%     end
% end
% W21=reshape(W21,N,M);
W21=zeros(M,N);
method='knn';
[idx_k,D]=getW12_idx(X2',X1',method,3);
tr_idx=truthmap(idx_k);
for i=1:size(tr_idx,1)
   
tj=tabulate(tr_idx(i,:));
[max_v,dqwz]=max(tj(:,2));
max_n=length(find(tj(:,2)==max_v));%最大的个数出现了几次
if max_n==1
    j_loc=find(tr_idx(i,:)==tj(dqwz,1));
    j_loc=j_loc(1);
    W21(i,idx_k(i,j_loc))=exp(D(i,j_loc)^2/-2*sigma^2);
end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%无监督的方法求W21
% W21=generateWeight_gauss(X1', X2', 5,1); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%加上空间权重
a=0
sigma_spa=25;
W2_spa=L2_distance(loc05',loc05',0);
W2_spa=exp(-W2_spa.^2./(2*sigma_spa*sigma_spa));
W2_all=a.*(W2.*W2_spa)+W2;
W21_spa=L2_distance(loc05',loc03',0);
W21_spa=exp(-W21_spa.^2./(2*sigma_spa*sigma_spa));
W21_all=W21+a.*W21.*W21_spa;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%求拉普拉斯图
D = sqrt(sum(W2_all))';
% D(D==0) = 1;
D = spdiags(1./D,0,M,M);
% Compute Laplacian
Lt = speye(M) - D*W2_all*D;
Lt(isnan(Lt)) = 0; 
Lt(isinf(Lt)) = 0; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lt=full(Lt);
Fu=pinv(Lt)*(W21_all)*Ms;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qingdaoClass = knnclassify(Fu,Ms,truthmap,1);
 ResLatent=assessment(truthmap2,qingdaoClass);
 fprintf('对准后的分类精度： %.3f', ResLatent.OA);
 %% 直接原始图像分类
 qingdaoClass = knnclassify(X2,X1,truthmap,1);
 ResLatentS=assessment(truthmap2,qingdaoClass);
  fprintf('对准前的分类精度： %.3f', ResLatentS.OA);
  %% 投影到流形空间以后自己分类自己
[ idx_tr, idx_te, label_tr, label_te, numclass ] =randtrain( truthmap, 0.5 );
rX=Ms(idx_tr,:);
rY=Ms(idx_te,:);
qingdaoClass = knnclassify(rY,rX,label_tr, 1);
result00=assessment(label_te,qingdaoClass);clear qingdaoClass 
fprintf('投影到流形空间以后自己分类自己： %.3f', result00.OA);