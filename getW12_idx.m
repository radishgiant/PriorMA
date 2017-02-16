function [idx,varargout]=getW12_idx(GF1_13_spe_norm,GF1_14_spe_norm,method,k,varargin);

if ~exist('method','var')
    method='min_ecud'
    if ~exist('p','var')
        p=0.5*size(GF1_13_spe_norm,2);
    end 
elseif strcmp(method,'clust_dist')
    if ~exist('Gauss_sigma','var')
        Gauss_sigma=1;
    end
    if ~exist('k','var')
        k=6;
    end
end
switch method
    case 'min_ecud'
        dist=L2_distance(GF1_13_spe_norm,GF1_14_spe_norm,0);
        idx=[];
        for idx_i=1:p
            min_d=min(min(dist));
            [min_r,min_c,~]=find(dist==min_d);
            idx=[idx;min_r,min_c];
            dist(min_r,min_c)=nan;
        end
    case 'knn'
         [idx,D]=knnsearch(GF1_14_spe_norm',GF1_13_spe_norm','K',k);
         if nargout>1
             varargout{1}=D;
         end
    case 'clust_dist'
        K=knGauss(Rt',Rt',Gauss_sigma);
        Rt_model.state = kkmeans_train(K, parameters);
        Rt_model.X=Rt';
        Rt_model.label=Rt_model.state.clustering;
        for Rt_class_i=1:max(Rt_model.label)
            Rt_class_idx=find(Rt_model.label==Rt_class_i);
            Rt_class(Rt_class_i)=[{Rt_model.X(:,Rt_class_idx)}];
        end
        
end