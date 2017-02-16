function [ idx_tr, idx_te, label_tr, label_te, numclass ] =...
    randtrain( truthmap, nptrain )

numclass = max(truthmap(:));
truthmap = truthmap(:);
idx_tr = [];
idx_te = [];
label_te = [];
if nptrain < 1
    if nptrain >= 1
        error('Percent of number train should < 1');
    end
    for j = 1:numclass
        nec(j) = sum(truthmap == j);
    end
    numtrain = ceil(nec*nptrain);
    label_tr =[];
    for i = 1:numclass
        label_tr = [ label_tr; repmat(i,numtrain(i),1) ];
    end
    for i = 1:numclass
        idx_t = find(truthmap==i);
        idx_r = randperm(length(idx_t));
        idx_tr = [ idx_tr; idx_t(idx_r(1:numtrain(i))) ];
        idx_t(1:numtrain(i)) = [];
        idx_te = [ idx_te; idx_t ];
        label_te = [ label_te; double(i)*ones(length(idx_t),1) ];
    end
else
    numtrain = nptrain;
    label_tr = repmat(1:numclass,nptrain,1);
    label_tr = label_tr(:);
    for i = 1:numclass
        idx_t = find(truthmap==i);
        idx_r = randperm(length(idx_t));
        idx_tr = [ idx_tr; idx_t(idx_r(1:numtrain)) ];
        idx_t(1:numtrain) = [];
        idx_te = [ idx_te; idx_t ];
        label_te = [ label_te; double(i)*ones(length(idx_t),1) ];
    end
end


idx_tr = uint32(idx_tr);
idx_te = uint32(idx_te);
label_tr = double(label_tr);
label_te = double(label_te);
numclass = double(numclass);