function W=createKnnGraph(N1,varargin)

    n1=size(N1, 1);
     K=size(N1, 2);
     
    W=sparse(n1,n1); 
    if nargin>2
        D=varargin{1};
        sigma=varargin{2};
        for i=1:n1; 
        for j=1:K;
            W(i,N1(i,j))=exp(D(i,j)^2/(-2*sigma^2));
        end
    end
    W=0.5*(W+W');
    
    else
    for i=1:n1; 
        for j=1:K;
            W(i,N1(i,j))=1;
        end
    end
    W=0.5*(W+W');
    end
end