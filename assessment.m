function RESULTS = assessment(Labels,PreLabels)

Etiquetas = union(Labels,PreLabels); % Class labels
NumClases = length(Etiquetas); % Number of classes
% Compute confusion matrix
for i=1:NumClases
    for j=1:NumClases
        ConfusionMatrix(i,j) = length(find(PreLabels==Etiquetas(i) & Labels==Etiquetas(j)));
    end;
end;

% Compute overall accuracy
n      = sum(ConfusionMatrix(:));
PA     = sum(diag(ConfusionMatrix));
OA     = PA/n;

% Kappa statistics
npj = sum(ConfusionMatrix,1);
nip = sum(ConfusionMatrix,2);
PE  = npj*nip;
Kappa  = (n*PA- PE)/(n^2-PE);

% Pro.Acc and User.Acc
for i=1:NumClases
 
ProAcc(i)=100*ConfusionMatrix(i,i)/sum(ConfusionMatrix(:,i));
UserAcc(i)=100*ConfusionMatrix(i,i)/sum(ConfusionMatrix(i,:));

end
ConfusionMatrix=[ConfusionMatrix;ProAcc];
UserAcc=[UserAcc,100*OA]';
ConfusionMatrix=[ConfusionMatrix,UserAcc];
% Outputs
RESULTS.ConfusionMatrix = ConfusionMatrix;
RESULTS.Kappa           = Kappa;
RESULTS.OA              = 100*OA;
RESULTS.ProAcc          =ProAcc;
RESULTS.UserAcc         =UserAcc;



