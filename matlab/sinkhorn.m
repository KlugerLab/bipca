function [A_ds,x,y] = sinkhorn(A,tol,maxItr)
[m,n] = size(A);
x0 = (n)./sum(A,2); y0 = m./(A.'*x0); x1 = (n)./(A*y0); y1 = m./(A.'*x1);
itr = 1;
while ((max(abs(x0./x1 - 1))>tol)||(max(abs(y0./y1 - 1))>tol))&&(itr<maxItr)
    x0 = x1;  y0 = y1;
    x1 = (n)./(A*y0);
    y1 = m./(A.'*x1);
    itr = itr + 1;
end
if (itr>=maxItr)
    disp(['Sinkhorn-Knopp: Maximum number of iterations reached. Error: ', num2str(max(max(abs(x0./x1 - 1)),(max(abs(y0./y1 - 1))>tol)))]);
end
x = x1;                            
y = y1;
A_ds = bsxfun(@times,x,bsxfun(@times,A,y.'));   % diag(x)*A*diag(y)
end
