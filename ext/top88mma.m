%%%% AN 88 LINE TOPOLOGY OPTIMIZATION CODE Nov, 2010 %%%% input: top88(60,20,0.5,3,1.5,1)
clear all; clc;
nelx = 120; nely = 40; volfrac = 0.3; penal = 3; rmin = 1.5; ft = 2;
%% MATERIAL PROPERTIES
E0 = 1;
Emin = 1e-9;
nu = 0.3;
nele = nelx*nely;
%% USER-DEFINED LOAD DOFs
[i_load, j_load] = meshgrid(nelx+1,floor(nely/2));                      % Coordinates (1-based indexing)
loadnid = (i_load-1)*(nely+1) + j_load;                % Node IDs
loaddofs = [2*loadnid(:)']; % DOFs
%% USER-DEFINED SUPPORT FIXED DOFs
[i_fix,j_fix] = meshgrid(1,1:nely+1);                  % Coordinates (1-based indexing)
fixednid = (i_fix-1)*(nely+1) + j_fix;                 % Node IDs
fixeddofs = union(2*fixednid(:)'-1, 2*fixednid(:)');                        % DOFs
%% PREPARE FINITE ELEMENT ANALYSIS
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]); % 8 x 8 matrix stiffness matrix for one 4-node quad-element under plane stress
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx); % nodenrs(i,j) labels each mesh node.
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1); % column vector of global node index of first degree of freedom of lower left node of each element
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1); % gives, for each element e, the 8 global DOF indices [ux,uy]×4 nodes
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1); % iK are the row indices for building the global sparse stiffness matrix K in one go
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1); % jK are the column indices for building the global sparse stiffness matrix K in one go
% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
ndof = 2*(nely+1)*(nelx+1);
F = sparse(loaddofs,... % the dof number of where the load is applied. E.g. 2 means vertical dof of node 1
           1,...        % the column no. of F where the loads go into
           -1,...       % the value of the load to place in each row with loading
           ndof,...     % the total number of rows in F
           1);          % total no. of columns in F
U = zeros(ndof,1);
freedofs = setdiff(1:ndof,fixeddofs);
%% PREPARE FILTER
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
  for j1 = 1:nely
    e1 = (i1-1)*nely+j1;
    for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
      for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
        e2 = (i2-1)*nely+j2;
        k = k+1;
        iH(k) = e1;
        jH(k) = e2;
        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
      end
    end
  end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);
%% INITIALIZE ITERATION
x = repmat(volfrac,nely,nelx);
xPhys = x;
loop = 0;
change = 1;
% INITIALIZE MMA OPTIMIZER
m     = 1;                % The number of general constraints.
n     = nele;             % The number of design variables x_j.
xmin  = zeros(n,1);       % Column vector with the lower bounds for the variables x_j.
xmax  = ones(n,1);        % Column vector with the upper bounds for the variables x_j.
xold1 = x(:);             % xval, one iteration ago (provided that iter>1).
xold2 = x(:);             % xval, two iterations ago (provided that iter>2).
low   = ones(n,1);        % Column vector with the lower asymptotes from the previous iteration (provided that iter>1).
upp   = ones(n,1);        % Column vector with the upper asymptotes from the previous iteration (provided that iter>1).
a0    = 1;                % The constants a_0 in the term a_0*z.
a     = zeros(m,1);       % Column vector with the constants a_i in the terms a_i*z.
c_MMA = 10000*ones(m,1);  % Column vector with the constants c_i in the terms c_i*y_i.
d     = zeros(m,1);       % Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
%% START ITERATION
while change > 0.01
  loop = loop + 1;
  %% FE-ANALYSIS
  sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*nelx*nely,1);
  K = sparse(iK,jK,sK); K = (K+K')/2;
  U(freedofs) = K(freedofs,freedofs)\F(freedofs);
  %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
  ce = reshape(sum((U(edofMat)*KE).*U(edofMat),2),nely,nelx);
  c = sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce));
  dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
  dv = ones(nely,nelx);
  %% FILTERING/MODIFICATION OF SENSITIVITIES
  if ft == 1
    dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
  elseif ft == 2
    dc(:) = H*(dc(:)./Hs);
    dv(:) = H*(dv(:)./Hs);
  end
  % METHOD OF MOVING ASYMPTOTES
  xval  = x(:);
  f0val = c;
  df0dx = dc(:);
  fval  = sum(xPhys(:))/(volfrac*nele) - 1;
  dfdx  = dv(:)' / (volfrac*nele);
  [xmma, ~, ~, ~, ~, ~, ~, ~, ~, low,upp] = ...
  mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2, ...
  f0val,df0dx,fval,dfdx,low,upp,a0,a,c_MMA,d);
  % Update MMA Variables
  xnew     = reshape(xmma,nely,nelx);
  xPhys(:) = (H*xnew(:))./Hs;
  xold2    = xold1(:);
  xold1    = x(:);
  change = max(abs(xnew(:)-x(:)));
  x = xnew;
  %% PRINT RESULTS
  fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c, ...
    mean(xPhys(:)),change);
  %% PLOT DENSITIES
  colormap(gray); imagesc(1-xPhys); caxis([0 1]); axis equal; axis off; drawnow;
end