%%%% AN 88 LINE TOPOLOGY OPTIMIZATION CODE Nov, 2010 %%%%

%% FILTERING METHODS:
% ft = 1: sensitivity filtering
% ft = 2: density filtering
% ft = 3: heaviside projection

%% INPUTS
% top(128, 64, 0.2, 3, 3.84, 1)

function top88custom(nelx, nely, volfrac, penal, rmin, ft)
%% MATERIAL PROPERTIES
E0 = 1;
Emin = 1e-9;
nu = 0.3;

%% PREPARE FINITE ELEMENT ANALYSIS
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);

% DEFINE LOADS AND SUPPORTS

% Define passive elements (forced to be void or solid)
% 0 => free to change
% 1 => fixed at void
% 2 => fixed at solid
passive = zeros(nely, nelx);

% Define loading forces by x and y components
forcesx = sparse(nely + 1, nelx + 1);
forcesy = sparse(nely + 1, nelx + 1);

% Define nodes which are fixed in x and y directions
% Set a node value to 1 to fix it in that direction
fixedx = sparse(nely + 1, nelx + 1);
fixedy = sparse(nely + 1, nelx + 1);

%%%
%%% DEFINE BOUNDARY CONDITIONS HERE
%%%

% Upwards hydrostatic pressure on the bottom
forcesy(nely + 1, :) = 1;
% Thruster load 1 (leftwards distributed load on the right edge)
forcesx(:, nelx + 1) = -1;
% Thruster load 2
forcesy(1, nelx - 4:nelx+1) = -1;

% Fix the left edge of the design domain
fixedx(:, 1) = 1;
fixedy(:, 1) = 1;
% Fix the horizontal components of the right edge
fixedx(:, nelx + 1) = 1;

nforces = nnz(forcesx) + nnz(forcesy);

[x_row, x_col, x_val] = find(forcesx);
[y_row, y_col, y_val] = find(forcesy);

F = sparse( ...
    [ ...
        2 * ((x_col - 1) * (nely + 1) + x_row) - 1; ... % Forces in x direction
        2 * ((y_col - 1) * (nely + 1) + y_row) ...      % Forces in y direction
    ], ...
    [1:nforces], ...         % The vector number/column in the matrix (one column for each force)
    [x_val; y_val], ...      % Combine x and y forces into one vector
    2*(nely+1)*(nelx+1), ... % One entry for each DOF
    nforces ...              % One column is created for each force
);

% Displacement vectors
U = zeros(2*(nely+1)*(nelx+1), nforces); % One column for each force

% Fixed degrees of freedom
[x_row, x_col, ~] = find(fixedx);
[y_row, y_col, ~] = find(fixedy);
fixeddofs = [ ...
    2 * ((x_col - 1) * (nely + 1) + x_row) - 1; ... % Nodes fixed in x direction
    2 * ((y_col - 1) * (nely + 1) + y_row) ...      % Nodes fixed in y direction
];

alldofs = [1:2*(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs,fixeddofs);

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
beta = 1;
if ft == 1 || ft == 2
    xPhys = x;
elseif ft == 3
    xTilde = x;
    xPhys = 1 - exp(-beta * xTilde) + xTilde * exp(-beta);
end

loopbeta = 0;
loop = 0;
change = 1;

%% START ITERATION
while change > 0.01
    loopbeta = loopbeta + 1;
    loop = loop + 1;

    %% FE-ANALYSIS
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),64*nelx*nely,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);

    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    c = 0;
    dc = 0;
    for i = 1:size(F, 2)
        Ui = U(:, i);
        ce = reshape(sum((Ui(edofMat) * KE).*Ui(edofMat), 2), nely, nelx);
        c = c + sum(sum((Emin + xPhys.^penal*(E0-Emin)).*ce));
        dc = dc - penal * (E0-Emin) * xPhys.^(penal-1).*ce;
    end
    dv = ones(nely,nelx);

    %% FILTERING/MODIFICATION OF SENSITIVITIES
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    elseif ft == 3
        dx = beta * exp(-beta * xTilde) + exp(-beta);
        dc(:) = H*(dc(:).*dx(:)./Hs);
        dv(:) = H*(dv(:).*dx(:)./Hs);
    end

    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        if ft == 1
            xPhys = xnew;
        elseif ft == 2
            xPhys(:) = (H*xnew(:))./Hs;
        elseif ft == 3
            xTilde(:) = (H*xnew(:))./Hs;
            xPhys = 1-exp(-beta*xTilde) + xTilde*exp(-beta);
        end

        % Force passive elements to be void or solid
        xPhys(passive==1) = 0;
        xPhys(passive==2) = 1;

        if sum(xPhys(:)) > volfrac*nelx*nely
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;
  
    %% UPDATE BETA
    if ft == 3 && beta < 512 && (loopbeta >= 50 || change <= 0.01)
        beta = 2*beta;
        loopbeta = 0;
        change = 1;
        fprintf('Parameter beta increased to %g.\n', beta);
    end
    
    %% PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,c, ...
      mean(xPhys(:)),change);

    %% PLOT DENSITIES
    colormap(gray); imagesc(1-xPhys); caxis([0 1]); axis equal; axis off; drawnow;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab code was written by E. Andreassen, A. Clausen, M. Schevenels,%
% B. S. Lazarov and O. Sigmund,  Department of Solid  Mechanics,           %
%  Technical University of Denmark,                                        %
%  DK-2800 Lyngby, Denmark.                                                %
% Please sent your comments to: sigmund@fam.dtu.dk                         %
%                                                                          %
% The code is intended for educational purposes and theoretical details    %
% are discussed in the paper                                               %
% "Efficient topology optimization in MATLAB using 88 lines of code,       %
% E. Andreassen, A. Clausen, M. Schevenels,                                %
% B. S. Lazarov and O. Sigmund, Struct Multidisc Optim, 2010               %
% This version is based on earlier 99-line code                            %
% by Ole Sigmund (2001), Structural and Multidisciplinary Optimization,    %
% Vol 21, pp. 120--127.                                                    %
%                                                                          %
% The code as well as a postscript version of the paper can be             %
% downloaded from the web-site: http://www.topopt.dtu.dk                   %
%                                                                          %
% Disclaimer:                                                              %
% The authors reserves all rights but do not guaranty that the code is     %
% free from errors. Furthermore, we shall not be liable in any event       %
% caused by the use of the program.                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

