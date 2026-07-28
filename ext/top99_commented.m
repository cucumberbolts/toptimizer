%%%% A 99 LINE TOPOLOGY OPTIMIZATION CODE BY OLE SIGMUND, JANUARY 2000 %%%
%%%% CODE MODIFIED FOR INCREASED SPEED, September 2002, BY OLE SIGMUND %%%

function top(nelx,nely,volfrac,penal,rmin);
% nelx = number of elements in x (horizontal direction)
% nely = number of elements in y (vertical direction)
% volfrac = prescribed volume fraction (fraction of the design domain to remain solid), which limits the amount of material used. Basically this ensures that the material volume used in the design does not exceed a fraction volfrac of the total volume V_0 of the design space
% penal = the penalization factor in SIMP method, typically set to 3, is used to penalize intermediate material densities, pushing the solution towards 0 or 1 (void or solid)
% rmin = filter radius to ensure mesh-independency filtering (helps avoid checkerboarding and mesh-sensitive designs). It defines a circular (or spherical, in 3D problems) neighborhood around each element within which the properties (such as sensitivities or densities) of the surrounding elements are averaged

% INITIALIZE
x(1:nely,1:nelx) = volfrac;
% x = the design variable representing the material density distribution across the design domain. Initially, it is filled with the value volfrac, meaning the initial guess is uniformly distributing the material according to the volume fraction
% starting off, x is just a 2D array filled with volfrac in all its elements
loop = 0; % counter to track number of optimization iterations
change = 1.; % Variable to store maximum change in material distribution between iterations

% START ITERATION
while change > 0.01 % optimization stops when the change in material distribution is less than 1%
  loop = loop + 1; % increase loop counter 
  xold = x; % storing the current material density distrubution so that optimization can be done in the next step directly on x

  % FE-ANALYSIS
  [U]=FE(nelx,nely,x,penal);
  % [U] = global displacement vector
  % FE = FEA function
  % FEA solves the linear system KU = F

% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
  [KE] = lk;
  % [KE] = local element stiffness matrix
  % lk = function to calculate local stiffness element matrix for unit
  % density element

  c = 0.; % initialize compliance (objective function) to zero

  % loop over all elements in the design domain
  for ely = 1:nely
    for elx = 1:nelx
      n1 = (nely+1)*(elx-1)+ely; % denotes upper left element node number in global node matrix
      n2 = (nely+1)* elx   +ely; % denotes upper right element node number in global node matrix

      % example global node matrix for nelx = 4 and nely = 3
      % ex = element number (column by column starting from the left)
      % 1    5    9    13   17
      %  ---- ---- ---- -----
      % | e1 | e4 | e7 | e10 |
      %  ---- ---- ---- -----
      % 2    6    10   14   18
      %  ---- ---- ---- -----
      % | e2 | e5 | e8 | e11 |
      %  ---- ---- ---- -----
      % 3    7    11   15   19
      %  ---- ---- ---- -----
      % | e3 | e6 | e9 | e12 |
      %  ---- ---- ---- -----
      % 4    8    12   16   20
      
      Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1); % extracts the element displacement vector Ue from the global displacement vector U. This is needed to compute the element's contribution to the compliance and its sensitivity
      % compliance calculation
      c = c + x(ely,elx)^penal*Ue'*KE*Ue; % accumulates the total compliance
      dc(ely,elx) = -penal*x(ely,elx)^(penal-1)*Ue'*KE*Ue; % sensitivity of compliance with respect to material density calculation
    end
  end
% FILTERING OF SENSITIVITIES
  [dc]   = check(nelx,nely,rmin,x,dc); % calls sensitivity filtering function check to ensure mesh-independency. The function smooths the sensitivities across neighboring elements within a radius of rmin, averaging them to prevent localized, mesh-dependent changes    

% DENSITY FILTER AND THRESHOLD PROJECTION
  
  
% DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD
  [x]    = OC(nelx,nely,x,volfrac,dc); % updates the design variables (x) using the Optimality Criteria (OC) method. The OC method adjusts the material densities based on the sensitivities and ensures that the volume constraint (volfrac) is satisfied
% PRINT RESULTS
  change = max(max(abs(x-xold))); %change = max(max(abs(x - xold))): Computes the maximum change in the design variables between iterations. If this value is less than 0.01, the loop will terminate
  disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',c) ...
       ' Vol.: ' sprintf('%6.3f',sum(sum(x))/(nelx*nely)) ...
        ' ch.: ' sprintf('%6.3f',change )])
% PLOT DENSITIES  
  colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);
end
%%%%%%%%%% OPTIMALITY CRITERIA UPDATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xnew]=OC(nelx,nely,x,volfrac,dc) % OC method used to update the design variables (densities)  
% [xnew] = updated design variable with new densities
% nelx = number of elements in x (horizontal direction)
% nely = number of elements in y (vertical direction)
% x = current design variables (density values) for each element
% volfrac = prescribed volume fraction, i.e., the total amount of material allowed to be used.
% dc = sensitivity of the compliance (objective function) with respect to the design variables x (dc/dx_e, equation (4) in paper)

l1 = 0; l2 = 100000; move = 0.2;
% l1 = the lower bound of the Lagrange multiplier λ, which is used to enforce the volume constraint
% l2 = the upper bound of λ. These bounds will be updated during the bisection method to find the optimal λ
% move = the move limit, which restricts how much the material densities can change between iterations. This ensures stability and prevents large, abrupt changes in the design

% bisection method to find optimal λ
while (l2-l1 > 1e-4) % run the loop until the difference between the upper and lower bounds is really small (1e-4)
  lmid = 0.5*(l2+l1); % midpoint of upper and lower bounds chosen as next guess for λ

  xnew = max(0.001,max(x-move,min(1.,min(x+move,x.*sqrt(-dc./lmid)))));% this line computes the updated material densities xnew for each element using the sensitivities dc and the current guess for the Lagrange multiplier lmid
  % dc values are already negative to start with. -dc./lmid is dividing the sensitivity of the compliance with respect to the design variable x (dc)
  % The term sqrt(-dc./lmid) represents the optimal step size for updating x, and the scaling factor lmid ensures that the total material volume is controlled.
  % min(x+move,x.*sqrt(-dc./lmid)): ensures the updated density xnew does not increase by more than the move limit 
  % max(x-move,min(1.,min(x+move,x.*sqrt(-dc./lmid)))): ensures that the updated density xnew does not decrease by more than the move limit.
  % max(0.001,max(x-move,min(1.,min(x+move,x.*sqrt(-dc./lmid))))): This ensures that the updated density xnew does not fall below a small value (0.001), which prevents numerical issues such as division by zero. This lower bound ensures that elements do not become fully void.

  if sum(sum(xnew)) - volfrac*nelx*nely > 0;
    l1 = lmid;
  else
    l2 = lmid;
  end
  % sum(sum(xnew)): This calculates the total volume of material in the updated design by summing all the density values in xnew
  % volfrac * nelx * nely: This represents the maximum allowable material volume (as dictated by the volume fraction volfrac).
  % If the total material volume in xnew exceeds the allowed volume (volfrac * nelx * nely), the Lagrange multiplier l1l1 is increased (l1 = lmid) to penalize material use and reduce the total volume.
  % Otherwise, if the total material volume is within the allowable limit, the Lagrange multiplier l2l2 is decreased (l2 = lmid) to allow for more material to be used.
end
%%%%%%%%%% MESH-INDEPENDENCY FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dcn]=check(nelx,nely,rmin,x,dc)
% [dcn]: the output is the filtered sensitivities, denoted as dcn. These are modified versions of the original sensitivities dc after applying the mesh-independency filter.
% nelx = number of elements in x (horizontal direction)
% nely = number of elements in y (vertical direction)
% rmin = filter radius that determines the extent of the neighborhood for filtering. It defines how far the filter will reach to neighboring elements to smooth out the sensitivities.
% x = design variable with current material density distribution
% dc = sensitivity of the compliance (objective function) with respect to the design variables x (dc/dx_e, equation (4) in paper)

dcn=zeros(nely,nelx); % initialize an array containing the filtered sensitivities of the compliance with respect to design variable x (material distribution)
% iterate over all elements in the mesh
for i = 1:nelx
  for j = 1:nely
    sum=0.0; % accumulates the sum of the weight factors fac used in the filtering process for each element
    % filtering neighbouring elements within the radius rmin
    % these loops iterate over neughbouring elements within a square filter of size 2*rmin around the current element (i,j)
    
    % k: represents the column index of a neighbouring element
    % l: represents the row index of a neighbouring element
    for k = max(i-floor(rmin),1):min(i+floor(rmin),nelx) % ensures index k stays within 1 and nelx boundaries
      for l = max(j-floor(rmin),1):min(j+floor(rmin),nely) % ensures index l stays within 1 and nely boundaries
        % fac: weight factor used to smooth sensitivities
        fac = rmin-sqrt((i-k)^2+(j-l)^2); % weight factor decreases as the distance increases. Elements closer to the current element (i,j) have a higher weight in the filtering process. This ensures that elements within the filter radius rmin have a positive influence and elements beyond the radius have no influence
        sum = sum+max(0,fac); % elements beyond rmin should have no influence
        dcn(j,i) = dcn(j,i) + max(0,fac)*x(l,k)*dc(l,k);
        % Adds the weighted sensitivity of the neighboring element (l, k) to the filtered sensitivity dcn(j, i) of the current element (i, j).
        % The weight factor fac is multiplied by the density x(l,k) and the original sensitivity dc(l,k) of the neighboring element.
      end
    end
    dcn(j,i) = dcn(j,i)/(x(j,i)*sum); %After all the neighboring elements' contributions have been added, the filtered sensitivity dcn(j, i) is normalized by dividing it by the total sum of the weight factors.
  end
end
%%%%%%%%%% FE-ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U]=FE(nelx,nely,x,penal)
% [U]: output is the global displacement vector U that results from FEA
% nelx: number of elements in x (horizontal direction)
% nely: number of elements in y (vertical direction)
% x: current material density distribution for each element
% penal: penalization factor (typically set to 3) used in SIMP method to penalize intermediate densities, pushing them toward either solid (1) or void (0)

[KE] = lk; % calls function lk to compute local element stiffness matrix KE for a single element. This matrix is the same for all elements (but scaled by their densities) 

% most of the elements in K and F are zeros. By using sparse matrices, the
% code efficiently handles the large, mostly-zero matrices generated in
% FEA, saving memory and improving computational performance. Only non-zero
% elements are stored
K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1)); % 2*(nelx+1)*(nely+1) x 2*(nelx+1)*(nely+1) matrix (2DOFs = horizontal and vertical translational displacements)
F = sparse(2*(nely+1)*(nelx+1),1); U = zeros(2*(nely+1)*(nelx+1),1);

% iterate over all elements
for elx = 1:nelx
  for ely = 1:nely
    n1 = (nely+1)*(elx-1)+ely; % denotes upper left element node number in global node matrix
    n2 = (nely+1)* elx   +ely; % denotes upper right element node number in global node matrix
    edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2]; % each element has 4 nodes, each node has 2 DOFs (horizontal and vertical translational displacement)
    % order of edof:
    % index for x displacement of upper-left node
    % index for y displacement of upper-left node
    % index for x displacement of upper-right node
    % index for y displacement of upper-right node
    % index for x displacement of lower-right node
    % index for y displacement of lower-right node
    % index for x displacement of lower-left node
    % index for y displacement of lower-left node
    K(edof,edof) = K(edof,edof) + x(ely,elx)^penal*KE; % update global stiffness matrix K by adding contribution of current element
  end
end
% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
F(2,1) = -1; % applies vertical load of -1 (downward force) at the top-left corner of the structure
fixeddofs   = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]); % defines the fixed DOFs for the supports
% The array [1:2:2*(nely+1)] selects every second DOF along the left boundary (to apply horizontal and vertical constraints).
% The array [2*(nelx+1)*(nely+1)] selects the last vertical DOF on the right boundary (to apply a horizontal support).
alldofs     = [1:2*(nely+1)*(nelx+1)]; % This creates an array of all degrees of freedom in the structure.
freedofs    = setdiff(alldofs,fixeddofs); % This calculates the free degrees of freedom, i.e., all DOFs that are not constrained (not part of fixeddofs). These are the DOFs that will be solved for in the linear system.
% SOLVING
U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
U(fixeddofs,:)= 0;
%%%%%%%%%% ELEMENT STIFFNESS MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [KE]=lk
% [KE]: output is the local element stiffness matrix for a single quadrilateral element
E = 1.; % Young'e modulus set to 1 for simplicity in non-dimensionoalized problems
nu = 0.3; % Poisson's ratio of material (typical value for many materials is 0.3)
k=[ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
   -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
% array k contains the coefficients used to construct the 8x8 stiffness
% matrix KE
KE = E/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                  k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                  k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                  k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                  k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                  k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                  k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                  k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab code was written by Ole Sigmund, Department of Solid         %
% Mechanics, Technical University of Denmark, DK-2800 Lyngby, Denmark.     %
% Please sent your comments to the author: sigmund@fam.dtu.dk              %
%                                                                          %
% The code is intended for educational purposes and theoretical details    %
% are discussed in the paper                                               %
% "A 99 line topology optimization code written in Matlab"                 %
% by Ole Sigmund (2001), Structural and Multidisciplinary Optimization,    %
% Vol 21, pp. 120--127.                                                    %
%                                                                          %
% The code as well as a postscript version of the paper can be             %
% downloaded from the web-site: http://www.topopt.dtu.dk                   %
%                                                                          %
% Disclaimer:                                                              %
% The author reserves all rights but does not guaranty that the code is    %
% free from errors. Furthermore, he shall not be liable in any event       %
% caused by the use of the program.                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
