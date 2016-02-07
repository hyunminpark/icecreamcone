using DataFrames
using Ipopt

#####################
## Data ##
#####################
# Load data
product = DataFrames.readtable("dataset_cleaned.csv", separator = ',', header = true);
population = DataFrames.readtable("population_data.csv", separator = ',', header = true);
# Define variables
x = convert(Array, product[:,3:6]);
p = convert(Array, product[:,7]);
z = convert(Array, product[:,8:13]);
s0 = convert(Array, product[:,14]);
s = convert(Array, product[:,2]);
iv = [x z];
inc = convert(Array, population[:,1]);
age = convert(Array, population[:,2]);
v = convert(Array, population[:,3:7]);
# Store dimensions
K = size(x,2);
L = K+size(z,2);
J = size(x,1);
N = size(v,1);
M = size(v,2);


function eval_jac_g(param, mode, rows, cols, values)
# Variables
alpha = param[1];
beta_par = param[2:5];
piInc = param[6:10];
piAge = param[11:15];
sigma = param[16:20];
xi = param[31:558];
d_alpha = zeros(J,N);
d_beta_par = zeros(J,K,N);
d_piInc = zeros(J,K+1,N);
d_piAge = zeros(J,K+1,N);
d_sigma = zeros(J,K+1,N);
d_xi = zeros(J,J,N);
denom = zeros(N,1);
tau = zeros(J,N);
for n = 1:N
denom[n] = sum( exp(
-(alpha + piInc[K+1]*inc[n] + piAge[K+1]*age[n] + sigma[K+1]*v[n,K+1])*p
+ x[:,1:K]*(beta_par[1:K] + piInc[1:K]*inc[n] + piAge[1:K]*age[n] + Diagonal(sigma[1:K])*v'[1:K,n] )
+ xi )
)
tau[:,n] = exp(
-(alpha + piInc[K+1]*inc[n] + piAge[K+1]*age[n] + sigma[K+1]*v[n,K+1])*p
+ x[:,1:K]*(beta_par[1:K] + piInc[1:K]*inc[n] + piAge[1:K]*age[n] + Diagonal(sigma[1:K])*v'[1:K,n] )
+ xi )/denom[n]
end
# Define derivatives point by point
for n = 1:N
for j = 1:J
d_alpha[j,n] = p[j]*tau[j,n] - sum(Diagonal(p)*tau[:,n])*tau[j,n]
d_piInc[j,K+1,N] = (p[j]*tau[j,n] - sum(Diagonal(p)*tau[:,n])*tau[j,n])*inc[n]
d_piAge[j,K+1,N] = (p[j]*tau[j,n] - sum(Diagonal(p)*tau[:,n])*tau[j,n])*age[n]
d_sigma[j,K+1,N] = (p[j]*tau[j,n] - sum(Diagonal(p)*tau[:,n])*tau[j,n])*v'[K+1,n]
for k=1:K
d_beta_par[j,k,n] = ( x[j,k]*tau[j,n] - sum(Diagonal(x[:,k])*tau[:,n])*tau[j,n] )
d_piInc[j,k,n] = ( x[j,k]*tau[j,n] - sum(Diagonal(x[:,k])*tau[:,n])*tau[j,n] )*inc[n]
d_piAge[j,k,n] = ( x[j,k]*tau[j,n] - sum(Diagonal(x[:,k])*tau[:,n])*tau[j,n] )*age[n]
d_sigma[j,k,n] = ( x[j,k]*tau[j,n] - sum(Diagonal(x[:,k])*tau[:,n])*tau[j,n] )*v'[k,n]
end
for jj=1:J
if j == jj
d_xi[j,jj,n] = tau[j,n]*(1 - tau[j,n])
else
d_xi[j,jj,N] = - tau[j,n]*tau[jj,n]
end
end
end
end
# Define derivative matrices
D_alpha = (1/N)*sum(d_alpha, 2);
D_beta_par = (1/N)*sum(d_beta_par, 3);
D_piInc = (1/N)*sum(d_piInc, 3);
D_piAge = (1/N)*sum(d_piAge, 3);
D_sigma = (1/N)*sum(d_sigma, 3);
D_theta = [D_beta_par D_alpha D_piInc D_piAge D_sigma];
D_xi = (1/N)*sum(d_xi, 3);
# Now that we have derivatives, the rest is easy.
len_theta = 1 + K + 3*(K+1)
zero_1 = zeros(J,L)
zero_2 = zeros(L,len_theta)
I_g = eye(L)
jac = [D_theta D_xi zero_1 ; zero_2 -iv' I_g]
jac = convert(Array{Float64,2}, jac[1:size(jac,1), 1:size(jac,2)])
(Eye, Jay, Vee) = findnz(jac);
if mode == :Structure
rows[:] = Eye; cols[:] = Jay;
else
values[:] = Vee;
end
end

