#=
Ipopt call
n::Int, # Number of variables
x_L::Vector{Float64}, # Variable lower bounds
x_U::Vector{Float64}, # Variable upper bounds
m::Int, # Number of constraints
g_L::Vector{Float64}, # Constraint lower bounds
g_U::Vector{Float64}, # Constraint upper bounds
nele_jac::Int, # Number of non-zeros in Jacobian
nele_hess::Int, # Number of non-zeros in Hessian
eval_f, # Callback: objective function
eval_g, # Callback: constraint evaluation
eval_grad_f, # Callback: objective function gradient
eval_jac_g, # Callback: Jacobian evaluation
eval_h = nothing) # Callback: Hessian evaluation
=#
#####################
## Setup ##
#####################
using Ipopt
using JuMP
using DataFrames
#cd("/Users/eliotabrams/Desktop/Advanced\ Industrial\ Organization\ 2/Julia_implementation_of_BLP")
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
##########################
## Simple Logit Model ##
##########################
# Setup the simple logit model
logit = Model(solver = IpoptSolver(tol = 1e-8, max_iter = 1000, output_file = "logit2.txt"));
# Define variables
@defVar(logit, g[1:L]);
@defVar(logit, xi[1:J]);
@defVar(logit, alpha);
@defVar(logit, beta_par[1:K]);
# We minimize the gmm objective with the identity as the weighting matrix
# subject to the constraints g = sum_j xi_j iv_j and market share equations
@setObjective(logit, Min, sum{g[l]^2,l=1:L});
@addConstraint(
logit,
constr[l=1:L],
g[l]==sum{xi[j]*iv[j,l], j=1:J}
);
@addNLConstraint(
logit,
constr[j=1:J],
xi[j]==log(s[j])-log(s0[j])+alpha*p[j]-sum{beta_par[k]*x[j,k],k=1:K}
);
# Solve the model
status = solve(logit);
# Print the results
print(status)
print("alpha = ", getValue(alpha))
print("beta_par = ", getValue(beta_par[1:K]))
# Save results to use in the setup of BLP Model
g_logit=getValue(g);
xi_logit=getValue(xi);
alpha_logit=getValue(alpha);
beta_par_logit=getValue(beta_par);
##################################
## Define Ipopt call for BLP ##
##################################
# There are 558 variables that we do not know much info on
n = 558;
x_L = -1000*ones(n);
x_U = 1000*ones(n);
# We have 538 equality constraints
m = 538;
g_L = zeros(m);
g_U = zeros(m);
# Number of non-zeros
nele_jac = 294196;
nele_hess = 0;
iv = convert(Array, iv);

# Calculating optimal weighting matrix
Omega = zeros(L, L);
for j=1:J
    Omega += (xi_logit[j]^2)*iv[j,:]'*iv[j,:]
end
W = inv((1/J)*Omega);

function eval_f(param)
    f=0;
    for n1=1:L
        for n2=1:L
            f += param[n1+20]*W[n1,n2]*param[n2+20]
        end
    end
    return f
end

function eval_grad_f(param, grad_f)
grad_f[1:29] = zeros(29)
grad_f[21:30] = 2*W*param[21:30]
grad_f[31:558] = zeros(528)
end

#=
g = [share, g]
g = [1:528, 529:538]
param = [beta, alpha, piInc, piAge, sigma, g, xi ]
param = [1:4, 5, 6:10, 11:15, 16:20, 21:30, 31:558]
=#

function eval_g(param, g)
# Variables
beta = param[1:4];
alpha = param[5];
piInc = param[6:10];
piAge = param[11:15];
sigma = param[16:20];
xi = param[31:558];
denom = zeros(N,1);
tau = zeros(J,N);
kau = zeros(J,N);
sumtau = zeros(J);
    for n=1:N
        denom[n] = 1 # Outside option 
        for j=1:J
		for k=1:K
			kau[j,n] += (beta[k]+piInc[k]*inc[n]+piAge[k]*age[n]+sigma[k]*v[n,k])*x[j,k]
		end
            tau[j,n] = exp(kau[j,n]-(alpha+piInc[K+1]*inc[n]+piAge[K+1]*age[n]+sigma[K+1]*v[n,K+1])*p[j]+xi[j])
            denom[n] += tau[j,n]
        end
        for j=1:J
            tau[j,n]=tau[j,n]/denom[n]
        end
    end 
    for j=1:J
        for n=1:N
            sumtau[j] += tau[j,n]
        end
        g[j] = s[j] - sumtau[j] / N
    end
g[529:538] = iv'*xi[:] - param[21:30]
end


function eval_jac_g(param, mode, rows, cols, values)
# Variables
xm = [x p]
Km = K+1
beta = param[1:4];
alpha = param[5];
piInc = param[6:10];
piAge = param[11:15];
sigma = param[16:20];
xi = param[31:558];
denom = zeros(N,1);
tau = zeros(J,N);
kau = zeros(J,N);
pau = zeros(20,N);
jac = zeros(J+L,J+L);
    for n=1:N
        denom[n] = 1 # Outside option 
        for j=1:J
		for k=1:K
			kau[j,n] += (beta[k]+piInc[k]*inc[n]+piAge[k]*age[n]+sigma[k]*v[n,k])*x[j,k]
		end
            tau[j,n] = exp(kau[j,n]-(alpha+piInc[K+1]*inc[n]+piAge[K+1]*age[n]+sigma[K+1]*v[n,K+1])*p[j]+xi[j])
            denom[n] += tau[j,n]
        end
        for j=1:J
            tau[j,n]=tau[j,n]/denom[n]
        end
    end

    for n=1:N
        for k=1:Km
            for j=1:N            
                pau[k,n] +=tau[j,n]*xm[j,k]
            end
        end
    end
    for j=1:J
        for k=1:Km
            for n=1:N
                jac[j,k] += tau[j,n]*(xm[j,k]-pau[k,n])
                jac[j,Km+k] += tau[j,n]*(xm[j,k]-pau[k,n])*piInc[n]
                jac[j,2*Km+k] += tau[j,n]*(xm[j,k]-pau[k,n])*piAge[n]
                jac[j,3*Km+k] += tau[j,n]*(xm[j,k]-pau[k,n])*v[n,k]
            end            
        end
        for kk=1:4*Km
            jac[j,kk] = jac[j,kk]/N
        end
        for n=1:N
            jac[j,4*Km+L+j] += tau[j,n]
            for jj=1:J
                jac[j,4*Km+L+jj] -= tau[j,n]*tau[jj,n]
            end
        end
        for jj=1:J
            jac[j,4*Km+L+jj] = jac[j,4*Km+L+jj]/N
        end
	end

    jac[J+1:J+L,4*Km+1:4*Km+L]=eye(L);
    jac[J+1:J+L,4*Km+L+1:4*Km+L+J] = -iv';
    
(Eye, Jay, Vee) = findnz(jac); 
if mode == :Structure
  	rows[:] = Eye; cols[:] = Jay;
else
  	values[:] = Vee;
end

end

#####################
## Call Ipopt ##
#####################
# Create the problem
prob = createProblem(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
eval_f, eval_g, eval_grad_f, eval_jac_g)
# Set warm start and additional options
prob.x = zeros(n)
addOption(prob, "hessian_approximation", "limited-memory")
# Solve
status = solveProblem(prob)
println(Ipopt.ApplicationReturnStatus[status])
println(prob.x)
println(prob.obj_val)
