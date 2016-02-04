### 2016 Winter Advanced IO PS2 by Eliot Abrams, Hyunmin Park, Alexandre Sollaci

# Load data
using DataFrames
product = readtable("dataset_cleaned.csv")
population = readtable("population_data.csv")

# Define variables
x = product[:,3:6]
p = product[:,7]
z = product[:,8:13]
s0 = product[:,14]
s = product[:,2]
iv = hcat(x,z)
iv = convert(Array, iv)

inc = population[:,1]
age = population[:,2]
v = population[:,3:7]

# Calculate dimensions
K = size(x,2)
L = K+size(z,2)
J = size(x,1)
N = size(v,1)
M = size(v,2)

### 
#Simple Logit Model (From this model, we will obtain the optimal weighting matrix and the starting values for BLP.)
###

tic()
# Setting up the model
using JuMP
m = Model()


# Defining variables
@defVar(m, g[1:L])
@defVar(m, xi[1:J])
@defVar(m, alpha)
@defVar(m, beta[1:K])

# We minimize the gmm objective 
@setObjective(m,Min,sum{g[l]^2,l=1:L})

# g = sum_j xi_j iv_j
for l=1:L
	@addConstraint(m,g[l]==sum{xi[j]*iv[j,l],j=1:J}) 
end

# market share equations
for j=1:J
	@addNLConstraint(m,xi[j]==log(s[j])-log(s0[j])+alpha*p[j]-sum{beta[k]*x[j,k],k=1:K})
end

using Ipopt
setSolver(m,IpoptSolver(tol = 1e-5, 
hessian_approximation="limited-memory", max_iter = 1000, output_file = "logit.txt"))

status=solve(m)
toc()

print(status)

g_logit=getValue(g)
xi_logit=getValue(xi)
alpha_logit=getValue(alpha)
beta_logit=getValue(beta)

println("alpha = ", getValue(alpha))
println("beta = ", getValue(beta[1:K]))

# Calculating optimal weighting matrix
Omega = zeros(L, L)
for j=1:J
Omega += (xi_logit[j]^2)*iv[j,:]'*iv[j,:]
end
W = inv((1/J)*Omega)

