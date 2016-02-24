###############################
####      Assignment 3      ###
###############################

# Setup
using Ipopt
using JuMP
using DataFrames
EnableNLPResolve()

# Load data
data = readtable("data.csv", separator = ',', header = true);
x = convert(Array, data[:x]);

# Generate a
actions = zeros(size(x,1),1)
for j = 1:size(x,1)-1
	actions[j] = x[j+1] < x[j]
end

# Transition matrix

# Discretize the state space (will loop through)
# Should run from 0 to 14
x_discretized = round(x, 0)

# Transition probabilities
p3 = Dict()
for i in unique(x_discretized)
	for j in unique(x_discretized)
		for a in [0, 1]
			count = 0
			for t in 1:size(x_discretized,1)-1
				count += (j == x_discretized[t+1]) & (x_discretized[t] == i) & (actions[t] == a)
			end
			p3[(j,i,a)] = count / size(x_discretized,1)
		end
	end
end

# Choice probabilities
P = Dict()
for i in unique(x_discretized)
	for a in [0, 1]
		count = 0
		for t in 1:size(x_discretized,1)
			count += (actions[t] == a) & (i == x_discretized[t])
		end
		P[(i,a)] = count / size(x_discretized,1)
	end
end

###### Hotz Miller

Ns = 1000; # Length of Simulated path
xgrid = unique(x_discretized);
y   = Dict()
z0  = Dict()
z1  = Dict()
z2  = Dict()
zrc = Dict()
for i in unique(x_discretized)
	# Simulate paths	
	Xs = rand(Ns,2)
	As = rand(Ns,2)
	Xs[1] = 
	for ns in 2:Ns
		Xs[ns]=
	end
	# For the Objective Function
	y[(i)]=log(P[(i,1)]/P[(i,0)])
	z0[(i)]=beta^t*
end



