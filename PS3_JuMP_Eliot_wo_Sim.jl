###############################
####      Assignment 3      ###
###############################
# Advanced IO 2
# Eliot Abrams, Hyunmin Park, Alexandre Sollaci


###############################
####         Setup          ###
###############################

using Ipopt
using JuMP
using DataFrames
using StatsBase
using Distributions
using SymPy
import Optim
EnableNLPResolve();
cd("/Users/eliotabrams/Desktop/Advanced\ Industrial\ Organization\ 2/Problem Set\ 3");
srand(111);


###############################
####       Functions        ###
###############################

function create_probabilities(x_discretized)

	# Create the state space and actions
	x_grid = sort(unique(x_discretized))
	actions = zeros(size(x_discretized,1),1);
	for j = 1:size(x_discretized,1)-1
		actions[j] = x_discretized[j+1] < x_discretized[j];
	end

	# State transition probabilities
	p3d, p2d, Pd  = Dict(), Dict(), Dict();
	for i in x_grid
		for j in x_grid
			for a in [0, 1]
				num3, denom3 = 0, 0;
				num2, denom2 = 0, 0;
				num, denom = 0, 0;
				for t in 1:size(x_discretized,1)-1
					num3 += (x_discretized[t+1] == j) & (x_discretized[t] == i) & (actions[t] == a);
					denom3 += (x_discretized[t] == i) & (actions[t] == a);
					num2 +=(x_discretized[t+1] == j) & (x_discretized[t] == i);
					denom2 += (x_discretized[t] == i);
					if j == minimum(x_grid)
						num += (actions[t] == a) & (x_discretized[t] == i);
						denom += (x_discretized[t] == i);
					end
				end
				p3d[(j,i,a)] = max(num3 / denom3, 1.0e-8);
				p2d[(j,i)] = max(num2 / denom2, 1.0e-8);
				if j == minimum(x_grid)
					Pd[(a,i)] = max(num / denom, 1.0e-8);
				end
			end
		end
	end

	# Convert dictionaries to matrices
	J = Int(size(x_grid, 1));
	p3 = zeros(J,J,2); 
	p2 = zeros(J,J);
	P = zeros(2,J);
	for i = 1:J
		for j = 1:J 
			p2[j,i] = p2d[(x_grid[j],x_grid[i])]; 
			for a = 1:2
				p3[j,i,a] = p3d[(x_grid[j],x_grid[i],a-1)];
				if j == 1
					P[a,i] = Pd[(a-1,x_grid[i])]; 
				end
			end 
		end
	end

	return p3, p2, P, x_grid, actions, J
end

function create_estimates(x, round_param, beta, time_interval)

	# Get range
	starting_value = rand(1:size(x,1) - time_interval);

	# Discretize the data
	x_discretized = round(x[range(starting_value,time_interval)], round_param);
	x_grid = sort(unique(x_discretized))

	p3, p2, P, x_grid, actions, J = create_probabilities(x_discretized)

	return [abrams_hotz_miller(J, x_discretized, x_grid, actions, p3, p2, P, beta); 
			MPEC(J, x_discretized, x_grid, actions, p3, p2, P, beta)]
end

function MPEC(J, x_discretized, x_grid, actions, p3, p2, P, beta)

	T = Int(size(x_discretized,1));

	index = []; # define index to sum EV's over time.
	for t = 1:T
		aux = find(x_grid -> x_grid == x_discretized[t], x_grid);
		index = [index;aux]
	end 

	m = Model(solver = IpoptSolver() );
	@defVar(m, EV_1[1:J]); #defined over x_grid
	@defVar(m, EV_0[1:J]);
	@defVar(m, theta[1:2]);
	@defVar(m, RC >= 0);

	@addNLConstraint(m, constr[i=1:J], EV_1[i] == sum{ log( exp( -RC + beta*EV_1[j] )  + exp(-theta[1]*x_grid[j] - theta[2]*(x_grid[j])^2 + beta*EV_0[j] ) ) * p3[j,i,2] , j = 1:J} );
	@addNLConstraint(m, constr[i=1:J], EV_0[i] == sum{ log( exp( -RC + beta*EV_1[j] )  + exp(-theta[1]*x_grid[j] - theta[2]*(x_grid[j])^2 + beta*EV_0[j] ) ) * p3[j,i,1] , j = 1:J} );

	@setNLObjective(m, Max, sum{ 
		log( (actions[t]*exp( -RC + beta*EV_1[index[t]] ) + (1-actions[t])*exp(-theta[1]*x_discretized[t] - theta[2]*(x_discretized[t])^2 + beta*EV_0[index[t]] ) )/ 
			( exp( -RC + beta*EV_1[index[t]] )  + exp(-theta[1]*x_discretized[t] - theta[2]*(x_discretized[t])^2 + beta*EV_0[index[t]] ) ) ) +
 		log( p3[index[t], index[t-1], actions[t-1] + 1] ) ,t = 2:T} ) ;

	status = solve(m);

	return [getValue(theta[1]), getValue(theta[2]), getValue(RC)];
end 

function hotz_miller(J, x_discretized, x_grid, actions, p3, p2, P, beta)

	# Length of Simulated path
	Ns = 1000; 
	y, z0, z1, z2, zrc  = zeros(J), zeros(J), zeros(J), zeros(J), zeros(J)
	# Is(i,a,ns) is simulated state index at time ns for initial state index i choosing initial action a
	# Xs(i,a,ns) is simulated state value at time ns for initial state index i choosing initial action a
	Is, Xs = zeros(J,2,Ns), zeros(J,2,Ns) 
	
	for i in 1:J

		# Calculate the simulated paths
		for a in 1:2
			Is[i,a,1] = sample(1:J,weights(p3[:,i,a]))
			Xs[i,a,1] = x_grid[Is[i,a,1]]
			for ns in 2:Ns
				Is[i,a,ns] = sample(1:J,weights(p2[:,Is[i,a,ns-1]]))
				Xs[i,a,ns] = x_grid[Is[i,a,ns]]
			end	
		end

		# To construct the coefficients for the objective function
		y[i]=log(P[2,i]/P[1,i])
		for ns in 1:Ns
			z0[i]  += (beta^ns)*(-P[2,Is[i,2,ns]]*log(P[2,Is[i,2,ns]]) -P[1,Is[i,2,ns]]*log(P[1,Is[i,2,ns]]) + P[2,Is[i,1,ns]]*log(P[2,Is[i,1,ns]]) + P[1,Is[i,1,ns]]*log(P[1,Is[i,1,ns]]))
			zrc[i] += (beta^ns)*(-P[2,Is[i,2,ns]] + P[2,Is[i,1,ns]])
			z1[i]  += (beta^ns)*(-P[1,Is[i,2,ns]]*Xs[i,2,ns] + P[1,Is[i,1,ns]]*Xs[i,1,ns])
			z2[i]  += (beta^ns)*(-P[1,Is[i,2,ns]]*(Xs[i,2,ns]^2) + P[1,Is[i,1,ns]]*(Xs[i,1,ns]^2))
		end

	end

	# Find the stationary distribution to use as a weighting matrix in the minimization
	x_grid2 = vcat(-1,x_grid)
	Px=hist(x_discretized,x_grid2)
	Px=Px[2]./size(x_discretized,1)

	# Minimum Distance
	hm = Model(solver = IpoptSolver() );
	@defVar(hm, theta[1:2]);
	@defVar(hm, RC >= 0);
	@setNLObjective(
		hm, 
		Min, 
		sum{Px[i]*(y[i]-z0[i]-zrc[i]*RC-z1[i]*theta[1]-z2[i]*theta[2])^2,i = 1:J}
	);
	status = solve(hm);

	return [getValue(theta[1]), getValue(theta[2]), getValue(RC)];
end

function abrams_hotz_miller(J, x_discretized, x_grid, actions, p3, p2, P, beta)
	data = readtable("data.csv", separator = ',', header = true);
	x = convert(Array, data[:x]);
	x_discretized = round(x);
	p3, p2, P, x_grid, actions, J = create_probabilities(x_discretized)
	beta = 0.9

	# Create the variables
	@vars theta1 theta2 RC

	# Create estimates for v
	# Currently only works for integer state space
	v = Dict()
	for x in x_grid
		for a in 1:2
			container = []
			for p in 1:25
				u = 0.0
				x_sim = [rand(x_grid)]
				for t in 1:100
					a = sample(1:2, weights(P[:,x_sim[t]+1]))
					if a == 1
						push!(x_sim, x_grid[sample(x_grid, weights(p3[:,x_sim[t]+1,1]))+1])
						u += beta*(-(theta1*x_sim[t] + theta2*x_sim[t]^2) + rand(Gumbel()))
					else 
						push!(x_sim, x_grid[sample(x_grid, weights(p3[:,x_sim[t]+1,2]))+1])
						u += beta*(-RC + rand(Gumbel()))
					end
				end
				push!(container, u)
			end
			v[(x,a)]=mean(container)
		end
	end

	# Function to optimize over
	function f(t1, t2, t3)
		d = Dict("theta1" => t1, "theta2" => t2, "RC" => t3)
		
		P_theta = []
		for i in 1:size(x_grid,1)
			try
				push!(P_theta, exp(subs(v[(i, 2)],d)) 
					/ (subs(exp(v[(i, 2)]),d) + subs(exp(v[(i, 1)]),d)) )
			catch
				push!(P_theta, 0.0)
			end
		end

		objective = []
		for i in 1:size(x_grid,1)
			push!(objective, (P[2,i] - P_theta[i])^2)
		end

		return  sum(objective)
	end

	# Optimize
	Optim.optimize(f, [0.0, 0.0], Optim.NelderMead())

	return 
end

###############################
####        Results         ###
###############################

# Read in data
data = readtable("data.csv", separator = ',', header = true);
x = convert(Array, data[:x]);

# Create the estimates for the different beta and discretization sensitivities
vars = [],[],[],[],[],[];
csvfile = open("results.csv","w");
write(csvfile, 
	"Beta, Rounding, HM_theta1, HM_theta2, HM_RC, MPEC_theta1, MPEC_theta2, MPEC_RC");
# [0, 0.5, 0.8, 0.9, 0.95]
# [0, 1]
for beta in [0, 0.5, 0.8, 0.9, 0.95]
	for round_param in [0, 1, 2]

		# Perform bootstrap
		for boot in 1:5
			results = create_estimates(x, round_param, beta, 99998);
			for i in 1:6
				push!(vars[i], results[i]);
			end
		end

		# Print results
		write(csvfile, string("\n", float(beta), ",", round_param, ","));
		for i in 1:6
			write(csvfile, string(vars[i][1], " (", std(convert(Array{Float64}, vars[i])), "),"));
		end
		vars = [],[],[],[],[],[];
		
	end
end
close(csvfile);




