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
EnableNLPResolve();
srand(12);


###############################
####       Functions        ###
###############################

function simulate(theta1, theta2, RC, periods)

	# Create new simulated data
	beta = 0.9
	mu = 0.2
	sigma = 0.5
	x_sim = [2.0]
	for t in 1:periods
		u_a_0 = -(theta1*x_sim[t] + theta2*x_sim[t]^2) + rand(Gumbel())
		u_a_1 = -RC + rand(Gumbel())

		if (u_a_0 > u_a_1)
			push!(x_sim, x_sim[t] + exp(rand(Normal(mu, sigma))))
		else 
			push!(x_sim, exp(rand(Normal(mu, sigma))))
		end
	end

	# Check the state space
	sort(unique(round(x_sim)))

	# Check that replacements were made reasonably frequently in the simulation
	actions_sim = zeros(size(x_sim,1),1);
	for j = 1:size(x_sim,1)-1
		actions_sim[j] = x_sim[j+1] < x_sim[j];
	end
	sum(actions_sim)

	return x_sim
end

function create_probabilities(x_discretized)

	# Create the state space and actions
	x_grid = sort(unique(x_discretized));
	actions = zeros(size(x_discretized,1),1);
	for j = 1:size(x_discretized,1)-1
		actions[j] = x_discretized[j+1] < x_discretized[j];
	end

	# State transition probabilities
	T = size(x_discretized,1);
	J = Int(size(x_grid, 1));
	p3 = zeros(J,J,2); 
	P = zeros(2,J);
	action_values = [0.0, 1.0]
	collected = collect(zip(x_discretized,actions))
	collected_future = collect(zip(x_discretized[2:T],x_discretized[1:T-1],actions[1:T-1]))
	for i in 1:J
		i_value = x_grid[i]
		for j in 1:J
			j_value = x_grid[j]
			for a in 1:2
				a_value = action_values[a]
				p3[j,i,a] = max(
					countnz(collected_future .== (j_value,i_value,a_value)) 
					/ countnz(collected .== (i_value,a_value)), 1.0e-100);
				if j == 1
					P[a,i] = max(
						countnz(collected .== (i_value,a_value)) 
						/ countnz(x_discretized .== i_value), 1.0e-100);
				end
			end
		end
	end

	return p3, P, x_grid, actions, J
end

function create_estimates(x, round_param, beta)

	# Discretize the data
	x_discretized = round(x, round_param);
	x_grid = sort(unique(x_discretized));

	p3, P, x_grid, actions, J = create_probabilities(x_discretized);

	return [MPEC(J, x_discretized, x_grid, actions, p3, P, beta); 
			HMSS(J, x_discretized, x_grid, actions, p3, P, beta)]
end

# NOW ONLY USES THE FIRST 10,000 OBSERVATIONS IT IS FED
function MPEC(J, x_discretized, x_grid, actions, p3, P, beta)

	T = 10000;

	index = [];
	for t = 1:T
		aux = find(x_grid -> x_grid == x_discretized[t], x_grid);
		index = [index; aux];
	end 

	m = Model(solver = IpoptSolver() );
	@defVar(m, EV_1[1:J]);
	@defVar(m, EV_0[1:J]);
	@defVar(m, theta[i=1:2]);
	@defVar(m, RC >= 0);

	@addNLConstraint(m, 
					constr[i=1:J], 
					EV_1[i] == sum{ 
						log( exp( -RC + beta*EV_1[j] ) + exp(-theta[1]*x_grid[j] 
							- theta[2]*(x_grid[j])^2 + beta*EV_0[j] ) )*p3[j,i,2]
					, j = 1:J} );
	@addNLConstraint(m, 
					constr[i=1:J], 
					EV_0[i] == sum{ 
						log( exp( -RC + beta*EV_1[j] ) + exp(-theta[1]*x_grid[j] 
							- theta[2]*(x_grid[j])^2 + beta*EV_0[j] ) )*p3[j,i,1] 
					, j = 1:J} );

	@setNLObjective(m, 
					Max, 
					sum{ 
					log( (actions[t]*exp( -RC + beta*EV_1[index[t]] ) 
						+ (1-actions[t])*exp(-theta[1]*x_discretized[t] - theta[2]
							*(x_discretized[t])^2 + beta*EV_0[index[t]] ) 
					) / 
					( exp( -RC + beta*EV_1[index[t]] )  + exp(-theta[1]*x_discretized[t] 
						- theta[2]*(x_discretized[t])^2 + beta*EV_0[index[t]] ) ) ) +
			 				log( p3[index[t], index[t-1], actions[t-1] + 1] ) 
			 		, t = 2:T} ) ;

	status = solve(m);

	return [getValue(theta[1]), getValue(theta[2]), getValue(RC)];
end 

function HMSS(J, x_discretized, x_grid, actions, p3, P, beta)
	# Length of Simulated path
	Ns = 500; 
	y, z0, z1, z2, zrc  = zeros(J), zeros(J), zeros(J), zeros(J), zeros(J)
	# Is(i,a,ns) is simulated state index at time ns for initial state index i choosing initial action a
	# Xs(i,a,ns) is simulated state value at time ns for initial state index i choosing initial action a
	Is, Xs = zeros(J,2,Ns), zeros(J,2,Ns) 
	p2 = zeros(J,J)
	for j in 1:J
		for i in 1:J
			p2[j,i] = p3[j,i,1]*P[1,i] + p3[j,i,2]*P[2,i]
		end
	end
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
	hmss = Model(solver = IpoptSolver() );
	@defVar(hmss, theta[1:2]);
	@defVar(hmss, RC >= 0);
	@setNLObjective(
		hmss, 
		Min, 
		sum{Px[i]*(y[i]-z0[i]-zrc[i]*RC-z1[i]*theta[1]-z2[i]*theta[2])^2,i = 1:J}
	);
	status = solve(hmss);

	return [getValue(theta[1]), getValue(theta[2]), getValue(RC)];
end

###############################
####       Simulation       ###
###############################
# The code works, but does not easily pick up the correct results

# Run the simulation for beta = 0
x_simulated = simulate(5.0, 0.0, 8.0, 20000);

# See whether the estimation algorithms work
results = create_estimates(x_simulated, 1, 0.0)
print(results)


###############################
####        Results         ###
###############################

# Read in data
# cd("/Users/eliotabrams/Desktop/Advanced\ Industrial\ Organization\ 2/Problem Set\ 3");
data = readtable("data.csv", separator = ',', header = true);
x = convert(Array, data[:x]);

# Create the estimates for the different beta and discretization sensitivities
vars = [],[],[],[],[],[];
csvfile = open("results.csv","w");
write(csvfile, 
	"Beta, Rounding, HM_theta1, HM_theta2, HM_RC, MPEC_theta1, MPEC_theta2, MPEC_RC");
for beta in [0, 0.5, 0.8, 0.9, 0.95]
	if beta == 0.9 
		round_params = [1, 2]
	else
		round_params = [1]
	end
	for round_param in round_params
		# Get main results
		results = create_estimates(x, round_param, beta);
		for i in 1:6
			push!(vars[i], results[i]);
		end

		# Perform bootstrap
		for boot in 1:5
			x_boot = []
			for s in sample(1:5000:100000, 20)
				x_boot = [x_boot; x[s:(s+4998)]]
			end

			results = create_estimates(x_boot, round_param, beta);
			for i in 1:6
				push!(vars[i], results[i]);
			end
		end

		# Print results
		write(csvfile, string("\n", float(beta), ",", round_param, ","));
		for i in 1:6
			write(csvfile, 
				string(vars[i][1], " (", 
					std(convert(Array{Float64}, vars[i][2:size(vars[i],1)])), "),"));
		end
		vars = [],[],[],[],[],[];	
	end
end
close(csvfile);




