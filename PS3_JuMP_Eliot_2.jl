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
srand(1234);


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

function HM(J, x_discretized, x_grid, actions, p3, P, beta)

	# Control the loops
	S = 100;
	T = 2 + 100*beta;

	# Create estimates for the value functions
	# Only create estimates for the i we want to use in optimization!
	weighting = zeros(J);
	holder = Dict()
	for i in x_grid
		place_index = find(x_grid .== i)[1]
		container = Array[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
		if ( P[2,place_index] > 0.001 && P[1,place_index] > 0.001 )
			weighting[place_index] += 1
			for action in 1:2
				for s in 1:S
					x_sim = [i];
					if action == 1
						container[1][1] -= x_sim[1];
						container[1][2] -= x_sim[1]^2;
						container[1][4] += 0.577 - log(P[action,find(x_grid .== x_sim[1])][1]);
					else
						container[2][3] -= 1;
						container[2][4]	+= 0.577 - log(P[action,find(x_grid .== x_sim[1])][1]);
					end

					push!(x_sim, sample(x_grid, weights(p3[:,find(x_grid .== x_sim[1]),action])));

					for t in 2:T
						a = sample(1:2, weights(P[:, find(x_grid .== x_sim[t])]));
						if a == 1
							container[action][1] -= beta*x_sim[t];
							container[action][2] -= beta*x_sim[t]^2;
							container[action][4] += beta*(0.577 - log(P[a,find(x_grid .== x_sim[t])][1]));
						else 
							container[action][3] -= beta*1;
							container[action][4] += beta*(0.577 - log(P[a,find(x_grid .== x_sim[t])][1]));
						end
						push!(x_sim, sample(x_grid, weights(p3[:,find(x_grid .== x_sim[t]),a])));
					end
				end
			end
		end
		holder[i] = container ./= S;
	end

	# Optimize with clever weighting to get proper identification
	println(sum(weighting))
	index = sort(collect(keys(holder)))
	hm = Model(solver = IpoptSolver() );
	@defVar(hm, t[i=1:3]);
	@setNLObjective(
		hm, 
		Min, 
		sum{weighting[i]*(
			(log(P[2,i]) - log(P[1,i])) -
			(
				(holder[index[i]][2][1]*t[1] + holder[index[i]][2][2]*t[2]
					+ holder[index[i]][2][3]*t[3] + holder[index[i]][2][4])
			  - (holder[index[i]][1][1]*t[1] + holder[index[i]][1][2]*t[2]
					+ holder[index[i]][1][3]*t[3] + holder[index[i]][1][4])
			)
			)^2,
		i = 1:J}
	);
	status = solve(hm);

	return [getValue(t[1]), getValue(t[2]), getValue(t[3])];
end


###############################
####        Results         ###
###############################

# Read in data
# cd("/Users/eliotabrams/Desktop/Advanced\ Industrial\ Organization\ 2/Problem Set\ 3");
data = readtable("data.csv", separator = ',', header = true);
x = convert(Array, data[:x]);

# Create the estimates for the different beta and discretization sensitivities
vars = [],[],[],[],[],[];
csvfile = open("results_3.csv","w");
write(csvfile, 
	"Beta, Rounding, MPEC_theta1, MPEC_theta2, MPEC_RC, HM_theta1, HM_theta2, HM_RC");
for beta in [0, 0.5, 0.8, 0.9, 0.95]
	for round_param in [1]

		# Get main results
		x_discretized = round(x, round_param);
		p3, P, x_grid, actions, J = create_probabilities(x_discretized);
		results = [MPEC(J, x_discretized, x_grid, actions, p3, P, beta); 
				   HM(J, x_discretized, x_grid, actions, p3, P, beta)];
		for i in 1:6
			push!(vars[i], results[i]);
		end

		# Perform bootstrap
		for boot in 1:5
			x_boot = []
			for s in sample(1:1000:990000, 10)
				x_boot = [x_boot; x[s:(s+1000)]]
			end
			x_discretized = round(x_boot, round_param);
			p3, P, x_grid, actions, J = create_probabilities(x_discretized);
			results = [MPEC(J, x_discretized, x_grid, actions, p3, P, beta); 
				   HM(J, x_discretized, x_grid, actions, p3, P, beta)];
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




