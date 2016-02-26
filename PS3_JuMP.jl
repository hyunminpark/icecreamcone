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
EnableNLPResolve()
cd("/Users/eliotabrams/Desktop/Advanced\ Industrial\ Organization\ 2/Problem Set\ 3")
srand(1234)


###############################
####       Functions        ###
###############################

function create_estimates(x, round_param, beta, time_interval)
	# Get range
	starting_value = rand(1:size(x,1) - time_interval);

	# Discretize the data
	x_discretized = round(x[range(starting_value,time_interval)], round_param);
	x_grid = sort(unique(x_discretized))
	actions = zeros(size(x_discretized,1),1);
	for j = 1:size(x_discretized,1)-1
		actions[j] = x_discretized[j+1] < x[j];
	end

	# State transition probabilities
	J = size(x_grid,1);
	p3 = zeros(J,J,2);
	p2 = zeros(J,J);
	for i in 1:J
		for j in 1:J
			for a in 1:2
				num3, denom3 = 0, 0;
				num2, denom2 = 0, 0;
				for t in 1:size(x_discretized,1)-1
					num3 += (x_grid[j] == x_discretized[t+1]) & (x_discretized[t] == x_grid[i]) & (actions[t] == a-1);
					num2 +=(x_grid[j] == x_discretized[t+1]) & (x_discretized[t] == x_grid[i]);
					denom3 += (x_discretized[t] == x_grid[i]) & (actions[t] == a-1);
					denom2 += (x_discretized[t] == x_grid[i]);
				end
				p3[j,i,a] = max(num3 / denom3, 0);
				p2[j,i] = max(num2 / denom2, 0);
				if p3[j,i,a] == 0
					p[j,i,a] = 1.0e-8; 
				end
				if p2[j,i] == 0
					p[j,i] = 1.0e-8; 
				end
			end
		end
	end

	# Action choice probabilities
	P = Dict()
	for i in x_grid
		for a in [0, 1]
			num, denom = 0, 0;
			for t in 1:size(x_discretized,1)
				num += (actions[t] == a) & (i == x_discretized[t]);
				denom += (i == x_discretized[t]);
			end
			P[(a,i)] = num / denom;
		end
	end

	return [hotz_miller(x_discretized, x_grid, actions, p3, p2, P, beta); 
			MPEC(x_discretized, x_grid, actions, p3, p2, P, beta)]
end

function MPEC(x, x_grid, actions, p3, p2, P, beta)
	ux = x_grid;
	J = size(ux,1);
	T = size(x,1);

	p = zeros(J,J,2); # JuMP does not match disctionary properly
	for i = 1:J
		for j = 1:J 
			for a = 1:2
				p[j,i,a] = p3[(ux[j],ux[i],a-1)]; # If = 0, add 10^(-8)
				if p[j,i,a] == 0
					p[j,i,a] = 1.0e-8; 
				end 
			end 
		end
	end

	index = []; # define index to sum EV's over time.
	for t = 1:T
		aux = find(ux -> ux == x[t], ux);
		index = [index;aux]
	end 

	m = Model(solver = IpoptSolver() );
	@defVar(m, EV_1[1:J]); #defined over the unique values of x
	@defVar(m, EV_0[1:J]);
	@defVar(m, theta[1:2]);
	@defVar(m, RC >= 0);

	@addNLConstraint(m, constr[i=1:J], EV_1[i] == sum{ log( exp( -RC + beta*EV_1[j] )  + exp(-theta[1]*ux[j] - theta[2]*(ux[j])^2 + beta*EV_0[j] ) ) * p[j,i,2] , j = 1:J} );
	@addNLConstraint(m, constr[i=1:J], EV_0[i] == sum{ log( exp( -RC + beta*EV_1[j] )  + exp(-theta[1]*ux[j] - theta[2]*(ux[j])^2 + beta*EV_0[j] ) ) * p[j,i,1] , j = 1:J} );

	@setNLObjective(m, Max, sum{ 
		log( exp( -RC + beta*EV_1[index[t]] ) / 
			( exp( -RC + beta*EV_1[index[t]] )  + exp(-theta[1]*x[t] - theta[2]*(x[t])^2 + beta*EV_0[index[t]] ) ) ) +
	 	log( exp(-theta[1]*x[t] - theta[2]*(x[t])^2 + beta*EV_0[index[t]] )  / 
	 		( exp( -RC + beta*EV_1[index[t]] )  +  exp(-theta[1]*x[t] - theta[2]*(x[t])^2 + beta*EV_0[index[t]] ) ) )  +
	 	log( p[index[t], index[t-1], actions[t-1] + 1] ) ,t = 2:T} ) ;

	status = solve(m);


	return [getValue(theta[1]), getValue(theta[2]), getValue(RC)];
end 

function hotz_miller(x_discretized, x_grid, actions, p3, p2, P, beta)
	theta1, theta2, RC = (0, 1, 2);
	Ns = 1000; # Length of Simulated path
	J=size(x_grid,1)
	y   = zeros(J)
	z0  = zeros(J)
	z1  = zeros(J)
	z2  = zeros(J)
	zrc = zeros(J)

for i in x_grid
	# Simulate paths	
	Xs = rand(Ns,2)
	Xs[1] = 
	for ns in 2:Ns
		Xs[ns]=
	end
	# For the Objective Function
	y[(i)]=log(P[(i,1)]/P[(i,0)])
	z0[(i)]=beta^t*
end
	return [theta1, theta2, RC]
end


###############################
####        Results         ###
###############################

# Read in data
data = readtable("data.csv", separator = ',', header = true);
x = convert(Array, data[:x]);

# Create the estimates for the different beta and discretization sensitivities
vars = [],[],[],[],[],[]
csvfile = open("results.csv","w")
write(csvfile, "Beta, Rounding, HM_theta1, HM_theta2, HM_RC, MPEC_theta1, MPEC_theta2, MPEC_RC")
# [0, 0.5, 0.8, 0.9, 0.95]
# [0, 1]
for beta in [0, 0.5]
	for round_param in [1]

		# Perform bootstrap
		for boot in 1:5
			results = create_estimates(x, round_param, beta, 5000)
			for i in 1:6
				push!(vars[i], results[i])
			end
		end

		# Print results
		write(csvfile, string("\n", float(beta), ",", round_param, ","))
		for i in 1:6
			write(csvfile, string(vars[i][1], " (", std(convert(Array{Float64}, vars[i])), "),"))
		end
		vars = [],[],[],[],[],[]
		
	end
end
close(csvfile)




