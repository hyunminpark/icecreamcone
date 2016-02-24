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
EnableNLPResolve()
srand(1234)


###############################
####         Data           ###
###############################

data = readtable("data.csv", separator = ',', header = true);
x = convert(Array, data[:x]);


###############################
####       Functions        ###
###############################

function create_estimates(x, round_param, beta, time_interval)
	# Get range
	starting_value = rand(1:size(x,1) - time_interval);

	# Discretize the data
	x_discretized = round(x[range(starting_value,time_interval)], round_param)
	actions = zeros(size(x_discretized,1),1);
	for j = 1:size(x_discretized,1)-1
		actions[j] = x_discretized[j+1] < x[j];
	end

	# State transition probabilities
	p3 = Dict()
	for i in 
		for j in unique(x_discretized)
			for a in [0, 1]
				count = 0;
				for t in 1:size(x_discretized,1)-1
					count += (j == x_discretized[t+1]) & (x_discretized[t] == i) & (actions[t] == a);
				end
				p3[(j,i,a)] = count / size(x_discretized,1);
			end
		end
	end

	# Action choice probabilities
	P = Dict()
	for i in unique(x_discretized)
		for a in [0, 1]
			count = 0;
			for t in 1:size(x_discretized,1)
				count += (actions[t] == a) & (i == x_discretized[t]);
			end
			P[(i,a)] = count / size(x_discretized,1);
		end
	end

	return [hotz_miller(x_discretized, p3, P, beta), MPEC(x_discretized, p3, P, beta)]
end

function hotz_miller(x_discretized, p3, P, beta)
end

function MPEC(x_discretized, p3, P, beta)
end


###############################
####        Calls           ###
###############################

"HERE WE LOOP THROUGH THE CALLS"

results = create_estimates(x, 0, 2.0, 10000);


