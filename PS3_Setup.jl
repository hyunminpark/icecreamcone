###############################
####         Setup          ###
###############################

using Ipopt
using JuMP
using DataFrames
using StatsBase
EnableNLPResolve()
srand(1234)

data = readtable("data.csv", separator = ',', header = true);
x = convert(Array, data[:x]);

time_interval=50000;
starting_value = 1;
# Discretize the data
x_discretized = round(x[range(starting_value,time_interval)], 1);

x_grid = sort(unique(x_discretized))
actions = zeros(size(x_discretized,1),1);
for j = 1:size(x_discretized,1)-1
	actions[j] = x_discretized[j+1] < x[j];
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
				num2 +=(x_discretized[t+1] == j) & (x_discretized[t] == i);
				denom3 += (x_discretized[t] == i) & (actions[t] == a);
				denom2 += (x_discretized[t] == i);
				if j == minimum(x_grid)
					num += (x_discretized[t] == i) & (actions[t] == a);
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
