	function MPEC(J, x_discretized, x_grid, actions, p3, beta)

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
			log( exp( -RC + beta*EV_1[index[t]] ) / 
				( exp( -RC + beta*EV_1[index[t]] )  + exp(-theta[1]*x_discretized[t] - theta[2]*(x_discretized[t])^2 + beta*EV_0[index[t]] ) ) ) +
		 	log( exp(-theta[1]*x_discretized[t] - theta[2]*(x_discretized[t])^2 + beta*EV_0[index[t]] )  / 
		 		( exp( -RC + beta*EV_1[index[t]] )  +  exp(-theta[1]*x_discretized[t] - theta[2]*(x_discretized[t])^2 + beta*EV_0[index[t]] ) ) )  +
		 	log( p3[index[t], index[t-1], actions[t-1] + 1] ) ,t = 2:T} ) ;

		status = solve(m);

		theta1, theta2, RC_opt = (getValue(theta[1]), getValue(theta[2]), getValue(RC));
		return [theta1; theta2; RC_opt];
	end 
