#Assignment 1#
##Monkey Business##

##long running experiments

  Saves each session into the experiments directory
	Sample structure:	
	

	/experiments
		/march_23-Results
			params.log
			test.log
			train.log
			model.net
			top_model.net

   - top_model.net is the neural network that achieved the best accuracy that session  
   - note that you still must manually specify how many epochs each session should run for, otherwise it will go forever

## visualizations

    we run this as follows

        python visualize.py AVGS  // to see the average performance recorded in each session

        python visualize.py 0     // to see the performance of each letter for the 1st session (0 indexed)

        python visualize.py 2 6   // to see the performance of each letter for the 3rd thru 6th session (0 indexed)

