
-- default is 4000 classes per representation model



function create_network()
 	n_outputs = 4000

	n_feats = 3
	width = 36
	height = 36
	n_inputs = n_feats * width * height

	n_states = {64,128,256,512}
	filter_size = 5
	pool_size = 2

	model = nn.Sequential()
	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
  	model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
  	model:add(nn.ReLU())
  	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

  	-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
  	model:add(nn.ReLU())
  	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

  	-- stage 3 : filter bank -> squashing -> L2 pooling -> normalization
  	model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize, filtsize))
  	model:add(nn.ReLU())
  	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

  	-- stage 3 : standard 2-layer neural network
  	model:add(nn.View(nstates[3]*filtsize*filtsize))
  	model:add(nn.Dropout(0.5))
  	model:add(nn.Linear(nstates[3]*filtsize*filtsize, nstates[4]))
  	model:add(nn.ReLU())
  	model:add(nn.Linear(nstates[3], noutputs))

  	print '==> here is the model:'
	print(model)
	return model
 end 

function train_model(model, data, labels)
	epoch = epoch or 1
	local time = sys.clock()
	model:training()
	shuffle = torch.randperm(data:size())

	local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end
end