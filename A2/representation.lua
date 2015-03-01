
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
   parameters,gradParameters = model:getParameters()
	-- epoch tracker
   epoch = epoch or 1
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd
   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
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
       optimMethod(feval, parameters, optimState)

       time = sys.clock() - time
	   time = time / trainData:size()
	   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	   -- print confusion matrix
	   print(confusion)

	   local filename = paths.concat(opt.save, 'model.net')
	   os.execute('mkdir -p ' .. sys.dirname(filename))
	   print('==> saving model to '..filename)
	   torch.save(filename, model)

	   -- next epoch
	   confusion:zero()
	   epoch = epoch + 1
end