require 'nn'
require 'torch'
require 'cunn'
require 'math'
require 'image'
require 'optim'
require 'xlua'


-- default is 4000 classes per representation model



function create_network( classes_count, input_size )
 	
	-- this function creates a network for the surrogate task
	-- the network generated when cuda is available is a much more complicated architecture

	input_size = input_size or 36
 	n_outputs = classes_count
	n_feats = 3
	width = input_size
	height = input_size
	n_inputs = n_feats * width * height

	filter_size = 5
	pool_size = 2

	if opt.type == 'cuda' then

		nstates = {64,128,256,512}

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

	  	-- stage 4 : standard 2-layer neural network
	  	model:add(nn.View(nstates[3]*filtsize*filtsize))
	  	model:add(nn.Dropout(0.5))
	  	model:add(nn.Linear(nstates[3]*filtsize*filtsize, nstates[4]))
	  	model:add(nn.ReLU())
	  	model:add(nn.Linear(nstates[4], noutputs))
		model:add(nn.LogSoftMax())
		criterion = nn.ClassNLLCriterion()
	else
	  n_states = {64,64,128}
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on this dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      dim = ((width - filtsize + 1)/poolsize - filtsize + 1)/poolsize
      model:add(nn.Reshape(nstates[2]*dim*dim))
      model:add(nn.Linear(nstates[2]*dim*dim, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))
   end
 	print '==> here is the model:'
	print(model)
	return model,criterion
 end 

function train_model(model, criterion, trainData)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(data:size())

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

                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

function test( model, testData )
   
   -- local vars
   local time = sys.clock()

   -- top score to save corresponding to saved model
   top_score = top_score or 0.1


   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      _, guess  = torch.max(pred,1)
      -- print("\n" .. target .. "\n")
      testSampleWrong[t]= testSampleWrong[t] + ((guess[1] ~= target) and 1 or 0)--DG addition
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- update log/plot ... along with accuracy scores for each digit

   -- here we check to see if the current model is the best yet, and if so, save it
   if top_score < confusion.totalValid * 100 then
      local top_filename = paths.concat(opt.save, 'winning_model.net')
      os.execute('mkdir -p ' .. sys.dirname(top_filename))
      print('==> saving new top model to '..top_filename)
      torch.save(top_filename, model)
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end   

end


function get_feature_model(model, n)
    
    -- return last layers of model

    if n == 0 then return model end
    ret = nn.Sequential()
    for i = 1,model:size()-n do
        ret:add(model:get(i):clone())
    end
    return ret
end 

function test(model, testData)
   
   -- local vars
   local time = sys.clock()

   -- top score to save corresponding to saved model
   top_score = top_score or 0.1


   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      _, guess  = torch.max(pred,1)
      -- print("\n" .. target .. "\n")
      testSampleWrong[t]= testSampleWrong[t] + ((guess[1] ~= target) and 1 or 0)--DG addition
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- update log/plot ... along with accuracy scores for each digit
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- DG addition. Print updated list of mislabeled samples every few iterations
   testWrite = {}
   for key, value in pairs(testSampleWrong) do
      table.insert(testWrite, {key,value})
   end
      table.insert(testWrite, {'Epoch',#testWrite+1})    
   table.sort(testWrite, sampleComparer)

   testWrite[1][2] = epoch - 1
   
   csvigo.save{path=paths.concat(opt.save, 'test_wrongSamples.log'), data=testWrite}
   
end

