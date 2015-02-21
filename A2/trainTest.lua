require 'torch'
require 'nn'
require 'optim'
require 'xlua'  
--These functions assume:
--all models use the same optimState 
--all models are of the same type, and differ by the parameters, or transformations to the data

--Must have:
--opt.eps-small value for determining if algorithm has converged 
--opt->enough information to create an initial state
--parameters.models
--parameters.outsize
--parameters.layersToRemove
--parameters.addSoftMax
--parameters.1..n, for each model
  
function CombineClassifiers(models, parameters)
  print '==>Combining classifiers'
  seq = nn.Sequential()
  conc = nn.Concat(1)--Assume this is the final layer of a classifier, without a SoftMax layer
  for i = 1,#models do
        conc:add(RemoveLastLayers(models[i], parameters.layersToRemove))
  end
  seq:add(conc)
  seq:add(nn.Reshape(#models, parameters.noutputs))
  if parameters.addSoftMax then 
    seq:add(nn.SoftMax())
  end
  seq:add(nn.Sum(1))
  print '==>Resultant classifier'
  print(seq)
  return seq
end
function TrainAndCompact(X, y, modelGen, parameters, opt, folds)
  models = {}
  for i=1,opt.models do
    if #parameters == 0 then models[i] = modelGen(parameters, opt) 
    else models[i] = modelGen(parameters[i], opt) end
  end
  print( '==>Training ' .. #models .. ' models.')
  folds = CreateFolds(opt.models, X:size(1), folds)--Reshapes, not creates
  opt_crit = OptimizerAndCriterion(opt)
  results = TrainModels(models, X, y, opt, opt_crit, parameters.noutputs, folds)
  return CombineClassifiers(models, parameters), results
end
function RemoveLastLayers(model, n)
  --This assumes we're using nn.Sequential as out base...fix it if you want
  if n == 0 then return model end
  ret = nn.Sequential()
  for i = 1,model:size()-n do
    ret:add(model:get(i):clone())
  end
  return ret
end

function CreateFolds(numModels, numSamples, folds)
  if folds == nil then
    folds = torch.randperm(numSamples)
  else
    if folds:size():size() == 1 then
      folds = torch.reshape(folds,folds:size()[1],1)
    end
    if folds:size(2) ~= 1 then
      print '==>Folds already in matrix form. Returning folds, assuming this is an appropriate fold matrix'
      return folds
    end
  end
  local n = math.floor(numSamples/numModels)
  folds = torch.reshape(folds[{{1,numModels * n}}], n, numModels)--To make this clean, ignore the last #samples % #models samples
  print ('==>Creating '.. numModels..' folds each with '.. n ..' samples')
  return folds
end

function TrainModels(models, X, y, opt, opt_crit, noutputs, folds)
  if #models ~= folds:size(2) then
    print 'Invalid number of folds/models'
    return
  end
  results = {}
  if folds:size():size() == 1 then
    folds = folds:reshape(folds:size()[1],1)
  end
  --Train individual models
  for foldIndex = 1,folds:size(2) do
    if folds:size(2) == 1 then
      trainInds = folds
      testInds = nil
    elseif opt.trainSetOnly == 1 then
      trainInds = folds[{{},foldIndex}]
      testInds = nil
    else 
      local numTraining = folds:size(1) * (folds:size(2) - 1)
      if foldIndex == 1 then 
        trainInds = torch.reshape(folds[{{},{2,-1}}], numTraining, 1)
      elseif foldIndex == folds:size(2) then
        trainInds = torch.reshape(folds[{{},{1,-2}}], numTraining,1)
      else
        trainInds = torch.reshape(torch.cat(folds[{{},{1,foldIndex-1}}],folds[{{},{foldIndex+1,-1}}]), numTraining,1)
      end
      testInds = folds[{{},{foldIndex}}]
    end  
    print ('==>Training model '.. foldIndex .. ' of ' .. folds:size(2))
    results[foldIndex] = TrainUntilConvergence(models[foldIndex], X, y, opt, opt_crit, noutputs, trainInds, testInds)
  end  
  return results
end
function TrainUntilConvergence(model, X, y, opt, opt_crit, noutputs, trainInds, testInds)
  local percentError = 1
  local bestPercentError = 1
  local bestEpoch = 1
  local epoch = 1
  local epochsLeft = opt.maxEpoch
  print (noutputs)
  trainConfusion= optim.ConfusionMatrix(noutputs)
  if testInds ~= nil then testConfusion = optim.ConfusionMatrix(noutputs) end
  while epochsLeft ~= 0 and percentError > 1e-3 do --If we have awesome performance, end early
      print ('===>Epoch ' .. epoch)
      print '====>Training'
      trainConfusion:zero()
      trainingResult = Train(model, X, y, opt, opt_crit, trainConfusion, trainInds)
      print ('====>Training error percentage: ' .. trainingResult.err)
      if testInds ~= nil then 
        testConfusion:zero()
        print '====>Testing'
        validationResult = Test(model, X, y, opt, testConfusion, testInds)
        print ('====>Validation error percentage: ' .. validationResult.err)
        percentError = validationResult.err 
        
      else 
        print '====>No Test Data'
        percentError = trainingResult.err
      end
      
      if bestPercentError > percentError then--If percent error goes down, update
        bestPercentError = percentError
        bestModel = model:clone()
        epochsLeft = opt.maxEpoch
      else
        epochsLeft = epochsLeft - 1
      end      
      
      epoch = epoch + 1
  end
  if testInds ~= nil then
    print '==>Final validation Confusion Matrix: '
    print (testConfusion)
  else
    print '==>Final training Confusion Matrix: '
    print (trainConfusion)
  end
  
  return {bestModel=bestModel, bestPercentError=bestPercentError, testConfusion=testConfusion, trainConfusion=trainConfusion}
end

function OptimizerAndCriterion(opt)
  --This only considers nll criterion as of now...so shoot me
  if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

  elseif opt.optimization == 'LBFGS' then
     optimState = {
        learningRate = opt.learningRate,
        maxIter = opt.maxIter,
        nCorrection = 10
     }
     optimMethod = optim.lbfgs

  elseif opt.optimization == 'SGD' then
     optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = 1e-7
     }
     optimMethod = optim.sgd

  elseif opt.optimization == 'ASGD' then
     optimState = {
        eta0 = opt.learningRate,
        t0 = trsize * opt.t0
     }
     optimMethod = optim.asgd

  else
     error('unknown optimization method')
  end
  if opt.loss == 'nll' then criterion = nn.ClassNLLCriterion() else error('nll only so far') end
  return {optimState=optimState, optimMethod=optimMethod, criterion=criterion}
end

function Train(model, X, y, opt, opt_crit, confusion, indicies)
  local optimMethod = opt_crit.optimMethod
  local optimState = opt_crit.optimState
  local criterion = opt_crit.criterion:clone()
  ret = {err=0}
  model:training()
  if indicies:size():size() == 2 then indicies = indicies:reshape(indicies:size(1)) end
  local parameters,gradParameters = model:getParameters()
  for t = 1,indicies:size(1),opt.batchSize do
    xlua.progress(t, indicies:size(1))
    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,indicies:size(1)) do
       -- load new sample
       local input = X[indicies[i]]
       local target = y[indicies[i]]
       if opt.type == 'double' then input = input:double()
       elseif opt.type == 'cuda' then input = input:cuda() end
       table.insert(inputs, {input, indicies[i]})--DG change, Inputs is now an array of tuples
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
                        local output = model:forward(inputs[i][1])
                        local err = criterion:forward(output, targets[i][1])
                        f = f + err

                        -- estimate df/dW
                        local df_do = criterion:backward(output, targets[i][1])
                        model:backward(inputs[i][1], df_do)
                        
                        -- log if samples are wrong. DG addition
                        local _, guess  = torch.max(output,1)
                        if confusion ~= nil then confusion:add(output, targets[i][1]) end
                        ret.err = ret.err + ((guess[1] ~= targets[i][1]) and 1 or 0)
                     end

                     -- normalize gradients and f(X)
                     gradParameters:div(#inputs)
                     f = f/#inputs

                     -- return f and df/dX
                     return f,gradParameters
                  end

    -- optimize on current mini-batch
    if optimMethod == optim.asgd then
       _,_ ,average = optimMethod(feval, parameters, optimState)
    else
       optimMethod(feval, parameters, optimState)
    end   
 end
 ret.err = ret.err/indicies:size(1)
 return ret
end
function Test(model, X, y, opt, parameters, confusion, indicies)--add parameters
  model:evaluate()
  ret = {err=0}
  if indicies == nil then 
    indicies = torch.range(1,X:size(1))
  elseif indicies:size():size() == 2 then
    indicies = indicies:reshape(indicies:size(1))
  end
  for t = 1,indicies:size(1) do
      -- get new sample
      local input = X[indicies[t]]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      if y:size():size() == 2 then
        target = y[indicies[t]][1]
      else
        target = y[indicies[t]]
      end
      
      local pred = model:forward(input)
      _, guess  = torch.max(pred,1)
      if  confusion ~= nil then confusion:add(output, target) end
      ret.err = ret.err + ((guess[1] ~= target) and 1 or 0)
   end
   ret.err = ret.err / indicies:size(1)
  return ret
end