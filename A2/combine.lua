require 'torch'
require 'nn'
require 'optim'
require 'xlua'  
dofile 'train_test.lua'
dofile 'helpers.lua'
--These functions assume:
--all models use the same optimState 
--all models are of the same type, and differ by the parameters, or transformations to the data
  
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
  logpackages = CreateLogPackages(opt, parameters)
  results = {converged = 0}
  epoch = 1
  while results.converged ~= #models do
    print ('==>Epoch ' .. epoch)
    TrainModels(models, X, y, opt, opt_crit, logpackages, results, folds)
    combined = CombineClassifiers(models, parameters)
    logpackages.logmodel(paths.concat(opt.save, 'model-combined.net'), combined)
    epoch = epoch + 1
  end
  print '==>Resultant classifier'
  print(combined)
  return combined, results
end

function TrainModels(models, X, y, opt, opt_crit, logpackages, results, folds)
  if #models ~= folds:size(2) then
    print 'Invalid number of folds/models'
    return
  end
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
    if results[foldIndex] == nil then results[foldIndex] = {} end
    convergedThisIteration = TrainUntilConvergence(models[foldIndex], X, y, opt, opt_crit, logpackages[foldIndex], trainInds, testInds, results[foldIndex])
    if convergedThisIteration then results.converged = results.converged + 1 end
  end  
end
function TrainUntilConvergence(model, X, y, opt, opt_crit, logpackage, trainInds, testInds, modelResults)
  if next(modelResults) == nil then
    modelResults.bestPercentError = 1
    modelResults.epochsLeft = opt.maxEpoch
  end
  if modelResults.epochsLeft > 0 then --If we have awesome performance, end early
    logpackage.trainConfusion:zero()
    trainingResult = Train(model, X, y, opt, opt_crit, logpackage.trainConfusion, trainInds)
    print ('====>Training error percentage: ' .. trainingResult.err)
    if testInds ~= nil then 
      logpackage.testConfusion:zero()
      print '====>Testing'
      validationResult = Test(model, X, y, opt, logpackage.testConfusion, testInds)
      print ('====>Validation error percentage: ' .. validationResult.err)
      percentError = validationResult.err 
      
    else 
      print '====>No Test Data'
      percentError = trainingResult.err
    end
    if modelResults.bestPercentError > percentError then--If percent error goes down, update
      modelResults.bestPercentError = percentError
      modelResults.epochsLeft = opt.maxEpoch + 1
    end      
    logpackage:log()
    modelResults.epochsLeft = modelResults.epochsLeft -1
    --Convergence conditions
    if modelResults.epochsLeft == 0 or modelResults.bestPercentError < 1e-3 then
      modelResults.epochsLeft = 0
      return true 
    end
  end  
  return false
end
