require 'torch'
require 'nn'
require 'optim'
require 'xlua'  
dofile 'helpers.lua'
--These functions assume:
--all models use the same optimState 
--all models are of the same type, and differ by the parameters, or transformations to the data
  
--TODO:
--Move X,y for functions into trainData struct again-done
--Re-implement trainData:size()-done
--In model, add ModifyModel (and ModifyCombinedModel?), make them both generic, appending simply to a generic model?
--Add global option for maximum number of epochs to use ever
--Move model, test_train, and OptimAndCriterion to one file
--have CreateModel() output {model=model, criterion=criterion}-done and more
--add LoadOptions(file)

--Semi-supervised pseudocode:
  --combined = TrainAndCompact(data)
  --Switch out old model and old test/train (and old criterion?)
  --LoadOptions()
  --ModifyCombinedModel(combined)
  --TrainAndCompact(data, combined)
  --Finally test on data


function CreateModels(opt, parameters, modelGen, model_optim_critList)
  --Setup
    parameterList = {}
    if #parameters == 0 then
      for i =1,opt.models do
        table.insert(parameterList, parameters)
      end    
    else
      parameterList = parameters
    end
    --Creating models
    if model_optim_critList == nil then
      model_optim_critList = {}
      for i = 1,opt.models do
        table.insert(model_optim_critList, modelGen(parameterList[i], opt))
      end
    else
      for i = 1,opt.models do
        table.insert(model_optim_critList, modelGen(parameterList[i], opt, model_optim_critList[i]))
      end
    end
    return model_optim_critList
end
--Minor note: The absolute minimum number of epochs is opt.maxEpoch+1
function TrainModels(model_optim_critList, opt, trainData, trainFun, folds, logpackages)
  --Setup and invalid data checking
  local Train = trainFun
  if folds ~= nil and #folds ~= #model_optim_critList then
    print 'WRONG NUMBER OF FOLDS'
    return
  elseif opt.models ~= #model_optim_critList then
    print 'WRONG NUMBER OF MODELS'
    return
  end
  --Create folds if necessary
  if folds == nil then 
    folds = CreateFolds(#model_optim_critList, trainData.size) 
  end
  --Setup internals
  local modelResults = {}
  for i=1,#model_optim_critList do 
    table.insert(modelResults, {bestPercentError=1.1, epochsLeft=opt.maxEpoch, finished= false, model=nil}) 
  end
  local trainLoop = 
    function(foldIndex) 
      print('===>Training')
      if logpackages ~= nil then logpackage = logpackages[foldIndex] end
      --Get inidices
      if opt.trainSetOnly == 1 or opt.models == 1 then
        trainInds = folds[{{},foldIndex}]
        testInds = nil
      else--for normal, training on multiple subsets
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
      --Train logic
      if not modelResults[foldIndex].finished then 
        --Train model
        logpackage.trainConfusion:zero()
        local trainingResult = Train(model_optim_critList[foldIndex], trainData, opt, logpackage.trainConfusion, trainInds)
        print ('===>Training error percentage: ' .. trainingResult.err)
        --Test on validation
        if testInds ~= nil then 
          logpackage.testConfusion:zero()
          print '===>Testing'
          validationResult = Test(model_optim_critList[foldIndex], trainData, opt, logpackage.testConfusion, testInds)
          print ('===>Validation error percentage: ' .. validationResult.err)
          percentError = validationResult.err 
        else 
          --If we don't have a validation set
          print '===>No Test Data'
          percentError = trainingResult.err
        end
        --Update
        if modelResults[foldIndex].bestPercentError > percentError then--If percent error goes down, update
          print('===>Updating best model')
          modelResults[foldIndex].bestPercentError = percentError
          modelResults[foldIndex].epochsLeft = opt.maxEpoch + 1
          modelResults[foldIndex].model = model_optim_critList[foldIndex].model:clone()
        end      
        modelResults[foldIndex].epochsLeft = modelResults[foldIndex].epochsLeft -1
        logpackage:log()--Log iteration
        
        --Convergence conditions
        if modelResults[foldIndex].epochsLeft == 0 or modelResults[foldIndex].bestPercentError < 1e-3 then
          modelResults[foldIndex].finished = true
          return 1 --Return 1 when the model finishes
        end
      end  
      return 0
    end
  local epoch = 1
  local foldIndex = 0
  local numberConverged = 0
  local conc
  --Loop until all models converge
  while numberConverged ~= #model_optim_critList do
    foldIndex = (foldIndex % #model_optim_critList) + 1
    print('\n===>Training model ' .. foldIndex .. '\n')
    numberConverged = numberConverged + trainLoop(foldIndex)
    epoch = epoch + 1
    --Save a combined model every epoch
    if foldIndex == #model_optim_critList then 
      if opt.models ~= 1 then 
        conc = nn.Concat(1)
        for i = 1,opt.models do
          conc:add(modelResults[i].model)
          LogModel(opt.save .. '-combined_model.net', conc)
        end
      else
        conc = modelResults[1].model
      end
      
    end
  end
  print ('Completed training, took ' .. epoch/opt.models .. ' epochs.')
  return conc
end
