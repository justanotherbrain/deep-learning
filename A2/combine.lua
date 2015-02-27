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
  if opt.models ~= #model_optim_critList then
    print 'WRONG NUMBER OF MODELS'
    return
  end
  local Train = trainFun
  --Create folds as needed
  if type(folds) == 'table'
    if #folds ~= #model_optim_critList then
      print 'WRONG NUMBER OF FOLDS'
      return
    end
  elseif type(folds) == 'number' then
    if folds >= 1 and folds ~= #model_optim_critList then
      print 'WRONG NUMBER OF FOLDS'
      return
    end
    folds = CreateFolds(folds, trainData.size) 
  else
    print 'INVALID FOLD DATA TYPE'
    return
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
      --Train logic
      if not modelResults[foldIndex].finished then 
        --Train model
        logpackage.trainConfusion:zero()
        local trainingResult = Train(model_optim_critList[foldIndex], trainData, opt, logpackage.trainConfusion, folds[foldIndex].training)
        print ('===>Training error percentage: ' .. trainingResult.err)
        --Test on validation
        if folds[foldIndex].validation ~= nil then 
          logpackage.testConfusion:zero()
          print '===>Testing'
          validationResult = Test(model_optim_critList[foldIndex], trainData, opt, logpackage.testConfusion, folds[foldIndex].validation)
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
  --Setup more variables
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
