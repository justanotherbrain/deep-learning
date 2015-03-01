require 'torch'
function RemoveLastLayers(model, n)
  --This assumes we're using nn.Sequential as out base...fix it if you want
  if n == 0 then return model end
  ret = nn.Sequential()
  for i = 1,model:size()-n do
    ret:add(model:get(i):clone())
  end
  return ret
end
function CreateFolds(numModels, numSamples)
  local folds = torch.randperm(numSamples)
  ret = {}
  if numModels < 1 then
    --If < 1, treat this number as percent used for training
    local numTraining = math.floor(numModels * numSamples)
    table.insert(ret,{
        training=folds[{{1,numTraining}}], 
        validation=folds[{{numTraining+1,-1}}]})
    print ('===>Splitting data into ' .. numTraining .. ' training samples and ' .. numSamples-numTraining .. ' validation samples.')
  elseif numModels == 1 then
    table.insert(ret,{training=folds})
    print ('===>Training on all test data-no validation set.')
  else
    local n = math.floor(numSamples/numModels)
    for i = 1,numModels do
      local training
      local validation = 
      folds[{{(i-1)* n + 1, i * n}}]
      if i == 1 then
        training = folds[{{i * n + 1, -1}}]
      elseif i == numModels then
        training = folds[{{1, (i-1) * n}}]
      else
        training = torch.cat(folds[{{1, (i-1) * n}}], folds[{{i* n + 1, -1}}])
      end
      table.insert(ret, {training=training, validation= validation})
    end
      print ('==>Creating '.. numModels..' folds each with '.. n ..' training samples')
  end
  --To make this clean, ignore the last #samples % #models samples
  return ret
end
function LogModel(filename, model) 
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
end
function Test(model, testData, opt, confusion, indicies)--add parameters
  model:evaluate()
  ret = {err=0}
  if indicies == nil then 
    indicies = torch.range(1,testData.size)
  elseif indicies:size():size() == 2 then
    indicies = indicies:reshape(indicies:size(1))
  end
  for t = 1,indicies:size(1) do
      -- get new sample
      local input = testData.X[indicies[t]]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      if testData.y:size():size() == 2 then
        target = testData.y[indicies[t]][1]
      else
        target = testData.y[indicies[t]]
      end
      
      local pred = model:forward(input)
      _, guess  = torch.max(pred,1)
      if  confusion ~= nil then confusion:add(pred, target) end
      ret.err = ret.err + ((guess[1] ~= target) and 1 or 0)
   end
   ret.err = ret.err / indicies:size(1)
  return ret
end
function LoadAndTest(opt, testData, modelName)
  require 'nn'
  print '==>Testing on test data'
  local model = torch.load(paths.concat(opt.save, modelName))
  local testCM = optim.ConfusionMatrix(parameters.noutputs)
  local testResults = Test(model, testData, opt, testCM)
  local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
  testLogger:add{['% mean class accuracy (test set)'] = testCM.totalValid * 100,
        ['1'] = testCM.valids[1],
        ['2'] = testCM.valids[2],
        ['3'] = testCM.valids[3],
        ['4'] = testCM.valids[4],
        ['5'] = testCM.valids[5],
        ['6'] = testCM.valids[6],
        ['7'] = testCM.valids[7],
        ['8'] = testCM.valids[8],
        ['9'] = testCM.valids[9],
        ['0'] = testCM.valids[10]
      }
 print(testCM)
end
