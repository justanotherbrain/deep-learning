require 'torch'
require 'nn'
require 'os'
function RemoveLastLayers(model, n)
  --This assumes we're using nn.Sequential as out base...fix it if you want
  if n == 0 then return model end
  ret = nn.Sequential()
  for i = 1,model:size()-n do
    ret:add(model:get(i):clone())
  end
  return ret
end
function CreateFolds(numFolds, numSamples)
  local folds = torch.randperm(numSamples)
  ret = {}
  if numFolds < 1 then
    --If < 1, treat this number as percent used for training
    local numTraining = math.floor(numFolds * numSamples)
    table.insert(ret,{
        training=folds[{{1,numTraining}}], 
        validation=folds[{{numTraining+1,-1}}]})
    print ('===>Splitting data into ' .. numTraining .. ' training samples and ' .. numSamples-numTraining .. ' validation samples.')
  elseif numFolds == 1 then
    table.insert(ret,{training=folds})
    print ('===>Training on all test data-no validation set.')
  else
    local n = math.floor(numSamples/numFolds)
    for i = 1,numFolds do
      local training
      local validation = 
      folds[{{(i-1)* n + 1, i * n}}]
      if i == 1 then
        training = folds[{{i * n + 1, -1}}]
      elseif i == numFolds then
        training = folds[{{1, (i-1) * n}}]
      else
        training = torch.cat(folds[{{1, (i-1) * n}}], folds[{{i* n + 1, -1}}])
      end
      table.insert(ret, {training=training, validation= validation})
    end
      print ('==>Creating '.. numFolds..' folds each with '.. n ..' training samples')
  end
  --To make this clean, ignore the last #samples % #models samples
  return ret
end
function LogModel(filename, model) 
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
end
function Test(model, testData, opt, confusion, indicies, kagglecsv)--add parameters
  if opt.type == 'cuda' then
    model = model:cuda()
  elseif opt.type == 'double' then
    model = model:double()
  end

  model:evaluate()
  ret = {}
  if kagglecsv ~= nil then
      require 'csvigo'
      table.insert(ret, {'Id','Category'})
  end
  err = 0
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
      --This is so ugly to properly handle ensemble learner output. For an ensemble, we take the highest output score from any model to be the "most sure" value, and select that as our answer.
      if  confusion ~= nil then confusion:add((guess[1] -1) % opt.noutputs + 1, target) end
      err = err + ((guess[1] ~= target) and 1 or 0)
      table.insert(ret, {indicies[t], guess[1]})
   end
   err = err / indicies:size(1)
   if kagglecsv ~= nil then 
       csvigo.save{data=ret, path=paths.concat(opt.save, kagglecsv)}
       os.execute('sed -i \'s/\\([0-9][0-9]*\\),\\([0-9][0-9]*\\)/\\1 , \\2/\' '.. paths.concat(opt.save, kagglecsv))
   end
   if opt.type == 'cuda' or opt.type == 'double' then
    model = model:float()
   end
   ret.err = err
  return ret
end
function LoadAndTest(opt, testData, modelName, kagglecsv)
  require 'nn'
  print ('==>Testing on test data, ' .. testData.size .. ' samples')
  local model = torch.load(paths.concat(opt.save, modelName))
  local testCM = optim.ConfusionMatrix(parameters.noutputs)
  local testResults = Test(model, testData, opt, testCM, nil, kagglecsv)
  local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
  print(testCM)
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
end
function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end
function append(from, to)
  for key, value in pairs(from) do
    to[key] = value
  end
end
function stringSplit(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end
function AddLSMtoConcatChildren(model)
  numModules = #model.modules
  for i = 1,numModules do
    numSubModules = #model.modules[i].modules
    if tostring(model.modules[i].modules[numSubModules]) ~= 'nn.LogSoftMax' then
      model.modules[i]:add(nn.LogSoftMax())
    end
  end
end
