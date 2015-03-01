

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
  folds = torch.randperm(numSamples)
  local n = math.floor(numSamples/numModels)
  folds = torch.reshape(folds[{{1,numModels * n}}], n, numModels)--To make this clean, ignore the last #samples % #models samples
  print ('==>Creating '.. numModels..' folds each with '.. n ..' samples')
  return folds
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
