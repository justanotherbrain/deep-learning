function Train(model, trainData, opt, opt_crit, confusion, indicies)
  local optimMethod = opt_crit.optimMethod
  local optimState = opt_crit.optimState
  local criterion = opt_crit.criterion:clone()
  ret = {err=0}
  model:training()
  if indicies:size():size() == 2 then indicies = indicies:reshape(indicies:size(1)) end
  local parameters,gradParameters = model:getParameters()
  for t = 1,indicies:size(1),opt.batchSize do
    --xlua.progress(t, indicies:size(1))
    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,indicies:size(1)) do
       -- load new sample
       local input = trainData.X[indicies[i]]
       local target = trainData.y[indicies[i]]
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