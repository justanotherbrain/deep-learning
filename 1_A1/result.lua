require 'torch'   
require 'optim' 
require 'nn'
require 'csvigo'

function setup()
  local loaded = torch.load('test_32x32.t7','ascii')
  testData = {
     data = loaded.X:transpose(3,4),
     labels = loaded.y[1],
     size = function() return 26032 end
  }
  model = torch.load('model.net')
  results = {{'samp#','target', 'pred'}}
end

function test()
   model:evaluate()
   for t = 1,testData:size() do
      local input = testData.data[t]:double()
      local target = testData.labels[t]
      local pred = model:forward(input)
      _, guess  = torch.max(pred,1)
      table.insert(results, {t, target, guess[1]})
   end   
end
setup()
test()
csvigo.save{data = results, path='predictions.csv'}
