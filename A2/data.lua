require 'torch' 
require 'image'
require 'nn' 

print '==> loading, reshaping, and transforming color of data'
--mt = require 'fb.mattorch'
--Load Data
--train = mt.load('stl10_matlab/trainX_y')
--test = mt.load('stl10_matlab/testX_y')
local train = torch.load('stl10_matlab/train.t7')
local test = torch.load('stl10_matlab/test.t7')
trsize = 5000
tesize = 8000
if not opt or opt.size == 'debug' then
  trsize = 5
  tesize = 1
end
if opt and opt.angle then angle = opt.angle else angle = math.pi/18 end

local traind = torch.reshape(train.X, torch.LongStorage{train.X:size(1), 3,96,96})[{{1,trsize},{},{},{}}]:float()
local testd = torch.reshape(test.X, torch.LongStorage{test.X:size(1), 3,96,96})[{{1,tesize},{},{},{}}]:float()
--Reshape images
trainData = {
   data = torch.cat(torch.cat(traind,traind,1),torch.cat(traind,traind,1),1),
   labels = torch.cat(torch.cat(train.y,train.y,1),torch.cat(train.y,train.y,1),1),
   size = function() return trsize end
}
testData = {
   data = torch.cat(torch.cat(testd,testd,1),torch.cat(testd,testd,1),1),
   labels = torch.cat(torch.cat(test.y,test.y,1),torch.cat(test.y,test.y,1),1),
   size = function() return tesize end
}
trsize = trsize * 4
tesize = tesize * 4
--Preprocess

--Rotate 10 degrees each way
for i = 1,trainData:size()/4 do
  trainData.data[i + trainData:size()/4] = image.rotate(trainData.data[i],angle)
  trainData.data[i + 2 * trainData:size()/4] = image.rotate(trainData.data[i],-angle)
  trainData.data[i + 3 * trainData:size()/4] = image.hflip(trainData.data[i])
  
end
for i = 1,testData:size()/4 do
   testData.data[i + testData:size()/4] = image.rotate(testData.data[i],angle)
   testData.data[i + 2 * testData:size()/4] =image.rotate(testData.data[i],-angle)
   testData.data[i + 3 * testData:size()/4] = image.hflip(testData.data[i])
   
end
  
--Convert colors
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end
-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end
-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size(1) do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size(1) do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end
--torch.save('test.torch', testData)
--torch.save('train.torch', trainData)

