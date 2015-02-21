require 'torch' 
require 'image'
require 'nn' 

print '==> loading, reshaping, and transforming color of data'
--mt = require 'fb.mattorch'
--Load Data
local test
local train
if true then
  train = torch.load('stl10_matlab/train.t7')
  test = torch.load('stl10_matlab/test.t7')
else
  train = torch.load('train.t7')
  test = torch.load('test.t7')
end
trsize = 5000
tesize = 8000
if not opt or opt.size == 'debug' then
  trsize = 2
  tesize = 2
end
traind = torch.reshape(train.X, torch.LongStorage{train.X:size(1), 3,96,96})[{{1,trsize},{},{},{}}]:float()
trainy = train.y[{{1,trsize},{}}]

testd = torch.reshape(test.X, torch.LongStorage{test.X:size(1), 3,96,96})[{{1,tesize},{},{},{}}]:float()
testy = test.y[{{1, tesize}, {}}]
trainData = {
   data = traind,
   labels = trainy,
   fold_indices = train.fold_indices,
   size = function() return trsize end
}
testData = {
   data = testd,
   labels = testy,
   size = function() return tesize end
}

print '===> Transform images'
if opt and opt.angle then angle = opt.angle else angle = math.pi/18 end
--Function for any general transformation to the data 
function transform(func)
    local traincp = traind:clone()
    for i = 1,traincp:size(1) do 
        traincp[i] = func.func(traincp[i], func.params)
    end
    trainData.data = torch.cat(trainData.data, traincp,1)
    trainData.labels = torch.cat(trainData.labels,trainy,1)
    trsize = trsize + traincp:size(1)
end
--Transformation functions
function rotate(slice, angle) 
    return image.rotate(slice, angle)
end
function hflip(slice, n_a)
    return image.hflip(slice)
end
--Rotate
if angle then
    transform({func=rotate, params=angle})
    transform({func=rotate, params=-angle})
end
--Flip
if not opt or opt.hflip == 1 then
    transform({func=hflip})
end
--Preprocess
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
torch.save('testData.t7', testData)
torch.save('trainData.t7', trainData)

