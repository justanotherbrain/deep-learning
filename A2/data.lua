require 'torch' 
require 'image'
require 'nn' 

print '==> loading, reshaping, and transforming color of data'

--Load Data
function ReadFiles(opt)
  -- Open the files and set little endian encoding
  local data_fd = torch.DiskFile("stl10_binary/train_X.bin", "r", true)
  data_fd:binary():littleEndianEncoding()
  local label_fd = torch.DiskFile("stl10_binary/train_y.bin", "r", true)
  label_fd:binary():littleEndianEncoding()

  -- Create and read the data
  local traind = torch.ByteTensor(5000, 3, 96, 96)
  data_fd:readByte(traind:storage())
  local trainy = torch.ByteTensor(5000,1)
  label_fd:readByte(trainy:storage())

  -- Because data is in column-major, transposing the last 2 dimensions gives result that can be correctly visualized
  --traind = traind:transpose(3, 4)

  -- Open the files and set little endian encoding
  local data_fd2 = torch.DiskFile("stl10_binary/test_X.bin", "r", true)
  data_fd2:binary():littleEndianEncoding()
  local label_fd2 = torch.DiskFile("stl10_binary/test_y.bin", "r", true)
  label_fd2:binary():littleEndianEncoding()

  -- Create and read the data
  local testd = torch.ByteTensor(8000, 3, 96, 96)
  data_fd2:readByte(testd:storage())
  local testy = torch.ByteTensor(8000,1)
  label_fd2:readByte(testy:storage())

  -- Because data is in column-major, transposing the last 2 dimensions gives result that can be correctly visualized
  --testd = testd:transpose(3, 4)

  if not opt or opt.size == 'debug' then
    trsize = 20
    tesize = 2
    print '==>Debugging, using ultra reduced dataset'
  else
    trsize = traind:size(1)
    tesize = testd:size(1)
    print ('==>Using full data set Train:' .. trsize .. ' Test:' .. tesize)
  end
  
  traind = torch.reshape(traind, torch.LongStorage{traind:size(1), 3,96,96})[{{1,trsize},{},{},{}}]:float()
  trainy = trainy[{{1,trsize},{}}]

  testd = torch.reshape(testd, torch.LongStorage{testd:size(1), 3,96,96})[{{1,tesize},{},{},{}}]:float()
  testy = testy[{{1, tesize}, {}}]
  
  trainData = {
     X = traind,
     y = trainy,
     size = traind:size(1)
  }
  testData = {
     X = testd,
     y = testy,
     size = testd:size(1)
  }
  return trainData, testData
end


--Apply transformations to grow your test set
function TransformImages(trainData, opt)
  print '===> Transform images'
  transforms = {}
  --Rotate
  if opt.angle then
      table.insert(transforms, transform({func=rotate, params=opt.angle}, trainData.X))
      table.insert(transforms, transform({func=rotate, params=-opt.angle}, trainData.X))
  end
  --Flip
  if opt.hflip == 1 then
      table.insert(transforms, transform({func=hflip}, trainData.X))
  end
  local newX = trainData.X
  local newY = trainData.y
  for i=1,#transforms do
      newX = newX.cat(newX, transforms[i],1)
      newY = newY.cat(newY, trainData.y, 1)
  end
  trainData.size = newX:size(1)
  trainData.X = newX
  trainData.y = newY
end
--Preprocess all images
function Preprocess(trainData, testData, opt)
  --Preprocess
  for i = 1,trainData.size do
     trainData.X[i] = image.rgb2yuv(trainData.X[i])
  end
  for i = 1,testData.size do
     testData.X[i] = image.rgb2yuv(testData.X[i])
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
     mean[i] = trainData.X[{ {},i,{},{} }]:mean()
     std[i] = trainData.X[{ {},i,{},{} }]:std()
     trainData.X[{ {},i,{},{} }]:add(-mean[i])
     trainData.X[{ {},i,{},{} }]:div(std[i])
  end

  -- Normalize test data, using the training means/stds
  for i,channel in ipairs(channels) do
     -- normalize each channel globally:
     testData.X[{ {},i,{},{} }]:add(-mean[i])
     testData.X[{ {},i,{},{} }]:div(std[i])
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
     for i = 1,trainData.size do
        trainData.X[{ i,{c},{},{} }] = normalization:forward(trainData.X[{ i,{c},{},{} }])
     end
     for i = 1,testData.size do
        testData.X[{ i,{c},{},{} }] = normalization:forward(testData.X[{ i,{c},{},{} }])
     end
  end

  print '==> verify statistics'

  -- It's always good practice to verify that data is properly
  -- normalized.

  for i,channel in ipairs(channels) do
     trainMean = trainData.X[{ {},i }]:mean()
     trainStd = trainData.X[{ {},i }]:std()

     testMean = testData.X[{ {},i }]:mean()
     testStd = testData.X[{ {},i }]:std()

     print('training data, '..channel..'-channel, mean: ' .. trainMean)
     print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

     print('test data, '..channel..'-channel, mean: ' .. testMean)
     print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
  end
end


--Function for any general transformation to the data 
function transform(func, X)
    ret = X:clone()
    for i = 1,ret:size(1) do 
        ret[i] = func.func(ret[i], func.params)
    end
    return ret
end
--Transformation functions
function rotate(slice, angle) 
    return image.rotate(slice, angle)
end

function hflip(slice, n_a)
    return image.hflip(slice)
end
