---------------------------------------------------------------------
-- MonkeyBusiness
-- 
-- This script loads and pre-processes the STL-10 data set.
--
--
-- Michael Rabadi
---------------------------------------------------------------------

require 'torch'
require 'image'
require 'nn'
require 'mattorch'

print '==> preprocessing options'
cmd = torch.CmdLine()
cmd:text()
cmd:text('STL-10 Dataset Preprocessing')
cmd:text()
cmd:text('Options:')
cmd:option('-shuffle', true, 'shuffle the images before training?')
cmd:option('-extend', true, 'make data set larger by transforming images?')
cmd:option('-filter', 'v1', 'how should images be filtered: v1 | yuv')
cmd:text()
opt = cmd:parse(arg or {})

----------------------------------------------------------------------
print '==> download dataset'

-- Download the dataset files. This will convert the .mat file format 
-- the tensor format using mattorch.

-- dir = '/scratch/courses/DSGA1008/A2/matlab'
dir = '/home/rabad/data/stl10_matlab/'

if not paths.filep(dir) then
	os.execute('wget -qO- http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz | tar xvz')
	os.execute('cd stl10_matlab')
	os.execute('mv * ..')
	train_file = 'train.mat'
	test_file = 'test.mat'
	unlabeled_file = 'unlabeled.mat'
else
	train_file = dir .. 'train.mat'
	test_file = dir .. 'test.mat'
	unlabeled_file = dir .. 'unlabeled.mat'
end

loaded = mattorch.load(train_file)
t = loaded.X:transpose(loaded)
train_size=t:size()
trainData = {
	data = torch.reshape(t,train_size[1],96,96,3),
	labels = loaded.y[1],
	size = function() return train_size end
}


loaded = mattorch.load(test_file)
t = loaded.X:transpose(loaded)
test_size=t:size()
testData = {
	data = torch.reshape(t,test_size[1],96,96,3),
	labels = loaded.y[1],
	size = function() return test_size end
}

loaded = mattorch.load(unlabeled_file)
t = loaded.X:transpose(loaded)
unlabeled_size = t:size()
unlabeledData = {
	data = torch.reshape(t,unlabeled_size[1],96,96,3),
	labels = loaded.y[1],
	size = function() return unlabeled_size end
}


-------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing data. Start with floating point.

trainData.data = trainData.data:float()
testData.data = testData.data:float()
unlabeledData.data = unlabeledData.data:float()

-- Convert images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'

for i = 1,trainData:size() do
	trainData.data[i] = image.rgb2yuv(trainData.data[i])
end

for i = 1,testData:size() do
	testData.data[i] = image.rgb2yuv(testData.data[i])
end

for i = 1,unlabeledData:size() do
	unlabeledData[i] = image.rgb2yuv(unlabeledData.data[i])
end


-- name channels
channels = {'y','u','v'}

-- normalize each channel and store mean/std per channel.
-- the channels will be normalized according to the unlabeled
-- data. this way we don't actually cheat on the cross validation

mean = {}
std = {}

for i,channel in ipairs(channels) do
	mean[t] = unlabeledData[{ {}, i, {}, {} }]:mean()
	std[i] = unlabeledData[{ {}, i, {}, {} }]:std()
	unlabeledData.data[{ {}, i, {}, {} }]:add(-mean[i])
	unlabeledData.data[{ {}, i, {}, {} }]:div(std[i])

	-- train data
	trainData.data[{ {}, i, {}, {} }]:add(-mean[i])
	trainData.data[{ {}, i, {}, {} }]:div(std[i])

	-- test data
	testData.data[{ {}, i, {}, {} }]:add(-mean[i])
	testData.data[{ {}, i, {}, {} }]:div(std[i])
end

for i,channel in ipairs(channels) do
	trainMean = trainData.data[{ {},i }]:mean()
	trainStd = trainData.data[{ {},i }]:std()

	testMean = testData.data[{ {},i }]:mean()
	testStd = testData.data[{ {},i }]:std()


	unlabeledMean = unlabeledData.data[{ {}, i }]:mean()
	unlabeledStd = unlabeledData.data[{ {}, i }]:std()

	print('training data, '..channel..'-channel, mean: ' .. trainMean)
	print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

	print('test data, '..channel..'-channel, mean: ' .. testMean)
	print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
	
	print('unlabeled data, '..channel..'-channel, mean: ' .. unlabeledMean)
	print('unlabeled data, '..channel..'-channel, standard deviation: ' .. unlabeledStd)
end









