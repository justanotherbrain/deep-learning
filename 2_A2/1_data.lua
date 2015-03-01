---------------------------------------------------------------------
-- MonkeyBusiness
-- 
-- This script loads and pre-processes the STL-10 data set.
--
--
-- Michael Rabadi
---------------------------------------------------------------------
require 'paths'
require 'torch'
require 'image'
require 'nn'
matio = require 'matio'

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

--dir = '/scratch/courses/DSGA1008/A2/matlab'
dir = paths.cwd() .. '/'
if not paths.filep(dir .. 'train.mat') then
	os.execute('wget -qO- http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz | tar xvz')
	os.execute('mv stl10_matlab/* .')
	train_file = 'train.mat'
	test_file = 'test.mat'
	unlabeled_file = 'unlabeled.mat'
else
	train_file = dir .. 'train.mat'
	test_file = dir .. 'test.mat'
	unlabeled_file = dir .. 'unlabeled.mat'
end

loaded = matio.load(train_file)
t = loaded.X:transpose(1,2)
train_size=t:size()
trainData = {
	data = torch.reshape(t,train_size[2],96,96,3),
	labels = loaded.y[1],
	size = function() return train_size end
}


loaded = matio.load(test_file)
t = loaded.X:transpose(1,2)
test_size=t:size()
testData = {
	data = torch.reshape(t,test_size[2],96,96,3),
	labels = loaded.y[1],
	size = function() return test_size end
}

-- loaded = matio.load(unlabeled_file)
-- t = loaded.X:transpose(1,2)
-- unlabeled_size = t:size()
loaded = torch.DiskFile('unlabeled_X.bin','r',true)
loaded:binary():littleEndianEncoding()
tmp = torch.ByteTensor(100000,96,96,3)
loaded:readByte(tmp:storage())
unlabeled_size = 100000
unlabeledData = {
	data = tmp:transpose(2,3),
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

for i = 1,train_size[2] do
	trainData.data[i] = image.rgb2yuv(trainData.data[i])
end

for i = 1,test_size[2] do
	testData.data[i] = image.rgb2yuv(testData.data[i])
end

for i = 1,unlabeled_size do
	unlabeledData.data[i] = image.rgb2yuv(unlabeledData.data[i])
end


-- name channels
channels = {'y','u','v'}

-- normalize each channel and store mean/std per channel.
-- the channels will be normalized according to the unlabeled
-- data. this way we don't actually cheat on the cross validation

mean = {}
std = {}

for i,channel in ipairs(channels) do
	mean[i] = unlabeledData.data[{ {}, i, {}, {} }]:mean()
	std[i] = unlabeledData.data[{ {}, i, {}, {} }]:std()
	unlabeledData.data[{ {}, i, {}, {} }]:add(-mean[i])
	unlabeledData.data[{ {}, i, {}, {} }]:div(std[i])

	-- train data
	trainData.data[{ {}, i, {}, {} }]:add(-mean[i])
	trainData.data[{ {}, i, {}, {} }]:div(std[i])

	-- test data
	testData.data[{ {}, i, {}, {} }]:add(-mean[i])
	testData.data[{ {}, i, {}, {} }]:div(std[i])
end


neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1,neighborhood,1):float()

for c in ipairs(channels) do
	for i = 1,train_size[2] do
		trainData.data[{i,{c},{},{}}] = normalization:forward(trainData.data[{i,{c},{},{},}])
	end
	for i = 1,test_size[2] do
		testData.data[{i,{c},{},{}}]=normalization:forward(testData.data[{i,{c},{},{}}])
	end
	for i = 1,unlabeled_size do
		unlabeledData.data[{i,{c},{},{}}]=normalization:forward(unlabeledData.data[{i,{c},{},{}}])
	end
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









