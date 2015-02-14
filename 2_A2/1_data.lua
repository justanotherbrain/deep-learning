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







