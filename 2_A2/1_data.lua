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

-- Download the dataset files. This will conver the .mat file format 
-- the tensor format using mattorch.









