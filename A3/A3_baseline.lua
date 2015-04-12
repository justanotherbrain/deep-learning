require 'torch'
require 'nn'
require 'optim'
require 'xlua'
--dofile 'A3_skeleton.lua'
dofile 'skel.lua'
torch.manualSeed(123)
ffi = require('ffi')

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

--- Here we simply encode each document as a fixed-length vector 
-- by computing the unweighted average of its word vectors.
-- A slightly better approach would be to weight each word by its tf-idf value
-- before computing the bag-of-words average; this limits the effects of words like "the".
-- Still better would be to concatenate the word vectors into a variable-length
-- 2D tensor and train a more powerful convolutional or recurrent model on this directly.
function preprocess_data(raw_data, wordvector_table, opt)
    local totalSamples = raw_data.index:size(1) * raw_data.index:size(2)
    local samplesPerClass = raw_data.index:size(2)
    if opt.debug == 1 then
      totalSamples = 2 * opt.minibatchSize * raw_data.index:size(1)
      samplesPerClass = 2 * opt.minibatchSize
    end
    local data = torch.zeros(totalSamples, opt.inputDim, 1)
    local labels = torch.zeros(totalSamples)
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(totalSamples)
    
    for i=1,raw_data.index:size(1) do
        for j=1,samplesPerClass do
            local k = order[(i-1)*samplesPerClass + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            -- break each review into words and compute the document average
            for word in document:gmatch("%S+") do
                if wordvector_table[word:gsub("%p+", "")] then
                    doc_size = doc_size + 1
                    data[k]:add(wordvector_table[word:gsub("%p+", "")])
                end
            end

            data[k]:div(doc_size)
            labels[k] = i
        end
    end

    return data, labels
end

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local minibatch = torch.Tensor(opt.minibatchSize,data:size(2))
    local minibatch_labels = torch.Tensor(opt.minibatchSize)
    if opt.type == 'cuda' then
      minibatch:cuda()
      minibatch_labels:cuda()
      model:cuda()
      criterion:cuda()
      test_data:cuda()
      test_labels:cuda()
    end
    local function feval(x) 
        minibatch:copy(data:sub(opt.idx, opt.idx + opt.minibatchSize - 1, 1, data:size(2)))
        minibatch_labels:copy(labels:sub(opt.idx, opt.idx + opt.minibatchSize - 1))
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    local bestAccuracy = 0
    local accuracy = 1
    local epoch = 1
    local nBatches = math.floor(data:size(1) / opt.minibatchSize)
    while accuracy - bestAccuracy > opt.accThresh and epoch <= opt.maxEpochs do
        local order = torch.randperm(nBatches) -- not really good randomization
        for batch=1,nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
        end
        accuracy = test_model(model, test_data, test_labels, opt)
        if epoch % opt.learningRateDecay == 0 then
          opt.learningRate = opt.learningRate / 2
        end
        print("epoch ", epoch, " error: ", accuracy)
        epoch = epoch + 1
    end
end

function test_model(model, data, labels, opt)
    
    model:evaluate()

    local pred = model:forward(data)
    local _, argmax = pred:max(2)
    local err = torch.ne(argmax:double(), labels:double()):sum() / labels:size(1)

    --local debugger = require('fb.debugger')
    --debugger.enter()

    return err
end

function ParseCommandLine()
    local cmd = torch.CmdLine()
    cmd:option('-glovePath','glove.6B.50d.txt','path to raw glove data .txt file')
    cmd:option('-dataPath','/scratch/courses/DSGA1008/A3/data/train.t7b','path to data file')
    cmd:option('-inputDim',50,'dim of word vectors')
    cmd:option('-debug', 0, 'debug=reduced data set')
    cmd:option('-valPerc',10,'percentage of all samples to use for validation')
    cmd:option('-maxEpochs', 10,'max epochs to train')
    cmd:option('-minibatchSize', 128,'minibatch size')
    cmd:option('-learningRate', 0.1,'learning rate for SGD')
    cmd:option('-momentum', 0.1,'momentum in sgd') 
    cmd:option('-idx', 1,'')
    cmd:option('-type', 'double', 'cuda,float,double')
    cmd:option('-tlep', 'inf','inf=maxpool, any positive number=tlep')
    cmd:option('-accThresh', 0.005, 'percentage diff of accuracy before stopping')
    cmd:option('-learningRateDecay', 5, 'halve the learning rate every n epochs')
    opt = cmd:parse(arg or {})
    if opt.debug == 1 then
      print('DEBUG MODE ACTIVATED!')
      opt.maxEpochs = 1
      opt.valPerc = 50
    end
    if opt.type == 'cuda' then
      if not opt.tlep == 'inf' then
        print('ERROR. TLEP not stable with cuda.')
      end
      print('Using Cuda')
      require 'cunn'
      torch.setdefaulttensortype('torch.FloatTensor')
    end
    return opt 
end

function ModelCrit(opt)
-- construct model:
    model = nn.Sequential()
   
    -- if you decide to just adapt the baseline code for part 2, you'll probably want to make this linear and remove pooling
    model:add(nn.TemporalConvolution(1, 20, 10, 1))
    
    --------------------------------------------------------------------------------------
    -- Replace this temporal max-pooling module with your log-exponential pooling module:
    --------------------------------------------------------------------------------------
    if opt.tlep == 'inf' then
      model:add(nn.TemporalMaxPooling(3, 1))
    else
      model:add(nn.TemporalLogExpPooling(3,1, tonumber(opt.tlep)))
    end
    
    model:add(nn.Reshape(20*39, true))
    model:add(nn.Linear(20*39, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
    return model, criterion
end

function main()
    print("Parse args...")
    local opt = ParseCommandLine()
    print(opt)
    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Create model and criterion...")
    model, criterion = ModelCrit(opt)
    print(model)
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    local processed_data, labels = preprocess_data(raw_data, glove_table, opt)
    
    -- split data into makeshift training and validation sets
    local training_data = processed_data:sub(1, processed_data:size(1)*(1-opt.valPerc/100), 1, processed_data:size(2)):clone()
    local training_labels = labels:sub(1, processed_data:size(1)*(1-opt.valPerc/100)):clone()
    
    -- make your own choices - here I have not created a separate test set
    local test_data = processed_data:sub(processed_data:size(1)*(1-opt.valPerc/100), processed_data:size(1), 1, processed_data:size(2)):clone()
    local train_labels = labels:sub(processed_data:size(1)*(1-opt.valPerc/100), processed_data:size(1)):clone()

    local test_data = training_data:clone() 
    local test_labels = training_labels:clone()
    print("Train model")
    train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    print("Test model")
    local results = test_model(model, test_data, test_labels)
    print(results)
end

main()
