require 'torch'
require 'nn'
require 'optim'
require 'xlua'
dofile 'MONKEY_BUSINESS_A3_skeleton.lua'
torch.manualSeed(123)
ffi = require('ffi')


--- Parses and loads the word2vec word vectors into a hash table:
-- word2vec_table['word'] = vector
function load_word2vec(opt)
    local path = opt.wordVectorPath
    local inputDim = opt.inputDim
    print('Using custom word vectors')
    local f = io.open(opt.wordTable, "r")
    if f ~= nil then return f end
    local word2vec_file = io.open(path)
    local word2vec_table = {}

    local line = word2vec_file:read("*l")
    while line do
        -- read the word2vec text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    word2vec_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                word2vec_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = word2vec_file:read("*l")
    end
    print('Saving dictionary as torch file for later')
    torch.save(opt.wordTable, word2vec_table)
    return word2vec_table
end

--- Parses and loads the wordVector word vectors into a hash table:
-- wordVector_table['word'] = vector
function load_wordVector(opt)
    local inputDim = opt.inputDim
    local path = opt.wordVectorPath
    if opt.wv == 'wv' then
      return load_word2vec(opt)
    end
    local wordVector_file = io.open(path)
    local wordVector_table = {}

    local line = wordVector_file:read("*l")
    while line do
        -- read the wordVector text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    wordVector_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                wordVector_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = wordVector_file:read("*l")
    end
    
    return wordVector_table
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

function preprocess_sentence(raw_data, wordvector_table, opt)
    --This code is very similar to the code above, but words are saved instead of added
    local totalSamples = raw_data.index:size(1) * raw_data.index:size(2)
    local samplesPerClass = raw_data.index:size(2)
    if opt.debug == 1 then
      totalSamples = 2 * opt.minibatchSize * raw_data.index:size(1)
      samplesPerClass = 2 * opt.minibatchSize
    end
    local data = torch.zeros(totalSamples, opt.sentenceDim + 2 * opt.padding, opt.inputDim)
    local labels = torch.zeros(totalSamples)
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(totalSamples)
    
    local tempSentence = torch.zeros(opt.sentenceDim, opt.inputDim)
    for i=1,raw_data.index:size(1) do
        for j=1,samplesPerClass do
            local k = order[(i-1)*samplesPerClass + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            -- Save as many words up to the max size, leaving the possibility of trailing zero vectors
            tempSentence:zero()
            for word in document:gmatch("%S+") do
                if wordvector_table[word:gsub("%p+", "")] then
                    doc_size = doc_size + 1
                    local wordVector = wordvector_table[word:gsub("%p+", "")]
                    tempSentence[{{doc_size},{}}] = wordVector
                    if doc_size == opt.sentenceDim then break end--Arbitrarily crop large review 
                end
            end
            --Not centering this as I'm unsure if this will have an effect.
            data[{{k},{opt.padding + 1, opt.padding + opt.sentenceDim},{1,opt.inputDim}}] = tempSentence
            labels[k] = i
        end
    end

    return data, labels
end

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()
    local logger = optim.Logger(paths.concat(opt.save, 'test.log'))
    -- optimization functional to train the model with torch's optim library
    local minibatch 
    if opt.sentenceDim == 0 then
      minibatch = torch.Tensor(opt.minibatchSize,data:size(2), 1)
    else
      minibatch = torch.Tensor(opt.minibatchSize,data:size(2), data:size(3))
    end
    local minibatch_labels = torch.Tensor(opt.minibatchSize)
    if opt.type == 'cuda' then
      minibatch = minibatch:cuda()
      minibatch_labels = minibatch_labels:cuda()
      test_data = test_data:cuda()
    end
    local function feval(x) 
        model:training()
        if opt.sentenceDim == 0 then
          minibatch:copy(data:sub(opt.idx, opt.idx + opt.minibatchSize - 1, 1, data:size(2)))
        else
          minibatch:copy(data:sub(opt.idx, opt.idx + opt.minibatchSize - 1, 1, data:size(2), 1, data:size(3)))
        end
        minibatch_labels:copy(labels:sub(opt.idx, opt.idx + opt.minibatchSize - 1))
        local f = model:forward(minibatch)
        local minibatch_loss = criterion:forward(f, minibatch_labels)
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    local err = 1
    local epoch = 1
    local nBatches = math.floor(data:size(1) / opt.minibatchSize)
    local time = os.time()
    local elapsed = 0
    local olderr = err + 1
    --If error is increasing or not decreasing a lot, or times up, quit
    while olderr - err  > opt.errThresh and elapsed/60  < opt.maxTime  do
        print('epoch: '..epoch)
        local order = torch.randperm(nBatches) -- not really good randomization
        for batch=1,nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
        end
        --Update the errors
        local newerr = test_model(model, test_data, test_labels, opt)
        olderr = err 
        err = newerr
        if epoch % opt.learningRateDecay == 0 then
          opt.learningRate = opt.learningRate / 2
        end
        elapsed = os.difftime(os.time(),time)
        logger:add{['elapsed']=elapsed/60,['epoch']=epoch,['error']= err}
        
        print("Saving model")
        torch.save(opt.filename, model:double())
        epoch = epoch + 1
    end

    print('Final validation error: ', err)
end

function test_model(model, data, labels, opt)
    model:evaluate()
    local err = 0
    for i=1,data:size(1) do
      local pred = model:forward(data[{{i}}])
      local _, argmax = pred:max(2)
      err = err + (argmax:double()[1][1] == labels[i] and 0 or 1)
    end
    --local debugger = require('fb.debugger')
    --debugger.enter()

    return err/labels:size(1)
end

function ParseCommandLine()
    local cmd = torch.CmdLine()
    cmd:option('-save', '', 'Where to save information.')
    cmd:option('-wv', 'wv', 'wv= custom,glove=not')
    cmd:option('-wordVectorPath','WordVectors/dictionary.txt','path to raw wordVector data .txt file')
    cmd:option('-wordTable', 'dictionary.t7b', 'Torch-format vector table')
    cmd:option('-dataPath','/scratch/courses/DSGA1008/A3/data/train.t7b','path to data file')
    cmd:option('-inputDim',200,'dim of word vectors')
    cmd:option('-debug', 0, 'debug=reduced data set')
    cmd:option('-valPerc',10,'percentage of all samples to use for validation')
    cmd:option('-minibatchSize', 128,'minibatch size')
    cmd:option('-learningRate', 0.1,'learning rate for SGD')
    cmd:option('-momentum', 0.1,'momentum in sgd') 
    cmd:option('-idx', 1,'')
    cmd:option('-type', 'double', 'cuda,float,double')
    cmd:option('-tlep', 'inf','inf=maxpool, any positive number=tlep')
    cmd:option('-errThresh', 0.00005, 'percentage diff of error before stopping')
    cmd:option('-learningRateDecay', 10, 'halve the learning rate every n epochs')
    cmd:option('-maxTime', 50, 'maximum training time (minutes)')
    cmd:option('-sentenceDim', 100, 'Number of words to use in sentence. If zero, use bag of words')
    cmd:option('-filename', 'model.net', 'Filename of model to output')
    cmd:option('-padding', 7, 'padding for sentence')
    cmd:option('-train', 0, ' 1=train 2=read other=debug in single sentence')
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
      require 'cutorch'
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
      print('MaxPooling')
      model:add(nn.TemporalMaxPooling(3, 1))
    else
      model:add(nn.TemporalLogExpPooling(3,1, tonumber(opt.tlep)))
    end
    
    model:add(nn.Reshape(20*39, true))
    model:add(nn.Linear(20*39, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
    if opt.type == 'cuda' then
      model = model:cuda()
      criterion = criterion:cuda()
    end
    return model, criterion
end

function SentenceModelCrit(opt)
  model = nn.Sequential()
  local transformedDim = 256
  local kernel = 7
  local kernel2 = 3
  local nOutputFrames = opt.sentenceDim + 2 * opt.padding
  --1
  model:add(nn.TemporalConvolution(opt.inputDim,transformedDim, kernel))
  nOutputFrames = math.floor((nOutputFrames-kernel)/1) + 1
  model:add(nn.Threshold())
  model:add(nn.TemporalMaxPooling(3,3))
  nOutputFrames = math.floor((nOutputFrames-3)/3) + 1
  --2
  model:add(nn.TemporalConvolution(transformedDim,transformedDim, kernel2))
  nOutputFrames = math.floor((nOutputFrames-kernel2)/1) + 1
  model:add(nn.Threshold())
  model:add(nn.TemporalMaxPooling(3,3))
  nOutputFrames = math.floor((nOutputFrames-3)/3) + 1
  --Reshape 
  model:add(nn.Reshape(nOutputFrames * transformedDim))
  --3
  model:add(nn.Linear(nOutputFrames * transformedDim, 256))
  model:add(nn.Threshold())
  model:add(nn.Dropout(0.5))
  
  --4/Final
  model:add(nn.Linear(256,5))
  model:add(nn.LogSoftMax())
  
  criterion = nn.ClassNLLCriterion()
  if opt.type == 'cuda' then
    model = model:cuda()
    criterion = criterion:cuda()
  end
  return model, criterion

end
function Train(opt)
    print("Parse args...")
    print(opt)
    
    print("Loading word vectors...")
    local wordVector_table = load_wordVector(opt)
    
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    if opt.sentenceDim == 0 then
      processed_data, labels = preprocess_data(raw_data, wordVector_table, opt)
      model, criterion = ModelCrit(opt)
    else
      processed_data, labels = preprocess_sentence(raw_data, wordVector_table, opt)
      model, criterion = SentenceModelCrit(opt)
    end
    print("Create model and criterion...")
    print(model)

    -- split data into makeshift training and validation sets
    local training_data = processed_data:sub(1, processed_data:size(1)*(1-opt.valPerc/100), 1, processed_data:size(2))
    local training_labels = labels:sub(1, processed_data:size(1)*(1-opt.valPerc/100))
    
    -- make your own choices - here I have not created a separate test set
    local test_data = processed_data:sub(processed_data:size(1)*(1-opt.valPerc/100)+1, processed_data:size(1), 1, processed_data:size(2))
    local test_labels = labels:sub(processed_data:size(1)*(1-opt.valPerc/100)+1, processed_data:size(1))

    print("Train model")
    train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
end

function process_sentence(data, wordvector_table, opt)
  local sentence = ffi.string(data):lower()
  local sentenceTensor = torch.zeros(opt.sentenceDim + 2 * opt.padding, opt.inputDim)
  for word in sentence:gmatch("%S+") do
      if wordvector_table[word:gsub("%p+", "")] then
          doc_size = doc_size + 1
          local wordVector = wordvector_table[word:gsub("%p+", "")]
          sentenceTensor[{{opt.padding + doc_size},{}}] = wordVector
          if doc_size == opt.sentenceDim then break end--Arbitrarily crop large review 
      end
  end
  sentenceTensor = sentenceTensor:reshape(1,opt.sentenceDim + 2 * opt.padding,opt.inputDim)
  return sentenceTensor 
end


function ReadSentence(opt)
  local model = torch.load(opt.filename)
  local n = tonumber(io.read("*l"))
  for i = 1,n do
    local sentenceRaw = io.read("*l")
    local wordvector_table = torch.load(opt.wordTable)
    local sentence = process_sentence(sentenceRaw, wordvector_table, opt)
    local _, argmax = model:forward(sentence):max(2)
    print(argmax[1][1])
  end
end

function DebugTest(opt)
  model, crit = SentenceModelCrit(opt)
  f = torch.Tensor(640, opt.padding * 2 + opt.sentenceDim, opt.inputDim)
  print(model:forward(f):size())
  torch.save(opt.filename, model)
end
local opt = ParseCommandLine()
if opt.train == 1 then
  Train(opt)
elseif opt.train == 2 then
  ReadSentence(opt)
else
  DebugTest(opt)
end
