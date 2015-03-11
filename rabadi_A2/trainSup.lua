require 'csvigo'
require 'representation'
require 'augment'
require 'torch'
require 'nn'
require 'cunn'
require 'xlua'

opt = {
        learningRate = 1e-3,
        optimization = 'SGD',
        momentum = 0,
        weightDecay = 0,
        t0 = 1,
        type = 'cuda',
        batchSize = 1,
        save = 'trainingModels'
}



--require 'representation.lua'
--dofile('representation.lua')

--data_directory = 'LearnedFeatures'

data_directory = 'stl10_binary/'

train_file = 'train_X.bin'
train_labels_file = 'train_y.bin'

test_file = 'test_X.bin'
test_labels_file = 'test_y.bin'




function generate_feats( test )
    
    -- generate features for data

    features = torch.Tensor(test:size()[1],3,36,36)
    for i=1, test:size()[1] do
    	features[i] = get_inset(test[i])
    end
    return features
end

function get_feature_model(model, n)
    
    -- remove last layers of model

    if n == 0 then return model end
    ret = nn.Sequential()
    for i = 1,model:size()-n do
        ret:add(model:get(i):clone())
    end
    return ret
end 



training_label_fd = torch.DiskFile(data_directory .. train_labels_file, "r", true)
training_label_fd:binary():littleEndianEncoding()
t_labels = torch.ByteTensor(5000, 1)
training_label_fd:readByte(t_labels:storage())

training_data_fd = torch.DiskFile(data_directory .. train_file, "r", true)
training_data_fd:binary():littleEndianEncoding()
t_data = torch.ByteTensor(5000, 3, 96, 96)
training_data_fd:readByte(t_data:storage())
t_data = t_data:transpose(3,4)
t_data = generate_feats(t_data)
training_data = torch.FloatTensor(5000,3,36,36)
training_labels = torch.FloatTensor(5000,1)

for i=1,training_data:size()[1] do
	training_data[i] = t_data[i]
	training_labels[i] = t_labels[i]
end

for i = 1,training_data:size()[1] do
    training_data[i] = training_data[i]:float()
    print('bueno')
end
-- load pre trained model


for i = 1,training_data:size()[1] do
        training_data[i] = image.rgb2yuv(training_data[i])
end

channels = {'y','u','v'}

mean = {}
std = {}

for i,channel in ipairs(channels) do
        mean[i] = training_data[{ {}, i, {}, {} }]:mean()
        std[i] = training_data[{ {}, i, {}, {} }]:std()
        training_data[{ {}, i, {}, {} }]:add(-mean[i])
        training_data[{ {}, i, {}, {} }]:div(std[i])
end


neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1,neighborhood,1):float()

for c in ipairs(channels) do

        for i = 1,training_data:size()[1] do
                training_data[{i,{c},{},{}}]=normalization:forward(training_data[{i,{c},{},{}}])
        end
end


nstates={64,128,256,512}
noutputs = 10
filtsize = 2
poolsize = 2
model = torch.load('featureEncoder/model.net')

model = get_feature_model(model, 6)

model:add(nn.SpatialConvolutionMM(nstates[3],nstates[4],filtsize,filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

model:add(nn.View(nstates[4]*(filtsize+1)*(filtsize+1)))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[4]*(filtsize+1)*(filtsize+1),nstates[4]))
model:add(nn.ReLU())

model:add(nn.Linear(nstates[4], noutputs))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()



for i = 0,100 do
	dofile('trainData.lua')
	train_model(model,criterion,trainData)
	test(model,validData)
end

-- folds = 5



-- for i=1,folds do
-- 	train = torch.Tensor(training_data:size()[1] - training_data:size()[1] / folds, 3, 36, 36)
-- 	train:sub(1, (i - 1) * (training_data:size()[1] / folds) + 1):fill()
-- 	validation = training_data:sub( (i - 1) * (training_data:size()[1] / folds) + 1 ) , i * (training_data:size()[1] / folds))
-- end



