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



test_label_fd = torch.DiskFile(data_directory .. test_labels_file, "r", true)
test_label_fd:binary():littleEndianEncoding()
t_labels = torch.ByteTensor(5000, 1)
test_label_fd:readByte(t_labels:storage())

test_data_fd = torch.DiskFile(data_directory .. test_file, "r", true)
test_data_fd:binary():littleEndianEncoding()
t_data = torch.ByteTensor(5000, 3, 96, 96)
test_data_fd:readByte(t_data:storage())
t_data = t_data:transpose(3,4)
t_data = generate_feats(t_data)
test_data = torch.FloatTensor(5000,3,36,36)
test_labels = torch.FloatTensor(5000,1)


for i=1,test_data:size()[1] do
        test_data[i] = t_data[i]
        test_labels[i] = t_labels[i]
end


for i = 1,test_data:size()[1] do
    test_data[i] = test_data[i]:float()
    print('bueno')
end
-- load pre trained model

model = torch.load('trainingModels/winning_model.net')



testData = {
        data = test_data,
        labels = test_labels,
        size = function() return test_data:size() end
}


for i = 1,testData.data:size()[1] do
        testData.data[i] = image.rgb2yuv(testData.data[i])
end

channels = {'y','u','v'}

mean = {}
std = {}

for i,channel in ipairs(channels) do
        mean[i] = testData.data[{ {}, i, {}, {} }]:mean()
        std[i] = testData.data[{ {}, i, {}, {} }]:std()
        testData.data[{ {}, i, {}, {} }]:add(-mean[i])
        testData.data[{ {}, i, {}, {} }]:div(std[i])
end


neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1,neighborhood,1):float()


for c in ipairs(channels) do

        for i = 1,testData.data:size()[1] do
                testData.data[{i,{c},{},{}}]=normalization:forward(testData.data[{i,{c},{},{}}])
        end
end



test(model,testData)




