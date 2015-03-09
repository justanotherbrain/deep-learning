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
test_labels = torch.ByteTensor(5000, 1)
test_label_fd:readByte(test_labels:storage())

test_data_fd = torch.DiskFile(data_directory .. test_file, "r", true)
test_data_fd:binary():littleEndianEncoding()
test_data = torch.ByteTensor(5000, 3, 96, 96)
test_data_fd:readByte(test_data:storage())
test_data = test_data:transpose(3,4)
test_data = generate_feats(test_data)

for i = 1,test_data:size()[1] do
    test_data[i] = test_data[i]:float()
    print('bueno')
end
-- load pre trained model

model = torch.load('trainingModels/model.net')



testData = {
        data = test_data,
        labels = test_labels,
        size = function() return test_data:size() end
}

	test(model,testData)




