require 'image'
require 'torch'


data_directory = '/Users/sam.royston/school/deep/deep-learning/A2/SurrogateData/'

t = torch.load(data_directory .. "surrogate_traindata_2.t7")
input = image.toDisplayTensor{
   input=t, padding=3, nrow=32, saturate=false
}

labels = torch.load(data_directory .. "surrogate_labels_1.t7")
print(labels)

 -- image.display(input)