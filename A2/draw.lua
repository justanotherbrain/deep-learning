require 'image'
require 'torch'


data_directory = '/Users/sam.royston/school/deep/deep-learning/A2/SurrogateData/'

t = torch.load(data_directory .. "surrogate_traindata_2.t7")
input = image.toDisplayTensor{
   input=t, padding=3, nrow=32, saturate=false
}

image.display(input)