require 'csvigo'
csv = csvigo.load('kaggle.csv')
dofile 'data.lua'
ReadFiles({1})
count = 0
for i = 1,8000 do
    if csv.Category[i]-0 == testData.y[i][1] then
        count = count +1
    end
end
print (count /8000)
