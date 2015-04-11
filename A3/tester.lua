dofile 'A3_skeleton.lua'
tlep = nn.TemporalLogExpPooling(3,2,1)
--[[
f = torch.ones(1,4,3)
f[{{},{3},{}}] = 3
f[{{},{},{3}}] = 4
print(f)
out = tlep:forward(f)
print(out)
gradIn = tlep:backward(f,torch.ones(out:size()))
print(gradIn)
]]
batch = 1 
max = 6
f2 = torch.Tensor(batch,max,2)
for i=1,max do
  f2[{{},{i},{1}}] = i
  f2[{{},{i},{2}}] = max-i + 1
end
print(f2)
out = tlep:forward(f2)
print(out)
gradOut = torch.ones(1,2,2)
gradOut[1][2] = -1
print(gradOut)
gradIn = tlep:backward(f2,gradOut)
print(gradIn[1])

