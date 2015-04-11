dofile 'A3_skeleton.lua'
tlep = nn.TemporalLogExpPooling(3,2,2)
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
f2 = torch.Tensor(7,1)
for i=1,7 do
  f2[i] = i
end
print(f2)
out = tlep:forward(f2)
print(out)
gradIn = tlep:backward(f2,torch.ones(out:size()))
print(gradIn)
