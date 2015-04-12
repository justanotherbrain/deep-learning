dofile 'skel.lua'
a = 3
b = 2
c = 1
batch = 128 
max = 101
print (a,b,c)
tlep = nn.TemporalLogExpPooling(a,b,c)
f2 = torch.Tensor(batch,max,2)
for i=1,max do
  f2[{{},{i},{1}}] = i
  f2[{{},{i},{2}}] = max-i + 1
end
print(f2)
out = tlep:forward(f2)
print(out)
gradIn = tlep:backward(f2,torch.ones(out:size()))
print(gradIn[1])

