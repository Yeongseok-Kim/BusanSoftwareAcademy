import torch

x_train=torch.FloatTensor([[100],[150],[300],[400],[130],[240],[350],[200],[100],[110],[190],[120],[130],[270],[255]])
y_train=torch.FloatTensor([[20],[24],[36],[47],[22],[32],[47],[42],[21],[21],[30],[25],[18],[38],[28]])

linear=torch.nn.Linear(1,1,bias=True)

model=torch.nn.Sequential(linear)

optimizer=torch.optim.SGD(model.parameters(),lr=0.00001)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis=model(x_train)

    cost=torch.mean((hypothesis-y_train)**2)
    cost.backward()
    optimizer.step()

    if step%1000==0:
        print(step,cost)
        with torch.no_grad():
            x_test=torch.FloatTensor([50.])
            predicted=model(x_test)
            print(predicted)