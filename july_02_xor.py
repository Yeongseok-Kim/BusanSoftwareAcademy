import torch

X=torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
Y=torch.FloatTensor([[0],[1],[1],[0]])

linear1=torch.nn.Linear(2,2,bias=True)
linear2=torch.nn.Linear(2,1,bias=True)
sigmoid=torch.nn.Sigmoid()

model=torch.nn.Sequential(linear1,sigmoid,linear2,sigmoid)

criterion=torch.nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1)

for step in range(1,10001):
    optimizer.zero_grad()
    hypothesis=model(X)

    cost=criterion(hypothesis,Y)
    cost.backward()
    optimizer.step()

    if step%1000==0:
        print(step,cost.item())

        with torch.no_grad():
            x_test=[[0,0],[0,1],[1,0],[1,1]]
            for x in x_test:
                print(model(torch.FloatTensor(x)))