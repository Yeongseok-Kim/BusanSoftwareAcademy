import torch
import random
from matplotlib import pyplot

X=[]
Y=[]

for _ in range(100):
    while True:
        x=random.uniform(-2.0,4.0)
        y=random.uniform(-2.0,4.0)
        if (x-1)**2+(y-1)**2<=9:
            break
    X.append([x])
    Y.append([y])

x_tranin1=torch.FloatTensor(X)
y_tranin1=torch.FloatTensor(Y)

X=[]
Y=[]

for _ in range(100):
    while True:
        x=random.uniform(1.0,7.0)
        y=random.uniform(1.0,7.0)
        if (x-4)**2+(y-4)**2<=9:
            break
    X.append([x])
    Y.append([y])

for _ in range(100):
    while True:
        x=random.uniform(-7.0,-1.0)
        y=random.uniform(1.0,7.0)
        if (x+4)**2+(y-4)**2<=9:
            break
    X.append([x])
    Y.append([y])

x_tranin2=torch.FloatTensor(X)
y_tranin2=torch.FloatTensor(Y)

data_input1=torch.cat([x_tranin1,y_tranin1],dim=1)
data_input2=torch.cat([x_tranin2,y_tranin2],dim=1)
data_input=torch.cat([data_input1,data_input2],dim=0)

output=[[0] for _ in range(100)]+[[1] for _ in range(200)]

data_output=torch.FloatTensor(output)

linear1=torch.nn.Linear(2,2,bias=True)
linear2=torch.nn.Linear(2,1,bias=True)
sigmoid=torch.nn.Sigmoid()

model=torch.nn.Sequential(linear1,sigmoid,linear2,sigmoid)

criterion=torch.nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis=model(data_input)

    cost=criterion(hypothesis,data_output)
    cost.backward()
    optimizer.step()

    if step%1000==0:
        print(step,cost.item())

        with torch.no_grad():
            predicted=model(torch.FloatTensor([4,4]))
            print(predicted)

pyplot.scatter(x_tranin1,y_tranin1)
pyplot.scatter(x_tranin2,y_tranin2)
pyplot.show()