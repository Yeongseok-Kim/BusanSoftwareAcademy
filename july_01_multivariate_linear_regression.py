import torch

x_train=torch.FloatTensor([[73,80,75],
                           [93,88,93],
                           [89,91,90],
                           [96,98,100],
                           [73,66,70]])
y_train=torch.FloatTensor([[152],[185],[180],[196],[142]])

linear=torch.nn.Linear(3,1,bias=True)

model=torch.nn.Sequential(linear)

optimizer=torch.optim.SGD(model.parameters(),lr=1e-5)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis=model(x_train)

    cost=torch.mean((hypothesis-y_train)**2)
    cost.backward()
    optimizer.step()

    if step%1000==0:
        print(step,cost)
        with torch.no_grad():
            x_test=torch.FloatTensor([60,80,90])
            predicted=model(x_test)
            print(predicted)