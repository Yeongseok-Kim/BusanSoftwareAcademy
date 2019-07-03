import torch
import random
from matplotlib import pyplot

def make_circle(center_x,center_y,r):
    point_list=[]
    for _ in range(100):
        while True:
            x=random.uniform(center_x-r,center_x+r)
            y=random.uniform(center_y-r,center_y+r)
            if (x-center_x)**2+(y-center_y)**2<=r**2:
                break
        point_list.append([x,y])
    return point_list

def make_graph_x(circle_list):
    graph_x=[]
    for i in circle_list:
        graph_x.append(i[0])
    return graph_x

def make_graph_y(circle_list):
    graph_y=[]
    for i in circle_list:
        graph_y.append(i[1])
    return graph_y

circle0=make_circle(1.0,1.0,3.0)
circle1=make_circle(4.0,4.0,3.0)+make_circle(-4.0,4.0,3.0)

data_input=torch.FloatTensor(circle0+circle1)
data_result=torch.FloatTensor([[0] for _ in range(100)]+[[1] for _ in range(200)])

linear1=torch.nn.Linear(2,2,bias=True)
linear2=torch.nn.Linear(2,1,bias=True)
sigmoid=torch.nn.Sigmoid()

model=torch.nn.Sequential(linear1,sigmoid,linear2,sigmoid)

criterion=torch.nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis=model(data_input)

    cost=criterion(hypothesis,data_result)
    cost.backward()
    optimizer.step()

    if step%1000==0:
        print(step,cost.item())

        with torch.no_grad():
            test_point=[[4.0,4.0],[5.0,5.0],[-1.0,-2.0],[-5.0,-5.0]]
            for point in test_point:
                predicted=model(torch.FloatTensor(point))
                print(predicted)

pyplot.scatter(make_graph_x(circle0),make_graph_y(circle0))
pyplot.scatter(make_graph_x(circle1),make_graph_y(circle1))
pyplot.show()