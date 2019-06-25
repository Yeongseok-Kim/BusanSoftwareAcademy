score=[38,65,89,16,95,71,63,48,49,66,37]
for i in score:
    if i>=95:
        print("A+",end=',')
    elif i>=60:
        print("B",end=',')
    else:
        print("F",end=',')