# 141
my_list=["A","b","c","D"]
for i in my_list:
    if i.isupper():
        print(i,end=' ')
print()

# 142
my_list=["A","b","c","D"]
for i in my_list:
    if i.islower():
        print(i,end=' ')
print()

# 143
my_list=["A","b","c","D"]
for i in my_list:
    if i.isupper():
        i=i.lower()
    else:
        i=i.upper()
    print(i,end=' ')
print()

# 144
file_list=['hello.py','ex01.py','ch02.py','intro.hwp']
for i in file_list:
    a,b=i.split('.')
    print(a,end=' ')
print()

# 145
filenames=['intra.h','intra.c','define.h','run.py']
for i in filenames:
    a,b=i.split('.')
    if b=='h':
        print(i,end=' ')
print()

# 146
filenames=['intra.h','intra.c','define.h','run.py']
for i in filenames:
    a,b=i.split('.')
    if b=='c' or b=='h':
        print(i,end=' ')
print()

# 147
my_list=[3,-20,-3,44]
new_list=[]
for i in my_list:
    if i>0:
        new_list.append(i)
print(new_list)

# 148
my_list=["A","b","c","D"]
upper_list=[]
for i in my_list:
    if i.isupper():
        upper_list.append(i)
print(upper_list)

# 149
my_list=[3,4,4,5,6,6]
sole_list=[]
for i in my_list:
    if i in sole_list:
        pass
    else:
        sole_list.append(i)
print(sole_list)

# 150
my_list=[3,4,5]
sum=0
for i in my_list:
    sum+=i
print(sum)