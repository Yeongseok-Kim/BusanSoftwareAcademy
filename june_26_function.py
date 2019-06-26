#231
def print_bitcoin():
    print("bitcoin")

#232
print_bitcoin()

#233
for i in range(100):
    print_bitcoin()

#234
def print_coins():
    for i in range(100):
        print("botcoin")

#235
'''
hello()
def hello():
    print("Hi")

Python은 인터프리터 언어이기 때문에 먼저 정의되지 않은 함수는 호출할 수 없다
'''

#236
'''
A
B
C
A
B
'''

#237
'''
A
B
C
'''

#238
'''
A
D
B
C
D
'''

#239
#SyntaxError: invalid syntax
#올바른 코드
def message1():
    print("A")
def message2():
    print("B")

message1()
message2()
'''
올바른 코드의 출력 결과:
A
B
'''

#240
#SyntaxError: invalid syntax
#올바른 코드
def message1():
    print("A")

def message2():
    print("B")

def message3():
    for i in range(3):
        message2()

print("C")
message1()
message3()
'''
올바른 코드의 출력 결과:
C
A
B
B
B
'''