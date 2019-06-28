import time

def print_menu():
    print("----------")
    print("1.입금")
    print("2.출금")
    print("3.잔고조회")
    print("4.회원가입")
    print("5.대출")
    print("6.신용조회")
    print("----------")

def calculation_interest(time_past,time_now,account_price):
    for _ in range(time_past,time_now,30):
        account_price=account_price*(1.1)
    return account_price

def renew_price(account_dictionary,account_number):
    time_past=int(account_dictionary[account_number][5])
    time_now=int(time.time())
    account_dictionary[account_number][5]=time_now
    account_dictionary[account_number][0]=calculation_interest(time_past,time_now,account_dictionary[account_number][0])
    return account_dictionary

def renew_loan(account_dictionary,account_number):
    time_past=int(account_dictionary[account_number][6])
    time_now=int(time.time())
    account_dictionary[account_number][6]=time_now
    account_dictionary[account_number][1]=calculation_interest(time_past,time_now,account_dictionary[account_number][1])
    return account_dictionary

def input_account_number(account_dictionary):
    while True:
        account_number=input("계좌:")
        if account_number in account_dictionary:
            return account_number
        else:
            print("없는 계좌입니다.")

def account_deposit(account_dictionary): #입금
    account_number=input_account_number(account_dictionary)
    account_price=float(input("입금:"))
    account_dictionary=renew_price(account_dictionary,account_number)
    account_dictionary[account_number][0]+=account_price
    return account_dictionary

def account_withdraw(account_dictionary): #출금
    account_number=input_account_number(account_dictionary)
    account_price=float(input("출금:"))
    account_dictionary=renew_price(account_dictionary,account_number)
    account_dictionary[account_number][0]-=account_price
    return account_dictionary

def account_lookup(account_dictionary): #잔고조회
    account_number=input_account_number(account_dictionary)
    account_dictionary=renew_price(account_dictionary,account_number)
    print("잔고:"+str(account_dictionary[account_number][0]))

def account_join(account_dictionary): #회원가입
    while True:
        account_number=input("계좌:")
        if account_number in account_dictionary:
            print("중복 계좌입니다.")
        else:
            break
    account_name=input("이름:")
    account_age=int(input("나이:"))
    account_sex=input("성별:")
    account_dictionary[account_number]=[0.0,0.0,account_name,account_age,account_sex,0,0]
    return account_dictionary

def account_loan_lookup(account_dictionary,account_number):
    account_dictionary=renew_loan(account_dictionary,account_number)
    print("대출:"+str(account_dictionary[account_number][1]))

def account_loan(account_dictionary,account_number):
    account_price=float(input("대출:"))
    account_dictionary=renew_loan(account_dictionary,account_number)
    account_dictionary[account_number][1]+=account_price
    return account_dictionary

def account_loan_repayment(account_dictionary,account_number):
    account_price=float(input("상환:"))
    account_dictionary=renew_loan(account_dictionary,account_number)
    account_dictionary[account_number][1]-=account_price
    return account_dictionary

def account_loan_menu(account_dictionary): #대출
    account_number=input_account_number(account_dictionary)
    print("1.대출조회")
    print("2.대출")
    print("3.대출상환")
    menu_selection=int(input("선택:"))
    if menu_selection==1:
        account_loan_lookup(account_dictionary,account_number)
    elif menu_selection==2:
        account_dictionary=account_loan(account_dictionary,account_number)
    else:
        account_dictionary=account_loan_repayment(account_dictionary,account_number)
    return account_dictionary

def account_credit_rating(account_dictionary): #신용조회
    account_number=input_account_number(account_dictionary)
    account_dictionary=renew_price(account_dictionary,account_number)
    account_dictionary=renew_loan(account_dictionary,account_number)
    if account_dictionary[account_number][0]>=account_dictionary[account_number][1]:
        print("1등급입니다.")
    else:
        print("9등급입니다.")

account_dictionary={} #계좌번호:[잔고,대출금,이름,나이,성별,잔액조회시간,대출조회시간]

while True:
    print_menu()
    menu_selection=int(input("선택:"))
    if menu_selection==1:
        account_dictionary=account_deposit(account_dictionary)
    elif menu_selection==2:
        account_dictionary=account_withdraw(account_dictionary)
    elif menu_selection==3:
        account_lookup(account_dictionary)
    elif menu_selection==4:
        account_dictionary=account_join(account_dictionary)
    elif menu_selection==5:
        account_dictionary=account_loan_menu(account_dictionary)
    else:
        account_credit_rating(account_dictionary)