mylist = ["Python","Ruby","Javascript","HTML"]

mylist

for item in mylist:
    print item

[item for item in mylist]

for item in mylist:
    print item + " is a fun programming language."

newlist = [item + " is a fun programming language." for item in mylist]

newlist

x = 10

if x > 5:
    print "It looks like x is greater than 5"

if x > 5 and x < 20:
    print "Hello"

number_list = [1,2,3,4,5,6,7,8,9,10]

for number in number_list:
    if number%2 == 0:
        print str(number) + " is even"
    elif number == 7:
        print str(number) + " is the best number!"
    else:
        print str(number) + " is odd"

