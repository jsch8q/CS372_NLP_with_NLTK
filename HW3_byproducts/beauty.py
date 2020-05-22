from add import Adder

def sub(self, a,b) :
    return a-b

Adder.add = sub

calc = Adder()
a = calc.add(4, 3)
print(a)
