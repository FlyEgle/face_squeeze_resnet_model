import random

tmp = ''
for i in range(4):
    n = random.randrange(0, 2)

    if n == 0:
        num = random.randrange(65, 91)
        tmp += chr(num)
    else:
        k = random.randrange(0, 10)
        tmp += str(k)

print(tmp)
