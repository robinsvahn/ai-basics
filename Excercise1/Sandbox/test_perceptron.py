import random

#########################################################################################
## Data skrivs in i formatet '11', '10', '01', eller '00'... 1 = sant, 0 = falskt      ##
##                                 t.ex. Groda har 4 ben(1) och är grön (1) -> 11      ##
#########################################################################################


threshold = 0.62


def summation(x1, x2, w1, w2):
    summa = (x1*w1)+(x2*w2)
    print(summa)
    if summa >= threshold:
        return 1
    else:
        return 0


def train(answer):
    correct = input('Var svaret rätt?: ')
    if correct == 'ja':
        return 0
    elif answer == 1:
        return -0.2
    else:
        return 0.2


w1 = random.randint(0, 100)*0.01
w2 = random.randint(0, 100)*0.01
print('Vikter: ', w1, ' ', w2)

while True:
    print()
    print('Vikter: ', w1, ' ', w2)

    vilkor = input('skriv in data: ')

    if vilkor[0] == '1':
        x1 = 1
    else:
        x1 = 0
    if vilkor[1] == '1':
        x2 = 1
    else:
        x2 = 0

    answer = summation(x1, x2, w1, w2)

    print('Svar:', answer)

    weight_change = train(answer)
    w1 += weight_change
    w2 += weight_change
