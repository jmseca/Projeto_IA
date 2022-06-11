import random

def rowcol_to_number(rc):
    size = 400
    list_bin_intervals = []
    i = 0
    while (size > 224):
        list_bin_intervals += [(i+1)*224]
        size -= 224
        i+=1
    list_bin_intervals += [size+list_bin_intervals[-1]]
    size_bin_intervals = i+1
    rc_number = []
    for n in range(size_bin_intervals):
        init = 0 if n==0 else list_bin_intervals[n-1]
        interval = rc[init:list_bin_intervals[n]]
        interval_number = int("".join(str(i) for i in interval),2)
        rc_number += [interval_number]
    return rc_number


rCol = []
for i in range(400):
    #rCol += [random.randint(0,1)]
    rCol += [0]


number = rowcol_to_number(rCol)
print(number)

binStr = ""
for element in number:
    binStr += "{0:b}".format(element)

check = "".join(str(i) for i in rCol)

print(check==binStr)
print(check)
print(binStr)