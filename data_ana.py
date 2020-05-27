import os

for i in os.listdir('data/val/'):
    file = os.path.join('data/val/', i)
    file = file + '/'
    count = 0
    for v in os.listdir(file):
        img = os.path.join(file, v)
        # print(img)
        if(count > 999):
            os.remove(img)
        count += 1
    print(file, count)