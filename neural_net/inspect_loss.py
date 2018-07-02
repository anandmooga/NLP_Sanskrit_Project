import csv
with open('Loss_mini.csv','r') as cf:
    rd = csv.reader(cf)
    d = dict()
    c = 0
    for row in rd:
        # print(type(d))
        # print(row)




        try:

            if(row[3]=='5' and row[2]=='False'):
                print(row)
                # break

            # if(row[0]=='10657'):
            #     print(row)

            if(row[3] in d.keys()):
                if(row[2] in d[row[3]].keys()):
                    d[row[3]][row[2]]+=1
                else:
                    d[row[3]][row[2]]=1
            else:
                d[row[3]] = dict()
                d[row[3]][row[2]] = 1
            # print("Success !!!!!")
        except Exception as e:
            print(e)
            print(row)
            # c+=1
            # if(c==10):
            #     break
    print(d)
    print(d.keys())
    c = 0
    m = 0
    s = set()
    for k in d.keys():
        print(k,d[k])
        m = max(m,int(k))
        s.add(int(k))
        print(d[k].keys())
        if('False' in d[k].keys()):
            c+=d[k]['False']
        if('True' in d[k].keys()):
            c+=d[k]['True']
    print(c)
    print(m)
    print(sorted(s))
