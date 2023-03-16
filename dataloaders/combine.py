import sys
f1 = open(sys.argv[1],'r')
f2 = open(sys.argv[2],'r')
f3 = open(sys.argv[3],'w')
lines1=f1.readlines()
lines2=f2.readlines()
for i in range(0,len(lines1)):
    txt1 = lines1[i].strip()
    txt2 = lines2[i].strip()
    txt = txt1 + txt2
    f3.write(txt+'\n')
f3.close()
    
    
