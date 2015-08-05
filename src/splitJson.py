import json
import os.path
import sys

f = open('training.json', 'r')
num = f.readline()
cnt=0
for line in f:
    cnt+=1
    try:
        js = json.loads(line, strict=False)
        dir = js["topic"]
        if not os.path.exists(dir):
            os.makedirs(dir)

        sf = open(dir + '/' + str(cnt) + '.json', 'w')
        q = js["question"].encode('ascii', 'replace')

        sf.writelines(q)
        sf.write('\n')
        exc = js["excerpt"].encode('ascii', 'replace')
        sf.write(exc)
        sf.close()
    except TypeError as e:
        print "Unexpected error [" + str(cnt) + "]", e
        print q
        raise
f.close()


