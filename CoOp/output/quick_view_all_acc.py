import os
import numpy as np

tots = []
for i in range(1, 100):
    try:
        f = open(os.path.join("seed{}".format(i), "log.txt"))
        items = f.readlines()
        f.close()
        for item in items:
            if item.startswith("* accuracy"):
                break
        tot = float(item.split(": ")[-1][:-2])
        print(tot)
        tots.append(tot)
    except:
        pass
tots = np.asarray(tots)
print("Average acc on {} trials = {}".format(len(tots), np.mean(tots)))
