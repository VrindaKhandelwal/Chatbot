import pandas as pd

csv = pd.read_csv ("output 2.csv")

data = ""

for i in range(len(csv['input'])):
    inp = str(csv['input'][i])
    if len(inp) <= 200:
        inp = inp.replace ("\n", ". ").replace ("\r", ". ")
        out = str(csv['output'][i])[:200]

        data += inp + "\n"
        data += out + "\n"

data = data[0:-2]

f = open ("chat.txt", "w")
f.write (data)
f.close ()