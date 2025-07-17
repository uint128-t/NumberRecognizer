from neuralnetwork import Layer
from config import *
import neuralnetwork
import struct
import math
rec=open(RECOGNIZE,"rb")
weights=open(WEIGHTS,"rb")
LAYERSZ=28*28
OUTSZ=10

def nextweight():
    wt=weights.read(8)
    if len(wt)==8:
        return struct.unpack('d',wt)[0]
    else:
        raise ValueError("Not enough weights in file")
neuralnetwork.nextweight=nextweight

layers:list[Layer]=[]
layers.append(Layer(LAYERSZ,0)) # Input
pv=LAYERSZ
for sz in HIDDEN_SIZES:
    layers.append(Layer(sz,pv))
    pv=sz
layers.append(Layer(OUTSZ,pv))

if weights.read(1):
    raise ValueError("Extra weights in file")

values=rec.read(28*28)
print_drawing(values)
layers[0].outputs=[x/255 for x in values]

for li in range(1,len(layers)):
    layers[li].compute(layers[li-1])
maxv=max(layers[-1].sums)
maxi=layers[-1].sums.index(maxv)
te=sum(math.e**v for v in layers[-1].sums)
print(f"Recognized as {maxi} with confidence {math.e**maxv/te:.2f}")