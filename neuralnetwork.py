from config import *
import random
class Layer:
    def __init__(self,size:int,psize:int):
        self.size=size
        self.psize=psize
        self.weights=[nextweight() for _ in range(size*psize)]
        self.biases=[nextweight() for _ in range(size)]
        self.sums=[0.0]*size
        self.outputs=[0.0]*size
        self.errorterm=[0.0]*size
        self.weightdv=[0.0]*size*psize
    def compute(self,pvlr:"Layer"):
        assert self.psize==pvlr.size
        for i in range(self.size):
            self.sums[i]=self.biases[i]
            for j in range(self.psize):
                self.sums[i]+=pvlr.outputs[j]*self.weights[i*self.psize+j]
            self.outputs[i]=act(self.sums[i])
    def backprop(self,nlr:"Layer"):
        # Using the next layer's error terms, calculate this layer's nodes' error terms
        assert nlr.psize==self.size
        for i in range(self.size):
            # How much this node's output will influence the loss
            self.errorterm[i]=0
            for j in range(nlr.size):
                self.errorterm[i]+=nlr.errorterm[j]*nlr.weights[j*self.size+i]
            # Multiply by the next nodes' influence (error term)
            self.errorterm[i]*=act_d(self.sums[i])
    def derv(self,pvlr:"Layer"):
        assert self.psize==pvlr.size
        for i in range(self.size):
            for j in range(pvlr.size):
                # If I tweak this weight, the previous outputs remain constant and this will affect the next nodes
                self.weightdv[i*pvlr.size+j]=self.errorterm[i]*pvlr.outputs[j]
    def update(self,rate:float):
        for i in range(self.size*self.psize):
            self.weights[i]-=rate*self.weightdv[i]
        for i in range(self.size):
            self.biases[i]-=rate*self.errorterm[i]
    def __repr__(self):
        return f"<Layer size={self.size} psize={self.psize}>"
    
def nextweight():
    return random.uniform(-0.1, 0.1)