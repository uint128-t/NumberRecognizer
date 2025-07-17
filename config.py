TRAINING_IMAGES="train-images.idx3-ubyte"
TRAINING_LABELS="train-labels.idx1-ubyte"
TEST_IMAGES="t10k-images.idx3-ubyte"
TEST_LABELS="t10k-labels.idx1-ubyte"
WEIGHTS="weights"
RECOGNIZE="pict.data"

LEARNING_RATE=0.001
HIDDEN_SIZES=[256,128,64]

def act(x):
    return max(0.0,x)
def act_d(x):
    return (0.0,1.0)[x>0]

def print_drawing(img):
    print("|",end="")
    for i in range(28*28):
        v=int(img[i])
        print(end=f"\x1b[48;2;{v};{v};{v}m  ")
        if i%28==27 and i!=28*28-1:
            print("\x1b[0m|",end="\n|")
    print("\x1b[0m|")