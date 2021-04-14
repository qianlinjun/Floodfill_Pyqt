import os
import sys



def val(path):
    with open(path, "r") as f:
        resList = f.readlines()
        for res in resList:
            res = res.strip()
            



if __name__ == "__main__":
    global_contour = np.array([[[0,0]],[[0,1024]],[[1024,1024]],[[1024,0]]])
    TestContourValid(global_contour)
