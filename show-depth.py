
import matplotlib.pyplot as plt
from PIL import Image
import sys

img_path = sys.argv[1]
img1 = Image.open(img_path)
# img2 = matplotlib.image.imread('data/membrane/train/image/ID_0000_Z_0142.png')
plt.figure()
# plt.subplot(1,2,1)
plt.imshow(img1)
# plt.subplot(1,2,2)
# plt.imshow(img2)
plt.show()