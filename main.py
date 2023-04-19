import cv2
import numpy as np

background = cv2.imread("./Imagens/background.bmp")
img = cv2.imread("./Imagens/corvo.bmp")

filename = "./img-corvo/corvo.bmp"

img_c = img.copy()

i = 0
for row in img_c:
    j = 0
    for col in row:
        if col[1] > 100:
            img_c[i][j] = (0, 0, 0)
        j+=1
    i+=1

cv2.imwrite(filename, img_c)
