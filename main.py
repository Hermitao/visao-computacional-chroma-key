import cv2
import numpy as np

tresh = 190

background = cv2.imread("./Imagens/background-bonus-3.bmp")

def alter_image(in_filename, out_filename):
    image = cv2.imread(in_filename)

    width = image.shape[1]
    height = image.shape[0]
    dim = (width, height)
    background_resized = cv2.resize(background, dim)

    img_c = image.copy()

    i = 0
    for row in img_c:
        j = 0
        for col in row:
            # checa se o pixel é mais verde do que qualquer outro canal, e ainda acima de um treshold
            if col[1] > col[0] and col[1] > col[2] and col[1] > tresh:
                img_c[i][j] = (0, 0, 0)
                img_c[i][j] = background_resized[i][j]
            j+=1
        i+=1

    out_filepath = f"./img-{out_filename}/{out_filename}.bmp"
    cv2.imwrite(out_filepath, img_c)

alter_image("./Imagens/corvo.bmp", "corvo")
alter_image("./Imagens/corvos.bmp", "corvos")
alter_image("./Imagens/formas.bmp", "formas")
alter_image("./Imagens/rainha.bmp", "rainha")
