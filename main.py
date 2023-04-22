import cv2
import numpy as np
import matplotlib.pyplot as plt

max_green_distance = 30

def alter_image(in_filename, out_filename, thresh = 100, background_path = "./Imagens/background.bmp"):
    background = cv2.imread(background_path)
    background_gray = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
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
            # verifica se o pixel é mais verde do que qualquer outro canal 
            # por uma determinada distância, e ainda com valor acima de um threshold
            if (int(col[1]) - int(col[0]) > max_green_distance) and (int(col[1]) - int(col[2]) > max_green_distance) and col[1] > thresh:
                img_c[i][j] = (0, 0, 0)
                img_c[i][j] = background_resized[i][j]
            j+=1
        i+=1

    out_filepath = f"./img-{out_filename}/{out_filename}.bmp"
    cv2.imwrite(out_filepath, img_c)


    # Anti-aliasing ----------------------

    img_edgy_color = cv2.imread(f"./img-{out_filename}/{out_filename}.bmp")
    #img_rgb = cv2.cvtColor(img_edgy_color, cv2.COLOR_BGR2RGB)
    img_rgb = img_edgy_color.copy()

    img_grayscale = cv2.imread(f"./img-{out_filename}/{out_filename}.bmp", cv2.IMREAD_GRAYSCALE)


    g_kernel_v = np.array([[-1, 0, 1], 
                           [-1, 0, 1],
                           [-1, 0, 1]])
    g_kernel_h = np.array([[-1, -1, -1],
                           [ 0, 0, 0],
                           [1, 1, 1]])

    img_float = img_grayscale.astype(np.float32) / 255
    np.max(img_float)


    result_h = cv2.filter2D(img_float, -1, g_kernel_h)
    result_v = cv2.filter2D(img_float, -1, g_kernel_v)

    thresh_internal = 0.2
    ret,im_bin_v = cv2.threshold(result_v, thresh_internal, 1, cv2.THRESH_BINARY)
    ret,im_bin_h = cv2.threshold(result_h, thresh_internal, 1, cv2.THRESH_BINARY) 

    mask = im_bin_v + im_bin_h
    np.clip(mask, 0, 1, out=mask)

    blur = cv2.GaussianBlur(255 - img_rgb, (25,25), 0)

    mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    img_aa = np.where(mask!=0,blur,(img_rgb / 255).astype(dtype='float32'))

    #cv2.imwrite(f"./img-{out_filename}/{out_filename}_final.bmp", (img_aa * 255).astype(dtype='uint8'))
    cv2.imwrite(f"./img-{out_filename}/{out_filename}_final.bmp", (img_aa * 255).astype(dtype='uint8'))


alter_image("./Imagens/corvo.bmp", "corvo", 210)
alter_image("./Imagens/corvos.bmp", "corvos")
alter_image("./Imagens/formas.bmp", "formas")
alter_image("./Imagens/rainha.bmp", "rainha", 87)

alter_image("./Imagens/corvo.bmp", "corvo", 210, background_path="./Imagens/background-bonus-3.bmp")
alter_image("./Imagens/corvos.bmp", "corvos", background_path="./Imagens/background-bonus-3.bmp")
alter_image("./Imagens/formas.bmp", "formas", background_path="./Imagens/background-bonus-3.bmp")
alter_image("./Imagens/rainha.bmp", "rainha", 87, background_path="./Imagens/background-bonus-3.bmp")
