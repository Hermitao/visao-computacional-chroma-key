# ⠄⠄⠄⠄⠄⠄⢴⡶⣶⣶⣶⡒⣶⣶⣖⠢⡄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⢠⣿⣋⣿⣿⣉⣿⣿⣯⣧⡰⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⣿⣿⣹⣿⣿⣏⣿⣿⡗⣿⣿⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⠟⡛⣉⣭⣭⣭⠌⠛⡻⢿⣿⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⠄⠄⣤⡌⣿⣷⣯⣭⣿⡆⣈⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⠄⢻⣿⣿⣿⣿⣿⣿⣿⣷⢛⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⠄⠄⢻⣷⣽⣿⣿⣿⢿⠃⣼⣧⣀⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⣛⣻⣿⠟⣀⡜⣻⢿⣿⣿⣶⣤⡀⠄⠄⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⠄⠄⢠⣤⣀⣨⣥⣾⢟⣧⣿⠸⣿⣿⣿⣿⣿⣤⡀⠄⠄⠄
# ⠄⠄⠄⠄⠄⠄⠄⠄⢟⣫⣯⡻⣋⣵⣟⡼⣛⠴⣫⣭⣽⣿⣷⣭⡻⣦⡀⠄
# ⠄⠄⠄⠄⠄⠄⠄⢰⣿⣿⣿⢏⣽⣿⢋⣾⡟⢺⣿⣿⣿⣿⣿⣿⣷⢹⣷⠄
# ⠄⠄⠄⠄⠄⠄⠄⣿⣿⣿⢣⣿⣿⣿⢸⣿⡇⣾⣿⠏⠉⣿⣿⣿⡇⣿⣿⡆
# ⠄⠄⠄⠄⠄⠄⠄⣿⣿⣿⢸⣿⣿⣿⠸⣿⡇⣿⣿⡆⣼⣿⣿⣿⡇⣿⣿⡇
# ⠇⢀⠄⠄⠄⠄⠄⠘⣿⣿⡘⣿⣿⣷⢀⣿⣷⣿⣿⡿⠿⢿⣿⣿⡇⣩⣿⡇
# ⣿⣿⠃⠄⠄⠄⠄⠄⠄⢻⣷⠙⠛⠋⣿⣿⣿⣿⣿⣷⣶⣿⣿⣿⡇⣿⣿⡇

import cv2
import numpy as np
import matplotlib.pyplot as plt

max_green_distance = 30

def alter_image(in_filename, out_filename, thresh = 100, background_path = "./Imagens/background.bmp", prefix=""):
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
                img_c[i][j] = background_resized[i][j]
            j+=1
        i+=1

    out_filepath_no_extension = f"./img-{out_filename}/{prefix}{out_filename}"
    out_filepath = f"{out_filepath_no_extension}.bmp"

    cv2.imwrite(out_filepath, img_c)

    # Anti-aliasing ----------------------

    img_edgy_color = cv2.imread(f"{out_filepath_no_extension}.bmp")
    #img_rgb = cv2.cvtColor(img_edgy_color, cv2.COLOR_BGR2RGB)
    img_rgb = img_edgy_color.copy()

    img_grayscale = cv2.imread(f"{out_filepath_no_extension}.bmp", cv2.IMREAD_GRAYSCALE)


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

    blur1 = cv2.GaussianBlur(255 - img_rgb, (25,25), 0)
    blur2 = cv2.GaussianBlur(255 - img_rgb, (75,75), 0)
    blur3 = cv2.GaussianBlur(255 - img_rgb, (101,101), 0)

    cv2.imwrite(f"{out_filepath_no_extension}_blur1.bmp", 255 - blur1)
    cv2.imwrite(f"{out_filepath_no_extension}_blur2.bmp", 255 - blur2)
    cv2.imwrite(f"{out_filepath_no_extension}_blur3.bmp", 255 - blur3)

    mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    img_aa1 = np.where(mask!=0,blur1,(img_rgb / 255).astype(dtype='float32'))
    img_aa2 = np.where(mask!=0,blur2,(img_rgb / 255).astype(dtype='float32'))
    img_aa3 = np.where(mask!=0,blur3,(img_rgb / 255).astype(dtype='float32'))

    cv2.imwrite(f"{out_filepath_no_extension}_final1.bmp", (img_aa1 * 255).astype(dtype='uint8'))
    cv2.imwrite(f"{out_filepath_no_extension}_final2.bmp", (img_aa2 * 255).astype(dtype='uint8'))
    cv2.imwrite(f"{out_filepath_no_extension}_final3.bmp", (img_aa3 * 255).astype(dtype='uint8'))


alter_image("./Imagens/corvo.bmp", "corvo", 210)
alter_image("./Imagens/corvos.bmp", "corvos")
alter_image("./Imagens/formas.bmp", "formas")
alter_image("./Imagens/rainha.bmp", "rainha", 87)

alter_image("./Imagens/corvo.bmp", "corvo", 210, background_path="./Imagens/background-bonus.bmp", prefix="bonus_")
alter_image("./Imagens/corvos.bmp", "corvos", background_path="./Imagens/background-bonus-4.bmp", prefix="bonus_")
alter_image("./Imagens/formas.bmp", "formas", background_path="./Imagens/background-bonus-2.bmp", prefix="bonus_")
alter_image("./Imagens/rainha.bmp", "rainha", 87, background_path="./Imagens/background-bonus-3.bmp", prefix="bonus_")
