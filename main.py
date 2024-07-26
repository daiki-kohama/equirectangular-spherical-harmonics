import math
import cv2
import numpy as np
import os
import argparse
from concurrent import futures
from src.equirectangular import get_color ,random_on_sphere
from src.spherical_harmonics import knm, Ynm


def calculate_parameters(img, sample_num, divide_number, NumMM, MMax):
    Chat = np.zeros((NumMM, 3))
    for i in range(sample_num):
        theta, phi = random_on_sphere()
        u = (phi / (2 * math.pi))
        v = (theta / math.pi)
        sampleCol = get_color(img, u, v)
        for k in range(0, NumMM):
            n, m = knm(k, MMax)
            Chat[k] += 2 * math.pi * sampleCol * Ynm(n, m, theta, phi) / divide_number
    return Chat


def reconstruct(thetas, phis, Chat, NumMM, MMax):
    converted_img = np.zeros((len(thetas), len(phis), 3), np.uint8)
    for h, theta in enumerate(thetas):
        for w, phi in enumerate(phis):
            color = np.zeros(3)
            for k in range(0, NumMM):
                n, m = knm(k, MMax)
                color += Chat[k] * Ynm(n, m, theta, phi)
            converted_img[h, w] = np.maximum(np.minimum(color, 255), 0)
    return converted_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image-path", required=False, type=str, default="sample.jpg", help="input equirectangular image path")
    parser.add_argument("--resize-ratio", required=False, type=float, default=0.3, help="resize ratio of input image")
    parser.add_argument("--sample-num", required=False, type=int, default=20000, help="number of samples")
    parser.add_argument("--NumMM", required=False, type=int, default=144, help="number of spherical harmonics")
    parser.add_argument("--MMax", required=False, type=int, default=11, help="maximum order of spherical harmonics")
    parser.add_argument("--H", required=False, type=int, default=200, help="height of output image")
    parser.add_argument("--W", required=False, type=int, default=400, help="width of output image")
    parser.add_argument("--multi-process", "-m", action="store_true", help="use multi process")
    args = parser.parse_args()

    assert args.NumMM <= (args.MMax + 1) ** 2, "NumMM must be less than or equal to (MMax + 1) ** 2"

    img = cv2.imread(args.input_image_path)
    resize_ratio = args.resize_ratio
    img = cv2.resize(img, (int(img.shape[1]*resize_ratio), int(img.shape[0]*resize_ratio)))

    ##### embedding #####
    # parameters
    sample_num = args.sample_num
    NumMM = args.NumMM
    MMax = args.MMax

    if not args.multi_process:
        # single process
        Chat = calculate_parameters(img, sample_num, sample_num, NumMM, MMax)
    else:
        # multi process
        cpu_count = os.cpu_count() - 1
        splited_sample_num = sample_num // cpu_count
        sample_num = splited_sample_num * cpu_count
        imgs = [img] * cpu_count
        future_list = []
        with futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            for i in range(cpu_count):
                future = executor.submit(calculate_parameters, imgs[i], splited_sample_num, sample_num, NumMM, MMax)
                future_list.append(future)
        Chat = np.zeros((NumMM, 3))
        for future in future_list:
            Chat += future.result()

    print(Chat)
    number_of_original_parameters = img.shape[0] * img.shape[1] * 3
    print(f"number of original parameters: {number_of_original_parameters}")
    number_of_sh_weights = Chat.shape[0] * Chat.shape[1]
    print(f"number of spherical harmonics weights: {number_of_sh_weights}")
    compression_rate = number_of_sh_weights / number_of_original_parameters
    print(f"compression rate: {compression_rate}")

    ##### reconstruction #####
    H, W = 200, 400
    reconstructed_img = np.zeros((H, W, 3), np.uint8)
    u = np.arange(W) / W
    v = np.arange(H) / H
    thetas = math.pi * v
    phis = 2 * math.pi * u

    if not args.multi_process:
        # single process
        reconstructed_img = reconstruct(thetas, phis, Chat, NumMM, MMax)
    else:
        # multi process
        dim_divide = int(math.sqrt(cpu_count))
        h_range = H // dim_divide
        w_range = W // dim_divide
        future_list = []
        with futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            for hi in range(dim_divide):
                for wi in range(dim_divide):
                    thetas_upper_index = (hi + 1) * h_range
                    if hi == dim_divide - 1:
                        thetas_upper_index = H
                    thetas_divided = thetas[hi*h_range:thetas_upper_index]
                    phis_upper_index = (wi + 1) * W // dim_divide
                    if wi == dim_divide - 1:
                        phis_upper_index = W
                    phis_divided = phis[wi*w_range:phis_upper_index]
                    future = executor.submit(reconstruct, thetas_divided, phis_divided, Chat, NumMM, MMax)
                    future_list.append(future)
        for hi in range(dim_divide):
            for wi in range(dim_divide):
                thetas_upper_index = (hi + 1) * h_range
                if hi == dim_divide - 1:
                    thetas_upper_index = H
                phis_upper_index = (wi + 1) * W // dim_divide
                if wi == dim_divide - 1:
                    phis_upper_index = W
                reconstructed_img[hi*h_range:thetas_upper_index, wi*w_range:phis_upper_index] = future_list[hi*dim_divide+wi].result()

    ##### show #####
    cv2.imshow("original image", img)
    cv2.imshow("reconstructed image", reconstructed_img)
    cv2.imwrite(os.path.join("output", f"original_resizeratio_{resize_ratio}.jpg"), img)
    cv2.imwrite(os.path.join("output", f"resizeratio_{resize_ratio}_samplenum_{sample_num}_NumMM_{NumMM}_MMax_{MMax}.jpg"), reconstructed_img)
    cv2.waitKey(0)
