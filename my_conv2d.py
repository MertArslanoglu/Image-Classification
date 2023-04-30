import numpy as np
from utils import part2Plots

def my_conv2d(input, kernels):
    # extracting the shapes of the kernels and frames:
    size_input = input.shape
    size_kernel = kernels.shape
    frame_number = size_input[0]
    channel_number = size_kernel[0]

    #initializing the output for stride of 1
    output = np.zeros((frame_number, channel_number, size_input[2] - size_kernel[2] + 1, size_input[3] - size_kernel[3] + 1)) #stride 1

    for fn in range(frame_number): #iteration over images
        image = np.reshape(input[fn], (input.shape[2], input.shape[3])) # reshaping image to get rid of the third dimension as 0
        for ch in range(channel_number): #iteration over kernel channels
            kernel = np.reshape(kernels[ch], (size_kernel[2], size_kernel[3])) # reshaping kernel to get rid of the third dimension as 0
            # changing the upper left corner of the kernel on image
            for window_x in range(size_input[2] - size_kernel[2] + 1):
                for window_y in range(size_input[3] - size_kernel[3] + 1):
                    holder = 0
                    # convolution between the image and the shifted kernel
                    for x in range(size_kernel[2]):
                        for y in range(size_kernel[3]):
                            holder += kernel[x][y] * image[window_x + x][window_y + y] # compute convolution cumulatively
                    output[fn][ch][window_x][window_y] = holder # assign the output pixel value

    return output

input = np.load("samples_7.npy")
kernel = np.load("kernel.npy")
out = my_conv2d(input, kernel)
part2Plots(out, save_dir="C://Users//mertf//PycharmProjects//homework_1", filename="visuals")
