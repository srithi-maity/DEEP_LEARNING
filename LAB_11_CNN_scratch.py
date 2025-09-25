import numpy as np


def valid_conv(image_height,filter_height,image_width,filter_width):
    output_height = image_height - filter_height + 1 # o/p size= (n-f+1) * (n-f+1)
    output_width = image_width - filter_width + 1
    return output_height,output_width

def same_conv(image_height,filter_height,image_width,filter_width,padding):
    output_height = image_height + (2*padding) - filter_height + 1 # o/p size= (n+2p-f+1) * (n+2p-f+1)
    output_width = image_width + (2*padding) - filter_width + 1
    return output_height,output_width

def strided_conv(image_height, filter_height, image_width, filter_width, stride,padding):
    output_height = ((image_height +(2*padding)- filter_height) // stride) + 1 #o/p size= ([(n+2p-f)//s]+1) * ([(n+2p-f)//s]+1)
    output_width = ((image_height +(2*padding)- filter_height) // stride) + 1
    return output_height, output_width

def Choose_conv():
    print("Choose Convolution Operation:")
    print("1. Valid Convolution")
    print("2. Same Convolution")
    print("3. Strided Convolution")

    choice = input("Enter choice [1/2/3]: ").strip()
    return choice

def perform_convolution(image, filter_matrix, stride=1, padding=0):

    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    image_height, image_width = image.shape
    filter_height, filter_width = filter_matrix.shape

    output_height = ((image_height - filter_height) // stride) + 1
    output_width = ((image_width - filter_width) // stride) + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = image[i * stride:i * stride + filter_height, j * stride:j * stride + filter_width]
            output[i, j] = np.sum(region * filter_matrix)

    return output

def main():
    # Random image and filter
    image = np.random.rand(32, 32)  # 32x32 input image
    filter_matrix = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])  # 3x3 vertical edge detection filter

    image_height, image_width = image.shape
    filter_height, filter_width = filter_matrix.shape

    choice = Choose_conv()

    if choice == "1":
        print("\nPerforming Valid Convolution...")
        padding = 0
        stride = 1
        output_height, output_width = valid_conv(image_height, filter_height, image_width, filter_width)
        print(f"Output Shape: {output_height} x {output_width}")
        output = perform_convolution(image, filter_matrix, stride=stride, padding=padding)
        print(f"The Output is:")
        print(output)
        print(output.shape)

    elif choice == "2":
        print("\nPerforming Same Convolution...")
        padding = (filter_height - 1) // 2  # To maintain same size
        stride = 1
        output_height, output_width = same_conv(image_height, filter_height, image_width, filter_width, padding)
        print(f"Output Shape: {output_height} x {output_width}")
        output = perform_convolution(image, filter_matrix, stride=stride, padding=padding)
        print(f"The Output is:")
        print(output)
        print(output.shape)

    elif choice == "3":
        print("\nPerforming Strided Convolution...")
        stride = int(input("Enter stride value: "))
        padding = int(input("Enter padding value: "))
        output_height, output_width = strided_conv(image_height, filter_height, image_width, filter_width, stride, padding)
        print(f"Output Shape: {output_height} x {output_width}")
        output = perform_convolution(image, filter_matrix, stride=stride, padding=padding)
        print(f"The Output is:")
        print(output)
        print(output.shape)

    else:
        print("Invalid choice! Exiting...")








if __name__ =="__main__":
    main()