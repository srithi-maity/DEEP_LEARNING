import numpy as np


def valid_conv(image_height, filter_height, image_width, filter_width):
    return image_height - filter_height + 1, image_width - filter_width + 1


def same_conv(image_height, filter_height, image_width, filter_width, padding):
    return image_height + (2 * padding) - filter_height + 1, image_width + (2 * padding) - filter_width + 1


def strided_conv(image_height, filter_height, image_width, filter_width, stride, padding):
    return ((image_height + (2 * padding) - filter_height) // stride) + 1, ((image_width + (2 * padding) - filter_width) // stride) + 1


def Choose_conv():
    print("Choose Convolution Operation:")
    print("1. Valid Convolution")
    print("2. Same Convolution")
    print("3. Strided Convolution")

    choice = input("Enter choice [1/2/3]: ").strip()
    return choice

def stride_pad_val(filter_height):
    choice = Choose_conv()
    if choice == "1":  # Valid Convolution
        return 1, 0
    elif choice == "2":  # Same Convolution
        return 1, (filter_height - 1) // 2
    elif choice == "3":  # Strided Convolution
        stride = int(input("Enter stride value: "))
        padding = int(input("Enter padding value: "))
        return stride, padding
    else:
        print("Invalid choice!")
        return 1, 0


def apply_convolution(image, filters, stride=1, padding=0):
    num_channels, img_h, img_w = image.shape
    num_filters, _, kernel_h, kernel_w = filters.shape

    # Apply padding
    if padding > 0:
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

    out_h = (img_h + 2 * padding - kernel_h) // stride + 1
    out_w = (img_w + 2 * padding - kernel_w) // stride + 1

    output = np.zeros((num_filters, out_h, out_w))

    for f in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                region = image[:, i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
                output[f, i, j] = np.sum(region * filters[f])
    return output


def main1():
    channels = int(input("Enter number of channels in image: "))
    height = int(input("Enter image height: "))
    width = int(input("Enter image width: "))

    # Generate random image (channels x height x width)
    image = np.random.randn(channels, height, width)


    num_filters = int(input("Enter number of filters: "))
    filter_size = int(input("Enter filter size : "))
    choice = int(input("Choose Convolution Type: \n1. Valid \n2. Same \n3. Strided \nEnter choice: "))


    # Generate filters (num_filters x channels x kernel_h x kernel_w)
    filters = np.zeros((num_filters, channels, filter_size, filter_size))

    # Take filter values from user
    for f in range(num_filters):
        print(f"\nEnter values for Filter {f + 1}:")
        for c in range(channels):
            print(f"Channel {c + 1}:")
            for i in range(filter_size):
                for j in range(filter_size):
                    filters[f, c, i, j] = float(input(f"Enter value for [{i},{j}] in Channel {c + 1}: "))

    # stride and padding based on choice
    if choice == 1:
        stride = 1
        padding = 0
        print("\nPerforming Valid Convolution...")
    elif choice == 2:
        stride = 1
        padding = (filter_size - 1) // 2
        print("\nPerforming Same Convolution...")
    else:
        stride = int(input("Enter stride value: "))
        padding = 0
        print("\nPerforming Strided Convolution...")

    #Apply convolution operation
    output = apply_convolution(image, filters, stride=stride, padding=padding)


    print("\nInput Image Shape:", image.shape)
    print("Filters Shape:", filters.shape)
    print("Output Shape (num_filters x height x width):", output.shape)



    print("\nFinal Output Feature Maps:\n", output)



if __name__ == "__main__":
    main1()

