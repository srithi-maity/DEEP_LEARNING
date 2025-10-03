import numpy as np
from random import random

def relu_function(x):
    return np.maximum(0, x)

def relu_function_derivative(x):
    # return np.minimum(1,r)
    if x > 0:
        res=1
    else:
        res=0
    return res

def softmax(x):
    x = np.array(x, ndmin=1)  # ensures x is array (even scalar becomes 1-element array)
    exp = []
    for i in x:
        exp.append(np.exp(i))
    exp_sum = np.sum(exp)
    res = []
    for exp_val in exp:
        res.append(exp_val / exp_sum)
    return res

def apply_activation(choice, z):
    if choice == '1':
        return relu_function(z)
    elif choice == '2':
        return relu_function_derivative(z)
    elif choice == '3':
        return softmax(z)  # use only for output layer
    else:
        print("Invalid choice. Using ReLU by default.")
        return relu_function(z)


def activation_function_menu():
    print("Choose Activation Function:")
    print("1. ReLU")
    print("2. ReLU Derivative")
    print("3. Softmax (used only in output layer)")

    choice = input("Enter choice [1/2/3]: ").strip()
    return choice





def matrix_multiplication(w, a):
    return np.dot(w, a)

def default_value(n):
    return [random() for _ in range(n)]

def getting_val(n, default=True):
    val = []
    if default:
        val = default_value(n)
    else:
        for i in range(1, n + 1):
            v = float(input(f"Give the {i}th value: "))
            val.append(v)
    return val

def layer_cal():
    in_no = int(input("Number of input features: "))

    default_flag = input("Do you want to use default (random) values? [y/n]: ").lower() == 'y'

    a_val = getting_val(in_no, default_flag)
    w_no = in_no

    layer_no = int(input("Number of hidden layers: "))
    for i in range(1, layer_no + 1):
        print(f"\n========== Layer {i} ==========")
        n = int(input(f"Number of neurons in layer {i}: "))
        a_vec = []
        for v in range(1, n + 1):
            print(f"\nLayer {i}, Neuron {v}:")
            b = float(input("Bias term: ")) if not default_flag else random()
            w = getting_val(w_no, default_flag)
            z = matrix_multiplication(w, a_val) + b
            print(f"z = {z}")

            a = apply_activation(activation_function_menu(),z)
            a_vec.append(a)
            print(f"a = {a}")
        a_val = a_vec
        w_no = n

    print("\n======= Output Layer =======")
    n = int(input("Number of output neurons: "))
    z_final = []
    for m in range(1, n + 1):
        print(f"\nOutput Neuron {m}:")
        w = getting_val(w_no, default_flag)
        b = float(input("Bias term: ")) if not default_flag else random()
        z = matrix_multiplication(w, a_val) + b
        print(f"z = {z}")
        z_final.append(z)

    a_final = apply_activation(activation_function_menu(),z)
    return a_final

def main():
    result = layer_cal()
    print("\n===== Final Output (XOR) =====")
    print(result)

if __name__ == "__main__":
    main()