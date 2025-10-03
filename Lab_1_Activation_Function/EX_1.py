import numpy as np
import matplotlib.pyplot as plt

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return 1-s

def tanh_function(x):
    return (2 / (1 + np.exp(-2*x))) - 1

def tanh_function_derivative(tanh):
    return 1 - tanh**2

def relu_function(x):
    return np.maximum(0,x)

def leaky_relu(x):

    res=[]
    for i in x :
        if i > 0 :
            res.append(np.maximum(i,0.01*i))
        else:
            res.append(np.maximum(0,0.01*i))
    return res

def relu_function_derivative(x):
    # return np.minimum(1,r)
    res=[]
    for i in x :
        if i > 0:
            res.append(1)
        else:
            res.append(0)
    return res

def softmax(x):

    exp=[]
    for i in x :
        exp.append(np.exp(i))
    print(exp)
    exp_sum=np.sum(exp)
    # print(exp_sum)
    res=[]
    for exp_val in exp:
        res.append( exp_val/exp_sum )
    return res

def main():
    # Generating 100 values from -10 to 10
    x = np.linspace(-10, 10, 100)
    print("Input z:", x)

    # activation functions

    #sigmoid function
    s = sigmoid_func(x)
    print("Sigmoid function :",s)

    der_s=sigmoid_derivative(s)
    print("Derivative of sigmoid function :",der_s)
    #tanh function using sigmoid output and without using that
    tanh= tanh_function(x)
    tanh_via_sigmoid = 2 * sigmoid_func(2*x) - 1
    print("Tanh (direct):", tanh)
    print("Tanh (using sigmoid):", tanh_via_sigmoid)

    der_tanh=tanh_function_derivative(tanh)
    print("Derivative of Tanh function :",der_tanh)

    #ReLU (Rectified Linear Unit) Function
    relu= relu_function(x)
    print("RElu function : ",relu)
    der_relu=relu_function_derivative(x)
    print("Derivative of Relu function :",der_relu)

    #leaky Relu function
    l_relu=leaky_relu(x)
    print("Leky relu :",l_relu)

    #Softmax function
    soft_max=softmax(x)
    print("Softmax function :",soft_max)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, s, label="Sigmoid", color='blue',linestyle='solid')
    plt.plot(x, tanh, label="Tanh", color='orange', linestyle='dashed')
    plt.plot(x, relu, label="Relu", color='green', linestyle='dashdot')
    plt.plot(x, soft_max, label="Softmax", color='red', linestyle='dashed')
    plt.plot(x, l_relu, label="Leaky Relu", color='violet', linestyle='dotted')


    plt.xlabel("Input")
    plt.ylabel("Activation Output")
    plt.title("Sigmoid and Tanh Activation Functions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()