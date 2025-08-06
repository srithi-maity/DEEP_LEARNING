import numpy as np


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
    exp = [np.exp(i) for i in x]
    exp_sum = np.sum(exp)
    res=[val/exp_sum for val in exp]
    return res

def compute_multiplication(w,x):
    return np.dot(w,x)

def compute_multiplication_derivative(x,w):
    return w,x

# def compute_addition(ip):#here we'll give the list or np array
#     return np.sum(ip)

def compute_addition(w,x):
    return w+x

# def compute_subtraction(w,x):
#     return w-x
#
# def compute_division(w,x):
#     return w/x
def compute_max_function(w,x):
    return np.max(w)

def derivative_addition_func(gl_grad):
    return gl_grad
def derivative_multiplication_func(local_grad,gl_grad):
    return local_grad*gl_grad
def derivative_max_func(local_grad):
    if local_grad > 0:
        return local_grad
    else:
        return 0

def main():
    print(f"=============FORWARD PASS=================")
    x1=2
    w1=3
    x2=4
    w2=5
    f1=compute_multiplication(w1,x1)
    f2=compute_multiplication(w2,x2)
    f3=compute_addition(f1,f2)
    print(f"output = {f3}")

    print(f"================BACKWARD====================")
    gl_grad=1

    df3_df2=derivative_addition_func(gl_grad)
    print(f"df3_df2={df3_df2}")
    df3_df1=derivative_addition_func(gl_grad)
    print(f"df3_df1={df3_df1}")
    df1_dx1,df1_dw1=compute_multiplication_derivative(x1,w1)
    df3_dx1=derivative_multiplication_func(df3_df1,df1_dx1)
    print(f"df3_dx1={df3_dx1}")
    df3_dw1=derivative_multiplication_func(df3_df1,df1_dw1)
    print(f"df3_dw1={df3_dw1}")
    df2_dx2,df2_dw2=compute_multiplication_derivative(x2,w2)
    df3_dx2=derivative_multiplication_func(df3_df2,df2_dx2)
    print(f"df3_dx2={df3_dx2}")
    df3_dw2=derivative_multiplication_func(df3_df2,df2_dw2)
    print(f"df3_dw2={df3_dw2}")



if __name__ == "__main__":
    main()