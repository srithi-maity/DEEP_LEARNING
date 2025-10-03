import numpy as  np

def relu_function(x):
    return np.maximum(0,x)

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

def matrix_multiplication(w,a):

    return np.dot(a,w)

def getting_val(n):
    val=[]
    for i in range (1,n+1):
        v=float(input(f"give the {i}th value :"))
        val.append(v)
    return val

def layer_cal():

    in_no=int(input(f"give the number of initial input you want to give :"))
    a_val=getting_val(in_no)
    w_no = in_no

    layer_no=int(input(f"give the number of hidden layer you want :"))
    for i in range(1,layer_no+1):
        print(f"============for {i}th layer =============")
        n=int(input(f"how many neuron you want in this layer: "))
        a_vec=[]
        for v in range (1,n+1):
            print(f"for {i}th hidden layer {v}th neuron ************* :")
            b=float(input("give the bias term of this neuron : "))
            w=getting_val(w_no)
            z=matrix_multiplication(w,a_val)+b
            print(f"z_value for {i}th layer {v}th neuron is :{z} ")
            a=relu_function(z)
            a_vec.append(a)
            print(f"the a_value for {i}th layer {v}th neuron is :{a}")
        print(f"the a_value for {i}th layer is : {a_vec}")
        a_val = a_vec
        w_no = n
    print(f"-----------------------------------------------------------")

    o_in = int(input(f"how many neuron you want in output layer: "))
    a_final = []
    for m in range(1, o_in + 1):
        w = getting_val(w_no)
        z = matrix_multiplication(w, a_val)
        a_res = softmax(z)
        a_final.append(a_res)
    return a_final



def main():
    result=layer_cal()
    print(f"the final output : {result}")



if __name__ == "__main__":
    main()