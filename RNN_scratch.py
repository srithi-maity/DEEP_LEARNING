import numpy as np


def tanh_function(x):
    return (2 / (1 + np.exp(-2*x))) - 1

def matrix_val(m,n,mat):
    matrix=[]
    for i in range(m):
        row = []
        for j in range(n):
            value = float(input(f"{mat} at position ({i + 1},{j + 1}): "))
            row.append(value)
        matrix.append(row)
    return matrix

def getting_val():
    val=[]

    n=int(input(f"give the dimension :"))
    for i in range (1,n+1):
        v=float(input(f"give the {i}th value :"))
        val.append(v)
    return n, val

def layer_cal():
    in_no = int(input(f"give the number of input you want to give :"))
    input_val=[]
    for no in range(0,in_no):
        print(f"For {no+1} input :")
        dim_x,x_val=getting_val()
        input_val.append(x_val)

    print(f"input_val = {input_val} of dimension = {input_val.shape()}")

    print(f"taking the H0 value :")
    dim_h,H0=getting_val()

    Wxh_row=dim_h
    Wxh_col=dim_x
    Wxh=matrix_val(Wxh_row,Wxh_col,"Wxh")
    print(f" Wxh matrix of {Wxh_row}*{Wxh_col} is : {Wxh}")

    Whh_row=dim_h
    Whh_col=dim_h
    Whh=matrix_val(Whh_row,Whh_col,"Whh")

    print(f" Whh matrix of {Whh_row}*{Whh_col} is : {Whh}")

    out_no=int(input(f"give the dimension of the output :"))

    Why_row=out_no
    Why_col=dim_h
    Why=matrix_val(Why_row,Why_col,"Why")

    layer_no = in_no



    h=H0
    for i in range(1,layer_no+1):
        print(f"============for {i}th RNN Block =============")

        x=input_val[i-1]
        print(f"H t-1 = {h} \n whh= {Whh} \n wxh = {Wxh} \n x = {x} ")
        H_val=(np.dot(Whh,h))+(np.dot(Wxh,x))
        H=tanh_function(H_val)

        print(f"why = {Why} \n H = {H}")
        Y=(np.dot(Why,H))

        print(f" the H[{i}] is : {H} \n the Y[{i}] is :{Y}")
        h=H


def main():
    layer_cal()




if __name__ == "__main__":
    main()