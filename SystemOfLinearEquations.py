import numpy as np
#Using NumPy package to set up the arrays corresponding to the system of linear equations

#Tasks going to be performed
#1.)Finding Determinant of a matrix and find the solutions of the system with Numpy linear algebra pacakge
#2.)Performing Row redcution to bring matrix into row echelon form.
#3.)Finding the solution for the system of linear equation using row reduced matrix.
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Constructing matrix A and vector b corresponding to linear equations
A = np.array([     
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]    
    ], dtype=np.dtype(float)) 
b = np.array([6, 3, 14, 8], dtype=np.dtype(float))
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Find the determinant d of matrix A and the solution vector x
d = np.linalg.det(A)
x = np.linalg.solve(A, b)
print(f"Determinant of matrix A: {d:.2f}")
print(f"Solution vector: {x}")

#Expected results
"""Determinant of matrix A: -17.00
Solution vector: [2. 3. 4. 1.]"""
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Elementray Operations and Row Reduction
def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()     
    # exchange row_num of the matrix M_new with its multiple by row_num_multiple
    # Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
    M_new[row_num] = M_new[row_num]*row_num_multiple 
    return M_new
    
def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()     
    # multiply row_num_1 by row_num_1_multiple and add it to the row_num_2, 
    # exchanging row_num_2 of the matrix M_new with the result
    M_new[row_num_2] = (M_new[row_num_1]*row_num_1_multiple)+M_new[row_num_2]
    return M_new

def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()     
    # exchange row_num_1 and row_num_2 of the matrix M_new
    M_new[[row_num_1,row_num_2]] =M_new[[row_num_2,row_num_1]] 
    return M_new

 #Expected Results
#Original matrix:
"""[[ 1 -2  3 -4]
 [-5  6 -7  8]
 [-4  3 -2  1]
 [ 8 -7  6 -5]]

Original matrix after its third row is multiplied by -2:
[[ 1 -2  3 -4]
 [-5  6 -7  8]
 [ 8 -6  4 -2]
 [ 8 -7  6 -5]]

Original matrix after exchange of the third row with the sum of itself and first row multiplied by 4:
[[  1  -2   3  -4]
 [ -5   6  -7   8]
 [  0  -5  10 -15]
 [  8  -7   6  -5]]

Original matrix after exchange of its first and third rows:
[[-4  3 -2  1]
 [-5  6 -7  8]
 [ 1 -2  3 -4]
 [ 8 -7  6 -5]]"""   
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Row reductions operations
def augmented_to_ref(A, b):    
    # stack horizontally matrix A and vector b, which needs to be reshaped as a vector (4, 1)
    A_system =  np.hstack((A, b.reshape((4, 1))))

    
    # swap row 0 and row 1 of matrix A_system
    A_system[[0, 1]] = A_system[[1, 0]]
    

    # multiply row 0 of the new matrix A_ref by -2 and add it to the row 1
    A_system[[1]] = (A_system[[0]]*-2)+A_system[[1]]
    
    # add row 0 of the new matrix A_ref to the row 2, replacing row 2
    A_system[[2]] = A_system[[0]]+A_system[2]
    
    # multiply row 0 of the new matrix A_ref by -1 and add it to the row 3
    A_system[[3]]=(A_system[[0]]*-1)+A_system[[3]]
    
    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_system[[3]]=A_system[[2]]+A_system[[3]]
    
    # swap row 1 and 3 of the new matrix A_ref
    A_system[[1,3]]=A_system[[3,1]]
    
    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_system[[3]]=A_system[[2]]+A_system[[3]]
    
    # multiply row 1 of the new matrix A_ref by -4 and add it to the row 2
    A_system[[2]]=(A_system[[1]]*-4)+A_system[[2]]
    
    # add row 1 of the new matrix A_ref to the row 3, replacing row 3
    A_system[[3]]=A_system[[1]]+A_system[[3]]
    
    # multiply row 3 of the new matrix A_ref by 2 and add it to the row 2
    A_system[[2]]=(A_system[[3]]*2)+A_system[[2]]
    
    # multiply row 2 of the new matrix A_ref by -8 and add it to the row 3
    A_system[[3]]=(A_system[[2]]*-8)+A_system[[3]]
    
    # multiply row 3 of the new matrix A_ref by -1/17
    A_system[[3]]=A_system[[3]]*(-1/17)
    A_ref=A_system
 
    
    return A_ref

A_ref = augmented_to_ref(A, b)

print(A_ref)
"""[[ 1.  2. -1. -1.  3.]
 [ 0.  1.  4.  3. 22.]
 [ 0.  0.  1.  3.  7.]
 [-0. -0. -0.  1.  1.]]"""
#Expected output
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Solutions for system of equation
x_4 =A_ref[[[-1]]]
x_4=x_4[-1]
x_4=x_4[-1]
x_4=int(x_4[3])


# find the value of x_3 from the previous row of the matrix. Use value of x_4.
x_3 = A_ref[[[-2]]]
x_3=x_3[0]
x_3=x_3[0]
x_3=x_3[-3]
x_3=int(7-3*x_4)

# find the value of x_2 from the second row of the matrix. Use values of x_3 and x_4
x_2 =22-(4*x_3)-(3*x_4)

# find the value of x_1 from the first row of the matrix. Use values of x_2, x_3 and x_4
x_1=3-(2*(x_2))+x_3+x_4
print(x_1, x_2, x_3, x_4)
"""2 3 4 1
"""
#final values of system of linear equations after solving
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#Reduce the linear eqaution into matrix reduced diagonal form
def ref_to_diagonal(A_ref):    
    # multiply row 3 of the matrix A_ref by -3 and add it to the row 2
    A_ref[[2]]=A_ref[[3]]*-3+A_ref[[2]]
    
    # multiply row 3 of the new matrix A_diag by -3 and add it to the row 1
    A_ref[[1]]=A_ref[[3]]*-3+A_ref[[1]]
    
    # add row 3 of the new matrix A_diag to the row 0, replacing row 0
    A_ref[[0]]=A_ref[[3]]+A_ref[[0]]
    
    # multiply row 2 of the new matrix A_diag by -4 and add it to the row 1
    A_ref[[1]]=A_ref[[2]]*-4+A_ref[[1]]
    
    # add row 2 of the new matrix A_diag to the row 0, replacing row 0
    A_ref[[0]]=A_ref[[2]]+A_ref[[0]]
    
    # multiply row 1 of the new matrix A_diag by -2 and add it to the row 0
    A_ref[[0]]=A_ref[[1]]*-2+A_ref[[0]]
    A_diag=A_ref
    
    return A_diag
    
A_diag = ref_to_diagonal(A_ref)

print(A_diag)
#Expected output
"""
[[1 0 0 0 2]
 [0 1 0 0 3]
 [0 0 1 0 4]
 [0 0 0 1 1]]"""
 #END OF CODE HERE......................................................






