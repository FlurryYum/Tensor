import numpy as np
import tensorly as tl
import copy
import cv2
import matplotlib.pyplot as plt
import os, fnmatch

from tensorly.decomposition import parafac, tucker, non_negative_tucker

# create a random 10x10x10 tensor
tensor = abs(np.random.randint(10, 20, size = (2,3,4)))
tensor = tensor.astype(float)

tensor_1 = np.array([tensor[i,:,:].reshape(-1, order = 'F') for i in range(tensor.shape[0])]) #unfolded mode 1
tensor_2 = np.array([tensor[:,i,:].reshape(-1, order = 'F') for i in range(tensor.shape[1])]) #unfolded mode 2
tensor_3 = np.array([tensor[:,:,i].reshape(-1, order = 'F') for i in range(tensor.shape[2])]) #unfolded mode 3

#################
#iteration start#
#################

# Setting Initial A, B and C
I = tensor.shape[0]
J = tensor.shape[1]
K = tensor.shape[2]

R = 5

initial_mean = (np.mean(tensor)/R)**(1/3) # since mode 3
initial_sd = (np.var(tensor))**(1/2)

#A = np.random.randint(10, 20, size = (I,R))
A = initial_mean + (initial_sd/R)*np.random.randn(I,R)
B = initial_mean + (initial_sd/R)*np.random.randn(J,R)
C = initial_mean + (initial_sd/R)*np.random.randn(K,R)

#Check whether the values are similar
CB_khatri = tl.tenalg.khatri_rao([C,B])
np.matmul(A, CB_khatri.T)

#tensor_1 - np.matmul(A,CB_khatri.T)
#tensor_1_CBk = np.matmul(tensor_1, CB_khatri)
iteration_number = 1
diff = 1000
tol = 1e-8

while diff > tol:
    ##########
    #update A#
    ##########
    CB_khatri = tl.tenalg.khatri_rao([C,B])
    
    CtC = np.matmul(C.T, C)
    BtB = np.matmul(B.T, B)
    
    CtCBtB = CtC*BtB
    CtCBtB_pinv = np.linalg.pinv(CtCBtB) #inv also possible
    
    later_matrix_1 = np.matmul(CB_khatri, CtCBtB_pinv)
    A_updated = np.matmul(tensor_1, later_matrix_1)
    #A - np.matmul(tensor_1, later_matrix_1)
    #A - A_updated
    
    print(np.sum( (tensor_1 - np.matmul(A, CB_khatri.T))**2 ))
    A = copy.copy(A_updated)
    #Check whether update is going well or not
    print(np.sum( (tensor_1 - np.matmul(A, CB_khatri.T))**2 ))
    
    ##########
    #update B#
    ##########
    CA_khatri = tl.tenalg.khatri_rao([C,A])
    
    CtC = np.matmul(C.T, C)
    AtA = np.matmul(A.T, A)
    
    CtCAtA = CtC*AtA
    CtCAtA_pinv = np.linalg.pinv(CtCAtA)
    
    later_matrix_2 = np.matmul(CA_khatri, CtCAtA_pinv)
    B_updated = np.matmul(tensor_2, later_matrix_2)
    #B - B_updated
    
    print(np.sum( (tensor_2 - np.matmul(B, CA_khatri.T))**2 ))
    B = copy.copy(B_updated)
    #Check whether update is going well or not
    print(np.sum( (tensor_2 - np.matmul(B, CA_khatri.T))**2 ))
    
    ##########
    #update C#
    ##########
    BA_khatri = tl.tenalg.khatri_rao([B,A])
    
    BtB = np.matmul(B.T, B)
    AtA = np.matmul(A.T, A)
    
    BtBAtA = BtB*AtA
    BtBAtA_pinv = np.linalg.pinv(BtBAtA)
    
    later_matrix_3 = np.matmul(BA_khatri, BtBAtA_pinv)
    C_updated = np.matmul(tensor_3, later_matrix_3)
    #C - C_updated
    
    diff_yesterday = np.sum( (tensor_3 - np.matmul(C, BA_khatri.T))**2 )
    print(np.sum( (tensor_3 - np.matmul(C, BA_khatri.T))**2 ))
    C = copy.copy(C_updated)
    #Check whether update is going well or not
    diff_today = np.sum( (tensor_3 - np.matmul(C, BA_khatri.T))**2 )
    print(np.sum( (tensor_3 - np.matmul(C, BA_khatri.T))**2 ))
    
    diff = diff_yesterday - diff_today
    iteration_number = iteration_number + 1
    
    print("iteration: " + str(iteration_number) + " Done")
    print("Difference Made at updating C: " + str(diff))
    
    

#Reconstruction
CB_khatri = tl.tenalg.khatri_rao([C,B])

tensor_reconstructed_1 = np.matmul(A, CB_khatri.T)

tensor_return = tensor_reconstructed_1.reshape(tensor.shape, order = 'F')


print("True Tensor")
print(tensor)

print("Tensor reconstruction made by myself")
print(tensor_return)
print("Sum Square Diff")
print(np.sum( (tensor - tensor_return)**2 ))

#compare with tensorly
factors = parafac(tensor, rank = R)
tensorly_recon = tl.kruskal_to_tensor(factors)
print("Tensor reconstruction made by tensorly")
print(tensorly_recon)
print("Sum Square Diff")
print(np.sum( (tensor - tensorly_recon)**2 ))