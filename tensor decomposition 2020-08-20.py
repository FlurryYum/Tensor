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


factors = parafac(tensor, rank = 5)


A = factors[0]
B = factors[1]
C = factors[2]

CB_khatri = tl.tenalg.khatri_rao([C,B])
CA_khatri = tl.tenalg.khatri_rao([C,A])
BA_khatri = tl.tenalg.khatri_rao([B,A])

X1_reconstructed = np.matmul(A,CB_khatri.T)
X2_reconstructed = np.matmul(B,CA_khatri.T)
X3_reconstructed = np.matmul(C,BA_khatri.T)

#tensor_1 = tl.unfold(tensor, mode = 0, order = 'F')
#np.round(X1_reconstructed - tensor_1)
#tensor_reconstructed = tl.kruskal_to_tensor(factors)
#np.round(tensor_reconstructed)

np.round(X1_reconstructed)
np.round(X2_reconstructed)
np.round(X3_reconstructed)

tensor_1 = np.array([tensor[i,:,:].reshape(-1, order = 'F') for i in range(tensor.shape[0])]) #unfolded mode 1
tensor_2 = np.array([tensor[:,i,:].reshape(-1, order = 'F') for i in range(tensor.shape[1])]) #unfolded mode 2
tensor_3 = np.array([tensor[:,:,i].reshape(-1, order = 'F') for i in range(tensor.shape[2])]) #unfolded mode 3

#tensor_return = tensor_1.reshape(tensor.shape, order = 'F')
#tensor == tensor_return

#################
#iteration start#
#################

#tensor_1 - np.matmul(A,CB_khatri.T)
#tensor_1_CBk = np.matmul(tensor_1, CB_khatri)

#update A
CB_khatri = tl.tenalg.khatri_rao([C,B])

CtC = np.matmul(C.T, C)
BtB = np.matmul(B.T, B)

CtCBtB = CtC*BtB
CtCBtB_pinv = np.linalg.pinv(CtCBtB) #inv also possible

later_matrix_1 = np.matmul(CB_khatri, CtCBtB_pinv)
A_updated = np.matmul(tensor_1, later_matrix_1)
#A - np.matmul(tensor_1, later_matrix_1)
#A - A_updated
A = copy.copy(A_updated)

#update B
CA_khatri = tl.tenalg.khatri_rao([C,A])

CtC = np.matmul(C.T, C)
AtA = np.matmul(A.T, A)

CtCAtA = CtC*AtA
CtCAtA_pinv = np.linalg.pinv(CtCAtA)

later_matrix_2 = np.matmul(CA_khatri, CtCAtA_pinv)
B_updated = np.matmul(tensor_2, later_matrix_2)
#B - B_updated
B = copy.copy(B_updated)

#update C
BA_khatri = tl.tenalg.khatri_rao([B,A])

BtB = np.matmul(B.T, B)
AtA = np.matmul(A.T, A)

BtBAtA = BtB*AtA
BtBAtA_pinv = np.linalg.pinv(BtBAtA)

later_matrix_3 = np.matmul(BA_khatri, BtBAtA_pinv)
C_updated = np.matmul(tensor_3, later_matrix_3)
#C - C_updated
C = copy.copy(C_updated)

#Reconstruction
CB_khatri = tl.tenalg.khatri_rao([C,B])

tensor_reconstructed_1 = np.matmul(A, CB_khatri.T)

tensor_return = tensor_reconstructed_1.reshape(tensor.shape, order = 'F')
tensor_return


