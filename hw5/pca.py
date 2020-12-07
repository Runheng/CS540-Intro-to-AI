from scipy.linalg import eigh  
import numpy as np
import matplotlib.pyplot as plt

'''load the dataset from a provided .npy file, re-center it around the origin and return it as a NumPy array of floats'''
def load_and_center_dataset(filename):
    x = np.load(filename)
    x = np.reshape(x,(2000,784))
    x = x - np.mean(x, axis=0)
    return x

'''calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)'''
def get_covariance(dataset):    
    return (1/(len(dataset)-1))*np.dot(np.transpose(dataset),dataset)

'''perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array)
with the largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the 
corresponding eigenvectors as columns '''
def get_eig(S, m):
    '''
    # compute eigenvalues and eigenvectors
    w,v = eigh(S)
    # only keep the largest m eigen values
    Lambda = w[len(S)-m :]
    # sort in decreasing order
    Lambda = Lambda[::-1]
    # keep correct eigenvetors after rearrangement
    U = v[:,len(S)-m:]
    U = np.transpose(np.transpose(U)[::-1]) 
    '''
    Lambda, U = eigh(S, eigvals=(len(S)-m, len(S)-1))
    Lambda = Lambda[::-1]
    U = np.transpose(np.transpose(U)[::-1]) 
    return np.diag(Lambda), U

'''similar to get_eig, but instead of returning the first m, return all eigenvectors that explains more than perc % of variance'''
def get_eig_perc(S, perc):
    L,U = eigh(S)
    # compute sum of all lambda values
    sum_lambda = np.sum(L)
    for i in range(0, len(S)):
        # delete the one no more than perc% of variance
        index = len(S) - i - 1
        if (L[index]/sum_lambda) <= perc:
            L = np.delete(L, index)
            U = np.delete(U, index, axis = 1)
    # sort in decreasing order
    L = L[::-1]
    U = np.transpose(np.transpose(U)[::-1])
    return np.diag(L), U

'''project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array'''
def project_image(image, U):
    # U@U.T@image
    return np.dot(np.dot(U,np.transpose(U)),image)

'''use matplotlib to display a visual representation of the original image and the projected image side-by-side'''
def display_image(orig, proj):
    # reshape the image to 28x28
    orig = np.reshape(orig,(28,28))
    proj = np.reshape(proj,(28,28))
    # create a figure with two subplots in one row
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
    # title each subplot
    ax0.set_title("Original")
    ax1.set_title("Projection")
    # render the orig in the 1st subplot, proj in the 2nd subplot
    c0 = ax0.imshow(orig, aspect='equal', cmap='gray')
    c1 = ax1.imshow(proj, aspect='equal', cmap='gray')
    # create a colorbar
    fig.colorbar(c0, ax=ax0)
    fig.colorbar(c1, ax=ax1)
    # render the plots
    plt.show()
    #plt.savefig("test.jpg")
    return 
