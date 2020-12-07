import numpy as np
import random
import csv

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        row_num = 0
        for row in reader:
            row_num += 1
            # skip the first row
            if row_num == 1:
                continue
            # skip the first col
            del row[0]
            # convert string to float
            for i in range(len(row)):
                row[i] = float(row[i])
            dataset.append(row)
    return np.array(dataset)

def print_stats(dataset, col):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.
    RETURNS:
        None
    """
    # print number of data points
    num = dataset.shape[0]
    print(num)
    
    # print sample mean
    mean = (np.sum(dataset.transpose()[col]))/num
    print('{:.2f}'.format(mean))
    
    # print sample standard deviation
    total = 0
    for i in range (num):
        total += (dataset[i][col] - mean) ** 2
    sd = (total/(num - 1)) ** 0.5
    print('{:.2f}'.format(sd))
    pass


def regression(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        mse of the regression model
    """
    mse = 0
    for i in range (len(dataset)):
        mse += (betas[0] + np.dot(dataset[i][cols],betas[1:]) - dataset[i][0])**2
    return mse/len(dataset)


def gradient_descent(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        An 1D array of gradients
    """
    grads = []
    for i in range(len(betas)):
        entry = 0
        for j in range(len(dataset)):
            temp = betas[0] + np.dot(dataset[j][cols],betas[1:]) - dataset[j][0]
            # if beta0, times nothing; 
            # if other beta, times corresponding x from dataset 
            if i == 0:
                entry += temp
            else:
                entry += temp * dataset[j][cols[i-1]]
        grads.append(entry*2/len(dataset))       
    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate
    RETURNS:
        None
    """
    # need to keep track beta value after each interation
    trace = []
    # add initial value to beta trace
    trace.append(betas)
    for i in range (T):
        # compute current iteration number 
        num = i + 1
        # compute new beta values 
        grads = gradient_descent(dataset, cols, trace[i])
        new_betas = []
        beta_out = []
        for j in range(len(betas)):
            val = trace[i][j] - eta * grads[j]
            new_betas.append(val)
            # round to two digits after decimal for output
            beta_out.append('{:.2f}'.format(val))
        trace.append(new_betas)
        # compute current MSE
        mse = regression(dataset, cols, trace[i + 1])
        # print out curent info
        print(num, '{:.2f}'.format(mse),*beta_out)
    pass


def compute_betas(dataset, cols):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    # retrieve the subset of features and labels
    y = []
    X = []
    for i in range (len(dataset)):
        y.append(dataset[i][0])
        temp = []
        temp.append(1)
        for col in cols:
            temp.append(dataset[i][col])
        X.append(temp)
    X = np.array(X)
    # compute betas in closed form
    betas = (np.linalg.inv(X.transpose()@X))@X.transpose()@y
    # compute mse
    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values
    RETURNS:
        The predicted body fat percentage value
    """
    betas = (compute_betas(dataset, cols))[1:]
    x = []
    x.append(1)
    for feature in features:
        x.append(feature)
    x = np.array(x)
    result = np.dot(x,betas)
    return result


def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.
    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,
    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)

        
def sgd(dataset, cols, betas, T, eta):
    """
    You must use random_index_generator() to select individual data points.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate
    RETURNS:
        None
    """
    # use the same iteration procedure as 'iterate_gradient'
    trace = []
    trace.append(betas)
    # set up random generator iterator
    seq_gen = random_index_generator(0, len(dataset)-1, seed=42)
    # start iteration here 
    for i in range (T):
        # randomly pick one of the n items
        index = next(seq_gen)
        # compute gradient for each beta
        grads = []
        entry = 2*(trace[i][0] + np.dot(dataset[index][cols],trace[i][1:]) - dataset[index][0])
        grads.append(entry)
        for col in cols:            
            grads.append(entry * dataset[index][col])
        grads = np.array(grads)
        # compute new betas
        new_betas = []
        out = []
        for j in range(len(betas)):
            val = trace[i][j] - eta * grads[j]
            new_betas.append(val)
            out.append('{:.2f}'.format(val))
        trace.append(new_betas)
        mse = regression(dataset, cols, trace[i+1])
        print(i+1, '{:.2f}'.format(mse), *out)
    pass


if __name__ == '__main__':
    # Your debugging code goes here.
    pass