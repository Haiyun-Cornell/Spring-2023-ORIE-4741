import cvxopt
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy.spatial.distance import cdist

def solve_SVM_primal(x, y, regularization=None):
    """ solve the primal problem of SVM
    :param x: dataset x
    :param y: dataset y
    :pram regularization: the C
    :return w: the weights
    :return b: the bias
    """
    N, m = x.shape

    # Q2(a)
    # step 1: convert the input so we can get offset
    # TODO: pad the input
    x = np.pad(x, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    m += 1

    # step 2: formalize the problem
    # TODO: prepare input for the cvxopt solver
    # tutorial: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    if regularization is None:
        # Q2(a)
        p_np = np.eye(m)
        p_np[m-1, m-1] = 0
        P = cvxopt_matrix(p_np)
        q = cvxopt_matrix(np.zeros(m))
        G = cvxopt_matrix(-y[:, None] * x)
        h = cvxopt_matrix(-np.ones(N))
    else:
        # Q2(c)
        p_diag = np.zeros(m+N)
        p_diag[:m-1] = 1.0
        P_np = np.diag(p_diag)
        P = cvxopt_matrix(P_np)
        q_np = np.zeros(m+N)
        q_np[m:] = regularization
        q = cvxopt_matrix(q_np)
        G_np = np.zeros((N*2, m+N))
        G_np[:N, :m] = -y[:, None] * x
        G_np[:N, m:] = -np.eye(N)
        G_np[N:, m:] = -np.eye(N)
        G = cvxopt_matrix(G_np)
        h_np = np.zeros(N*2)
        h_np[:N] = -1
        h = cvxopt_matrix(h_np)

    # step 3: solve the problem using cvxopt
    # TODO: call the cvxopt solver
    sol = cvxopt_solvers.qp(P,q,G,h)

    # step 4: convert the result and return
    # TODO: get the w and b from the solver's solution
    theta = sol['x']
    w = np.array(theta[:m-1]).squeeze()
    b = theta[m-1]
    print(f'weights: {w}; bias: {b}')
    return w, b


def solve_SVM_dual(affinities, y, regularization, folds=5):
    """ solver the dual problem of SVM
    :param affinities: the affinity matrix, 
        where A[i,j] is K(x_i, x_j), 
        K is the kernel function
    :param y: dataset y
    :param regularization: the C
    :param folds: number of folds for cross validation
    :return w: the weights
    :return b: the bias
    """
    N = affinities.shape[0]
    N_test = N // folds
    N_train = N - N_test

    accuracy_records = []

    idxs = np.arange(N)
    np.random.shuffle(idxs)
    for iter in range(folds):
        # TODO: Q2(d)(ii) and Q2(d)(iii)
        # step 0: split the data into train and test set
        test_idxs = idxs[iter*N_test:(iter+1)*N_test]
        train_idxs = np.concatenate((
            idxs[:iter*N_test],
            idxs[(iter+1)*N_test:]))
        truncated_data = affinities[:, train_idxs]
        train_x = truncated_data[train_idxs]
        test_x = truncated_data[test_idxs]
        train_y = y[train_idxs]
        test_y = y[test_idxs]

        # step 1: formalize the problem
        # TODO: prepare input for the cvxopt solver
        # tutorial: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
        P_np = train_x * np.outer(train_y, train_y)
        P = cvxopt_matrix(P_np)
        q_np = -np.ones(N_train)
        q = cvxopt_matrix(q_np)
        G_np = np.zeros((N_train*2, N_train))
        G_np[:N_train] = -np.eye(N_train)
        G_np[N_train:] = np.eye(N_train)
        G = cvxopt_matrix(G_np)
        h_np = np.zeros(N_train*2)
        h_np[N_train:] = regularization
        h = cvxopt_matrix(h_np)
        A_np = (train_y * 1.0)[None, :]
        A = cvxopt_matrix(A_np)
        b_np = np.zeros(1)
        b = cvxopt_matrix(b_np)

        # step 2: solve the problem using cvxopt
        # TODO: call the cvxopt solver
        sol = cvxopt_solvers.qp(P,q,G,h,A,b)
        a_coefficients = np.array(sol['x']).squeeze()

        # step 3: get the offset
        # TODO:
        coeffs = a_coefficients * train_y
        offset = 1 / train_y[0] - np.dot(coeffs, train_x[0])

        # step 4: fit training data
        # TODO: compute train accuracy
        train_predictions = np.sign((coeffs * train_x).sum(axis=1) + offset)
        train_agreements = (train_predictions * train_y) >= 0
        train_accuracy = np.sum(train_agreements) / len(train_agreements)
        print(f'iter {iter}: train accuracy={train_accuracy}')

        # step 5: fit test data and record the accuracy
        # TODO: compute the test accuracy
        predictions = np.sign((coeffs * test_x).sum(axis=1) + offset)
        agreements = (predictions * test_y) >= 0
        accuracy = np.sum(agreements) / len(agreements)
        print(f'iter {iter}: test accuracy={accuracy}')
        accuracy_records.append(accuracy)

    print(f'average test accuracy: {np.mean(accuracy_records)}')


def get_support_vectors(x, y, w, b, eps=1e-3):
    """ compute the support vectors
    :param x: the datapoints' x
    :param y: the datapoints' y
    :param w, b: y=sign(wx+b) is the decision boundary
    :param eps: a==b if |a-b| < eps
    :return positive_vectors: support vectors for positive examples
    :return positive_boundary: (w, b) for positive examples
    :return negative_vectors: support vectors for negative examples
    :return negative_boundary: (w, b) for the negative examples
    """
    # Q2(b)
    # TODO:
    quality = y * ((w * x).sum(axis=1) + b) - 1
    neg_idxs = np.where((quality < eps) & (y < 0))[0]
    pos_idxs = np.where((quality < eps) & (y > 0))[0]
    negative_vectors = x[neg_idxs]
    positive_vectors = x[pos_idxs]
    pos_b = b-1
    neg_b = b+1
    positive_boundary = (w, pos_b)
    negative_boundary = (w, neg_b)

    return positive_vectors, positive_boundary, \
        negative_vectors, negative_boundary


def get_affinity_matrix(X, method, **kwargs):
    """ apply kernel function to the data
    :param X: the input data (Nxd)
    :param method: 'product' or 'rbf'
    :param M: affinity matrix (NxN)
    """
    # TODO: Q2(d)(i)
    if method == 'product':
        products = cdist(X, X, lambda u, v: np.dot(u, v))
        return products
    elif method == 'rbf':
        # TODO: implement the kernel function
        distances = cdist(X, X, lambda u, v: ((u-v)**2).sum())
        gamma = kwargs.get('gamma')
        affinities = np.exp(-gamma * distances)
        return affinities
    else:
        raise NotImplementedError(f'Unknown kernel type {method}')

