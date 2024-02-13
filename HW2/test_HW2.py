from HW2 import naive_LU, solve_LU
import numpy as np
def test_naive_LU():
    A =  np.random.randint(0,10,(3,3))
    L,U = naive_LU(A)
    assert(np.allclose(A,np.matmul(L,U)))
def test_solve_LU():
    L = np.array([[1, 0, 0], [2/3, 1, 0], [-1/3, -4/11, 1]])
    U = np.array([[3, -2, 1], [0, -22/3, -14/3], [0, 0, 40/11]])
    b = np.array([-10,44,-26])
    assert(np.allclose(solve_LU(L,U,b), np.linalg.solve(np.matmul(L,U),b)))
    b = np.array([-2,44,-26])
    assert(np.allclose(solve_LU(L,U,b), np.linalg.solve(np.matmul(L,U),b)))

    b = np.array([-23,4,-26])
    assert(np.allclose(solve_LU(L,U,b), np.linalg.solve(np.matmul(L,U),b)))
