import numpy as np
def f(u):
    x, y = u
    return (x-1)**4 + (y+2)**4 + (x-1)**2 + 2*(y+2)**2 + (x-1)*(y+2)

def grad(u):
    x, y = u
    return np.array([
        4*(x-1)**3 + 2*(x-1) + (y+2),
        4*(y+2)**3 + 4*(y+2) + (x-1)
    ])

def hessian(u):
    x, y = u
    return np.array([
        [12*(x-1)**2 + 2,
        1
        ],
        [1,
        12*(y+2)**2 + 4
        ]
    ])

def newton_optim(u_0, grad_fn, hessian_fn, ep = 10e-8, max_it = 100 ):
    u = u_0
    for _ in range(max_it):
        grad = grad_fn(u)
        if np.linalg.norm(grad) < ep:
            return u
        hessian = hessian_fn(u)
        t = np.linalg.solve(hessian, -grad)
        u = u + t
    return u

if __name__ == "__main__":
    u_0 = (0, 0)
    u_optim = newton_optim(u_0, grad, hessian)
    print(u_optim)
