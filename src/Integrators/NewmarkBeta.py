import numpy as np
import numba
from src.Common import ParticleType

spec = [
    ('beta', numba.float64),
    ('gamma', numba.float64),
    ('M', numba.float64[:, :]),
    ('K', numba.float64[:, :]),
    ('C', numba.float64[:, :])
]
@numba.jitclass(spec)
class NewmarkBeta():
    def __init__(self, beta: float, gamma: float, M: np.array, K: np.array, C: np.array):
        """
            Newmark-beta integrator; Specifically designed for couplings
            The accompanying acceleration implementation is required.
            
            Parameters
            ----------
            beta: float
                Newmark-Beta parameter. Commonly 0.25
            gamma: float
                Newmark-Beta parameter. Commonly 0.5
            M: np.array [N x N]
                N by N matrix containing the masses of the nodes.
            K: np.array
                N by N stiffness matrix
            C: np.array
                N by N damping matrix
        """
        assert(beta * 2 >= gamma)
        self.beta = beta
        self.gamma = gamma
        
        # Set the matrices
        self.M = M
        self.K = K
        self.C = C
        
    def predict(self, dt: float, pA: np.array, damping: float):
        J = len(pA)
        for j in numba.prange(J):
            self.checkCoupling(pA[j])
            
            # Position
            pA[j]['x'] = pA[j]['x'] + pA[j]['vx'] * dt + pA[j]['ax'] * (0.5 - self.beta) * dt ** 2
            pA[j]['y'] = pA[j]['y'] + pA[j]['vy'] * dt + pA[j]['ay'] * (0.5 - self.beta) * dt ** 2
            
            # Velocity
            pA[j]['vx'] = pA[j]['vx'] + pA[j]['ax'] * (1 - self.gamma) * dt
            pA[j]['vy'] = pA[j]['vy'] + pA[j]['ay'] * (1 - self.gamma) * dt
        return pA
        
    def correct(self, dt: float, pA: np.array, damping: float):
        J = len(pA)
        for j in numba.prange(J):
            self.checkCoupling(pA[j])
            
            # Position
            pA[j]['x'] = pA[j]['x'] + pA[j]['ax'] * self.beta * dt ** 2
            pA[j]['y'] = pA[j]['y'] + pA[j]['ay'] * self.beta * dt ** 2
            
            # Velocity
            pA[j]['vx'] = pA[j]['vx'] + pA[j]['ax'] * self.gamma * dt
            pA[j]['vy'] = pA[j]['vy'] + pA[j]['ay'] * self.gamma * dt
        return pA
        
    def acceleration(self, dt: float, F: np.array, r: np.array, v: np.array):
        """
            Compute the acceleration according to the Newmark-Beta method.
            
            Parameters
            ----------
            dt: float
                The current time-step.
            F: np.array
                N-matrix containing the external (nodal) forces.
            r: np.array
                Displacement, N-matrix
            v: np.array
                Velocity, N-matrix
                
            Returns
            -------
            a: np.array
                Acceleration, N-matrix
        """
        rhs = F - np.dot(self.K, r) - np.dot(self.C, v)
        lhs = self.M + self.C * self.gamma * dt + self.K * self.beta * dt ** 2
        
        return np.dot(np.linalg.inv(lhs), rhs)

    def checkCoupling(self, p: np.array):
        if p['label'] != ParticleType.Coupled:
            print('\nNewmark-Beta integration method is only suited for solid coupling particles.')