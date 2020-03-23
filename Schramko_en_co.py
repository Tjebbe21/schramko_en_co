import numpy as np
from scipy.integrate import quad
from scipy import linalg
from scipy import optimize
from scipy.sparse import diags
from math import factorial
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self):
        self.Nt = 10              # number of Fourier components for phi
        self.N  = 10              # number of terms in Cosine series cos(n pi y)

        self.beta   = 1           # ratio of width B to longitudinal lengthscale, wich equals ratio of V velocity scale to U velocity scale
        self.sigma  = 0.7         # relative importance of interia to advection
        self.r      = 1           # relative importance of friction to advection
        self.Ro_inv = 0.5         # inverse Rossby number

        self.Lambdatilde = 0.01   # Effective bedslope parameter
        self._b1 = 2              # advective sediment transport power constant (2 is fast, others are slow)
        self.b2  = 2              # bed slope power constant (2 and 3 are fast, others are slow)

        self.ueq_dict = {}        # save Fourier modes of u_eq to speed up a little bit (clears when you change b1)


    @property
    def gamma(self):
        ''' 
        relative importance of torques by Coriolis to torques by friction 
        '''
        return self.Ro_inv /self.r
    
    @property
    def Lambda(self):
        '''
        Lambda = Lambdatilde <|u_0|^{b_2}>
        '''
        if self.b2 == 2:
            return self.Lambdatilde /2
        elif self.b2 == 3:
            return 4* self.Lambdatilde /(3*np.pi)
        else:
            integrand = lambda t : abs(np.cos(t))**self.b2
            integral = quad(integrand,0,2*np.pi)[0]
            return integral / (2*np.pi)

    @property
    def b1(self):
        '''
        advective sediment transport power constant
        '''
        return self._b1
    @b1.setter
    def b1(self,value):
        self.ueq_dict = {}  # clear ueq_dict when b1 is changed
        self._b1 = value


    def phi(self,p,k):
        '''
        Solve (σ/r ∂_t + u0 ik/r + 1)ϕ = u0 by tructated Fourier series.
        p is Fourier mode
        k is longitudinal wavenumber
        '''
        if abs(p) > self.Nt:
            return 0
        else:
            size = 2 * self.Nt + 1  #-N,...,-1,0,1,...,N
            u0s = linalg.toeplitz([self.u0(p) for p in range(size)])
            diagonals = diags([1j * p * self.sigma/self.r + 1 for p in range(-self.Nt,self.Nt+1)])
            A = diagonals + 1j * k / self.r * u0s
            b = [self.u0(p) for p in range(-self.Nt,self.Nt+1)]
            
            phi = np.linalg.solve(A, b)
            return phi[p + self.Nt]

    def u0(self,p):
        '''
        p-th Fourier component of u0
        '''
        if p in [-1,1]:
            return 0.5
        else:
            return 0

    def u0_pow_b1(self,p):
        '''
        p-th Fourier component of |u0|^b1
        '''
        if self.b1 == 2:
        # if False:
            if p == 0:
                return 0.5
            elif p in [-2,2]:
                return 0.25
            else:
                return 0
        else:
            if p not in self.ueq_dict:
                f_re = lambda t: np.real(abs(np.cos(t))**self.b1 * np.exp(-1j * p * t))
                f_im = lambda t: np.imag(abs(np.cos(t))**self.b1 * np.exp(-1j * p * t))
                self.ueq_dict[p] = 1/(2*np.pi) * (quad(f_re,0,2*np.pi)[0] + 1j * quad(f_im,0,2*np.pi)[0])
            return self.ueq_dict[p]
    

    def avg_u0_pow_b1_phi(self,k):
        '''
         < |u_0|^b1 phi >
        k is wavenumber
        '''
        if self.b1 == 2:
        # if False:
            term1 = 0.5 * self.phi(0,k)
            term2 = 0.25 * (self.phi(2,k) + self.phi(-2,k))
            return term1 + term2
        else:
            f_array = np.array([self.u0_pow_b1(p) for p in range(-self.Nt, self.Nt+1)])
            g_array = np.array([self.phi(-p,k) for p in range(-self.Nt, self.Nt+1)])
            return f_array @ g_array

    def alpha(self,k):
        return self.b1 * 1j * k * self.avg_u0_pow_b1_phi(k)


    def omega_noCor(self,m,k):
        '''
        growth rates when Coriolis is neglected
        omega_m(k) is growth rate corresponding to pattern cos(m pi y)exp(ikx) + c.c.
        '''
        adv = self.alpha(k) * (m*np.pi)**2 / ((self.beta * k)**2 + (m*np.pi)**2)
        bedload = -self.Lambda * ( (m*np.pi/self.beta)**2 + k**2 )
        return adv + bedload
    
    def noCor_matrix(self,k):
        '''
        Matrix D = diag(omega_0(k), dots omega_N(k))
        '''
        return np.diag([self.omega_noCor(m,k) for m in range(self.N)])

    def Amn(self,k,m,n):
        '''
        Elements of matrix A
        k is wavenumber
        '''
        if (m + n) % 2 == 0:
            return 0
        else:
            return self.alpha(k) * 1j * k * ( 4 * m**2  ) / ( (m**2 - n**2) * ((self.beta * k)**2 + (m * np.pi)**2) )

    def Cor_matrix(self,k):
        '''
        Matrix A multiplied by gamma
        '''
        A = np.zeros((self.N,self.N),dtype=complex)
        for m in range(self.N):
            A[m,:] = [self.Amn(k,m,n) for n in range(self.N)]
        return A

    def Matrix(self,k):
        '''
        Matrix D+gamma A for the eigenvalue problem
        '''
        return self.noCor_matrix(k) + self.gamma * self.Cor_matrix(k)
    
    def eig(self,k):
        '''
        solve eigenvalue problem omega h = (D+gamma A)h
        returns eigenvalues sorted by decreasing real part and corresponding eigenvectors
        '''
        A = self.Matrix(k)
        # Solve Ah = omega h
        eigenvalues, eigenvectors = linalg.eig(A)
        
        #sort on size realpart of eigenvalues
        idx = eigenvalues.argsort()[::-1] 
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        return eigenvalues,eigenvectors
    
    def pref_wavenumber(self, tol = None, output_growthrate=False):
        """
        Returns prefered wavenumber and growthrate (complex) using the optimazation brent from scipy.
        """

        omega = lambda k: -1 * self.eig(k)[0][0].real

        optimized_result = optimize.minimize_scalar(omega, bracket = (0.001, 10), tol=tol, method='Brent')
        # optimized_result = optimize.minimize_scalar(omega, tol=tol, method='Brent')
        k_pref = optimized_result.x
        omegas_pref, h_prefs = self.eig(k_pref)
        omega_pref = omegas_pref[0]

        if not optimized_result.success:
            exit('optimize_scalar did not succeed') 

        if k_pref < 0 : #omega is symmetric around k=0, so if k is pref than also -k; we are only looking for positive values of k.
            k_pref = -k_pref
        if omega_pref <=0:
            print('--> no instabilities!')
            returnvalue = np.nan, np.nan
        else:
            print('--> k_pref = %s' %k_pref)            
            returnvalue = k_pref, omega_pref

        if output_growthrate == True:
            return returnvalue
        else:
            return returnvalue[0]
    
    def h_u_v(self,k,eigenvector_nr=0,x_range=np.linspace(0,np.pi,100),y_range=np.linspace(0,1)):
        '''
        returns the bottom height h and the <u> and <v>
        on the grid x in [0,pi] en y in [0,1]
        k is wavenumber
        '''
        omega,h = self.eig(k)
        
        h = h[:,eigenvector_nr]

        exp = np.exp(1j * k * x_range)
        exp = exp[np.newaxis,:] # row vector with e^(ikx)
        cos = np.array([np.cos(np.arange(self.N) * np.pi * y0) for y0 in y_range])
        eigvec_y = cos @ h
        eigvec = np.real(eigvec_y[:,np.newaxis] @ exp)

        #calcualte v_res
        n_range = np.arange(self.N)
        denom = (self.beta*k)**2 + (n_range * np.pi)**2
        factor = (np.cosh(self.beta*k) - (-1)**n_range )/np.sinh(self.beta * k)
        term1_v = np.array([n_range * np.pi * np.sin(n_range * np.pi * y0) for y0 in y_range])
        term2_v = 1j * k * self.gamma  * np.array([np.cos(n_range * np.pi * y0) - np.cosh(self.beta * k * y0) + np.sinh(self.beta * k * y0) * factor for y0 in y_range])
        som_v = (term1_v + term2_v) @ (h/denom)
        res_v_y = 1j * k * self.phi(0,k) * som_v
        res_v = np.real(res_v_y[:,np.newaxis] @ exp)

        #calculate u_res via dv/dy
        term1_u = np.array([(n_range * np.pi)**2 * np.cos(n_range * np.pi * y0) for y0 in y_range])
        term2_u = -1j * k * self.gamma  * np.array([n_range * np.pi * np.sin(n_range * np.pi * y0) + self.beta * k * np.sinh(self.beta * k * y0) - self.beta * k * np.cosh(self.beta * k * y0) * factor for y0 in y_range])
        som_u = (term1_u + term2_u) @ (h/denom)
        res_u_y = -1 * self.phi(0,k) * som_u
        res_u = np.real(res_u_y[:,np.newaxis] @ exp)

        return eigvec, res_u, res_v

    def perturbed_h(self,x_range=np.linspace(0,np.pi,100),y_range=np.linspace(0,1)):
        '''
        O(gamma) correction on fastest growing bottom pattern without Coriolis.
        '''
        X,Y = np.meshgrid(x_range,y_range)

        k = self.pref_wavenumber()
        
        omegas = [self.omega_noCor(m,k) for m in range(self.N)]
        n_p = np.argsort([self.omega_noCor(m,k) for m in range(self.N)])[::-1][0]
        h1_n = np.array([self.Amn(k,m,n_p)/(omegas[n_p] - omegas[m]) if m != n_p else 0 for m in range(self.N)])
        h1_n /= linalg.norm(h1_n)



        exp = np.exp(1j * k * x_range)
        exp = exp[np.newaxis,:] # row vector with e^(ikx)

        cos = np.zeros((len(y_range),self.N))

        for j in range(len(y_range)):
            cos[j] = np.cos(np.arange(self.N) * np.pi * y_range[j])

        eigvec_y = cos @ h1_n
        eigvec = np.real(eigvec_y[:,np.newaxis] @ exp)
        return X,Y, eigvec, k





        
        

