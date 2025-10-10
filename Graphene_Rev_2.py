from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from ase.units import kB

class Hamiltonian:
    def __init__(self, a, N_unitcell, N_e_unitcell,t, a1, a2, b1, b2, KX, KY, C1, C2, T, V0):
        self.a = a
        self.N_unitcell = N_unitcell
        self.N_e_unitcell = N_e_unitcell
        self.t = t
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.kx = KX
        self.ky = KY
        self.c1 = C1
        self.c2 = C2
        self.V0 = V0
        self.T = T
        
        self.mu = 0
                
       
        
    
    # band structure methods
    
    def band(self):
        
        # real-space vectors R connecting neighboring unit cells
        R = np.array([
            [0, 0],         # R = [0, 0] 
            -self.a1,       # R = [-1, 0]
            -self.a2,       # R = [0, -1]
            self.a1,        # R = [1, 0]
            self.a2         # R = [0, 1]
        ])
        
        
        # array of 5 entries in which each entry is a 2x2 matrix
        h_R = np.array([
            #it's honeycomb so I have two atoms per unit cell

            [[0, self.t], [self.t, 0]],  # R = [0, 0]
            [[0, self.t], [0, 0]],       # R = [-1, 0]
            [[0, self.t], [0, 0]],       # R = [0, -1]
            [[0, 0], [self.t, 0]],       # R = [1, 0]
            [[0, 0], [self.t, 0]]        # R = [0, 1]
        ])

        # now for each k in the grid (self.kx self.ky) compute the hamiltonian matrix 
        # as sum over the entries of R of (exp(ikR)*h_R)
        # where R is the vector connecting the 1st unit cell to the nearest neighbor unit cells
        H_k = np.zeros((self.kx.shape[0], self.kx.shape[1], 2, 2), dtype=complex)
        
        
        for i in range(h_R.shape[0]):
            phase = np.exp(1j * (self.kx * R[i,0] + self.ky *R[i,1]))
            #R[i,0] is the x component of the i-th R vector
            #R[i,1] is the y component of the i-th R vector
            
            H_k += h_R[i] * phase[:,:, np.newaxis, np.newaxis]
         
            
        # now I want to compute the eigenvalues and eigenvectors of each 2x2 matrix in H_k
        val, u = np.linalg.eigh(H_k)
    
        return val, u
    
    def bandPlot(self, E):
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,  projection="3d")

        

        # alpha controls transparency, linewidth=0 removes grid lines on surface
        for i in range(E.shape[2]):
            surf = ax.plot_surface(self.c1, self.c2, E[:,:,i], alpha=0.8, linewidth=0)
        


        ax.set_xlabel(r"$C_1$")
        ax.set_ylabel(r"$C_2$")
        ax.set_zlabel("Energy")
        ax.set_title("Graphene Band Structure in 1st BZ")
        plt.show()
             
             
             
    # Scattering rate method
 
    def scatterRate(self, kx, ky, E, u):
        Nx = kx.shape[0]
        Ny = kx.shape[1]
        n_bands = E.shape[2]
        
        # flatten energies and eigenvectors
        E_flat = E.reshape(self.N_unitcell, n_bands)
        u_flat = u.reshape(self.N_unitcell, n_bands, n_bands)

        # build scattering Hamiltonian H_k
        R = np.array([
            [0, 0],         # R = [0, 0] 
        ])
        
        h_R = np.array([
            [[self.V0, 0], [0, 0]] # vacancy potential term
        ])  
        
        
        H_k = np.zeros((Nx, Ny, n_bands, n_bands), dtype=complex)
        for i in range(h_R.shape[0]):
            # phase is (Nx,Ny) array
            phase = np.exp(1j * (kx * R[i, 0] + ky * R[i, 1]))       
            H_k += h_R[i] * phase[:,:, np.newaxis, np.newaxis]
        
        H_flat = H_k.reshape(self.N_unitcell, n_bands, n_bands)

        
    

        # We need U_dagger(k_j), which requires transposing the orbital and band axes.
        # transpose (0,2,1) swaps the last two axes. 
        # For a stack of matrices, this is equivalent to taking the transpose of each individual matrix.
        u_flat_dagger = np.conjugate(u_flat).transpose(0, 2, 1)
        
        
        # M_full[j,i,b,d] = <u(k_j,b) | H(k_j) | u(k_i,d)>
        # indices: j=final k, i=initial k, b=final band, d=initial band
        # a,c are the internal orbital indices being summed over
        # so M_full will be (N,N,n_band,n_band)
        M_full = np.einsum('jba, jac, icd -> ijdb', u_flat_dagger, H_flat, u_flat, optimize=True)

        # |M|^2 (amplitude squared)
        M2 = np.abs(M_full)**2  

        #  modified Dirac delta 
        
        # broadcasting: np.newaxis adds dimensions of size 1 to the array
        # Ei is reshaped to "row vector" of initial states
        # Ej is reshaped to "column vector" of final states.
               
        # (N, 1, n_bands, 1)
        Ei = E_flat[:, np.newaxis, :, np.newaxis]  
        fi = self.fd_dist(Ei, self.mu)
        # (1, N, 1, n_bands)
        Ej = E_flat[np.newaxis, :, np.newaxis, :] 
        fj = self.fd_dist(Ej, self.mu)
        
        # (N,N,n_band,n_band)
        delta = -self.dirac_delta(Ei, fi, Ej, fj) 
        
        
      
        # sum over all final k and bands 
        # (N,n_bands)
        # sum over j (final k) and b (final band)
        rates = np.sum(M2 * delta, axis=(1, 3))  
        rates_grid = rates.reshape(Nx, Ny, n_bands)

        return 2 * np.pi * rates_grid / self.N_unitcell

        
    # Fermi Dirac distribution methods

    def fd_dist (self, E,mu):
        # Boltzmann constant in eV/K
        arg = (E - mu) / (kB * self.T)
        return expit(-arg)
    
    def fd_dist_der (self, E,mu):
        arg = (E - mu) / (kB * self.T)
        fx = expit(-arg)
        fx_m = expit(arg)
        return -fx * fx_m/ (kB * self.T)

    def dirac_delta(self, E_i, f_i, E_j, f_j):
        #all things with i index should be (N, 1, n_bands, 1) arrays
        #all things with j index should be (1, N, 1, n_bands) arrays
        
        
      
        # broadcasted (N,N,n_bands,n_bands) array
        delta_E = E_i - E_j
        
        # finite-difference ratio (N,N,n_bands,n_bands) array, ignoring warnings about division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (f_i - f_j) / delta_E
        
        # Find the locations where the energy difference is effectively zero
        zero_mask = np.isclose(delta_E, 0)

        # For these locations, calculate the analytical derivative    
        # der is (N, 1, n_bands, 1) but will be properly stretched by NumPy's broadcasting through np.where()    
        der = self.fd_dist_der(E_i,self.mu)
       
        # Use the analytical derivative where the energy difference is zero,
        # and the finite-difference ratio everywhere else.
        # result is (N, N, n_bands, n_bands)
        
        
        return np.where(zero_mask, der, ratio)
        
        
    # calculate mu methods
    
    def cal_e_density(self, mu, E, N):  
        
        return np.sum(self.fd_dist(E,mu))/ N

    def calc_mu(self, E, tol=1e-6):
        
        E_flat = E.flatten()
        
        mu_low = np.min(E_flat) 
        mu_high = np.max(E_flat) 
        
        # function whose root we want to find
        def root_function(mu):
            n_calc = self.cal_e_density(mu, E_flat, self.N_unitcell)
            return n_calc - self.N_e_unitcell

    
        return brentq(root_function, mu_low, mu_high, xtol=tol)
       
       
    # plot with scattering rates method
        
    def scatterPlot(self, E, u):
        # generic function plot band energy points colored by scattering rate
        
        # find scattering rates at each kx,ky
        rates = self.scatterRate(self.kx, self.ky, E, u)
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,  projection="3d")

        
        # set up shared normalization across both bands
        norm = Normalize(vmin=rates.min(), vmax=rates.max())
        cmap = plt.cm.viridis
        # compute facecolors for both bands using the same normalization
        for i in range(E.shape[2]):
             # facecolors uses a colormap to color the surface based on scattering rates
            facecolors = cmap(norm(rates[:,:,i]))
             # alpha controls transparency, linewidth=0 removes grid lines on surface
            surf = ax.plot_surface(self.c1, self.c2, E[:,:,i], alpha=0.8, linewidth=0, facecolors=facecolors)
        
             

        ax.set_xlabel(r'$c1$')
        ax.set_ylabel(r'$c2$')
        ax.set_title('Band Energies Colored by Scattering Rate')
        
        # add one colorbar for all bands
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([]) 
        fig.colorbar(sm, ax=ax, label="Scattering rate")
        plt.show()



# lattice constant, C-C distance, in Angstroms
a = 1.42
# hopping term
t = 2

#vacancy potential
V0 = 100


#electrons per unit cell (max 2)
N_e_unitcell = 1.2

# primitive lattice vectors for graphene
a1 = a*np.array([3/2, np.sqrt(3)/2])
a2 = a*np.array([3/2, -np.sqrt(3)/2])

# reciprocal lattice vectors
A = np.column_stack((a1, a2))
B = 2*np.pi * np.linalg.inv(A).T
b1 = B[:,0]
b2 = B[:,1]

# k-space grid 
N =100 # number of k-points per dimension, total points = N^2

#Temperature 
T = 300

# Create a grid of coefficients c1, c2 from -0.5 to 0.5
c_range = np.linspace(-0.5, 0.5, N)
C1, C2 = np.meshgrid(c_range, c_range)

# calculate the Cartesian kx, ky values 
KX = C1*b1[0] + C2*b2[0]
KY = C1*b1[1] + C2*b2[1]

H = Hamiltonian(a,N**2,N_e_unitcell,t,a1,a2,b1,b2,KX,KY,C1,C2,T,V0)


#band structure
E, u = H.band()
H.bandPlot(E)

# calculate the correct chemical potential mu
mu = H.calc_mu(E)
print(f"Chemical Potential mu for electron per unit cell n_e_unitcell={H.N_e_unitcell}: {mu:.4f} eV")
# update the mu attribute in the Hamiltonian object
H.mu = mu

# band surface with scattering rates
H.scatterPlot(E, u)
