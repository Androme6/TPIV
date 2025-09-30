from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.path import Path


class Hamiltonian:
    def __init__(self, a, N_unitcell, N_e_unitcell,t, a1, a2, b1, b2, KX, KY, C1, C2, T, V0, delta_t):
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
        self.delta_t = delta_t
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
            phase = np.exp(1j * (self.kx * R[i,0] + self.ky * R[i,1]))
            #R[i,0] is the x component of the i-th R vector
            #R[i,1] is the y component of the i-th R vector
            
            H_k += h_R[i] * phase[..., np.newaxis, np.newaxis]
         
            
        # now I want to compute the eigenvalues of each 2x2 matrix in H_k
        val = np.linalg.eigvalsh(H_k)
    
    
    
        #as output of this function I would like to return the two bans as two 2D arrays over the kx ky grid
        Em = val[:,:,0]
        Ep = val[:,:,1]
        

        return Ep, Em
    
    def bandPlot(self, Ep, Em):
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,  projection="3d")

        

        # alpha controls transparency, linewidth=0 removes grid lines on surface
        surf1 = ax.plot_surface(self.c1, self.c2, Ep, alpha=0.8, linewidth=0)
        surf2 = ax.plot_surface(self.c1, self.c2, Em, alpha=0.8, linewidth=0)


        ax.set_xlabel(r"$C_1$")
        ax.set_ylabel(r"$C_2$")
        ax.set_zlabel("Energy")
        ax.set_title("Graphene Band Structure in 1st BZ")
        plt.show()
             
    # Scattering rate methods

    #formFactor and M in scatterRate are specific to vacancy scattering in graphene
    # scatter is fully generic

    def formFactor(self, kx, ky, a):
        #form factor for graphene to help compute scattering rates
        return np.exp(1j*a*kx) + 2*np.exp(-1j*(a/2)*kx)*np.cos((np.sqrt(3)/2)*a*ky)

    def scatterRate(self, kx, ky, Ep, Em):
        #number of k points 
        N = kx.size 
        # constants for matrix elements
        cos1 = self.V0 / (2 * self.N_unitcell)
        cos2 = self.delta_t / (2 * self.N_unitcell)

        kx_flat = kx.flatten()
        ky_flat = ky.flatten()
        f = self.formFactor(kx_flat, ky_flat, self.a)
        abs_f = np.abs(f)
        #so abs_f is 1D array of length n
           
        
        
        # broadcasting to create n x n interaction matrices, instead of nested for loops
        # element at (row i, column j) corresponds to the interaction
        # between initial state i and final state j
        
        # abs_fi will be a column vector (n, 1)
        # abs_fj will be a row vector (1, n)
        
        
        abs_fi = abs_f[:, np.newaxis]
        abs_fj = abs_f[np.newaxis, :]
        
        
        
        
        Ep_flat = Ep.flatten()
        Em_flat = Em.flatten()
        Ep_flat_i = Ep_flat[:, np.newaxis]
        Ep_flat_j = Ep_flat[np.newaxis, :]
        Em_flat_i = Em_flat[:, np.newaxis]
        Em_flat_j = Em_flat[np.newaxis, :]
        
        
        fdp = self.fd_dist(Ep_flat, self.mu)
        fdm = self.fd_dist(Em_flat, self.mu)
        fdp_i = fdp[:, np.newaxis]
        fdp_j = fdp[np.newaxis, :]
        fdm_i = fdm[:, np.newaxis]
        fdm_j = fdm[np.newaxis, :]
        
                
        delta_pp = -self.dirac_delta(Ep_flat_i, fdp_i, Ep_flat_j, fdp_j)
        delta_mm = -self.dirac_delta(Em_flat_i, fdm_i, Em_flat_j, fdm_j)
        delta_pm = -self.dirac_delta(Ep_flat_i, fdp_i, Em_flat_j, fdm_j)
        delta_mp = -self.dirac_delta(Em_flat_i, fdm_i, Ep_flat_j, fdp_j)
        
        

        # Broadcasting: performing an operation like abs_fi - abs_fj, 
        # NumPy sees the mismatched shapes (n, 1) and (1, n), then virtually "stretches"
        # both arrays to a common shape (n, n). 
        M_pm = (np.abs(cos1 - cos2 * (abs_fi - abs_fj))**2)*delta_pm
        M_mp = (np.abs(cos1 - cos2 * (-abs_fi + abs_fj))**2)*delta_mp
        M_pp = (np.abs(cos1 - cos2 * (abs_fi + abs_fj))**2)*delta_pp
        M_mm = (np.abs(cos1 - cos2 * (-abs_fi - abs_fj))**2)*delta_mm
        
      
        
        # For band Ep
        # Sum over all final states for M_pm, but only j!=i for M_pp
        # sums all the elements along the columns (axis=1) for each row
        # The result is a 1D array of size n, where the i-th element is the sum over j of M_pm
        # we subtract the diagonal elemetns of M_pp since they correspond to same initial and final state
        rate_p = np.sum(M_pm, axis=1) + (np.sum(M_pp, axis=1) - np.diag(M_pp))

        # For band Em
        rate_m = np.sum(M_mp, axis=1) + (np.sum(M_mm, axis=1) - np.diag(M_mm))

        # Get the rates into the original 2D grid shape for plotting
        rates = np.vstack((rate_p, rate_m)).T
        #rates is now (n,2) array
        rates_grid = rates.reshape(self.kx.shape[0], self.kx.shape[1], 2)
        #reshapes into the original grid shape of the k-points, (N, N, 2)
        
        
        #multiply by 2pi coming from fermi golden rule formula
        #divide by number of sampled k points
        return 2*np.pi*rates_grid / N




    def fd_dist (self, E,mu):
        # Boltzmann constant in eV/K
        kb = 8.61733e-5
        return 1/(np.exp((E-mu)/(kb*self.T))+1)
    
    def fd_dist_der (self, E,mu):
        # Boltzmann constant in eV/K
        kb = 8.61733e-5
        return -np.exp((E-mu)/(kb*self.T))/(((np.exp((E-mu)/(kb*self.T))+1)**2)*(kb*self.T))

    def dirac_delta(self, E_i, f_i, E_j, f_j):
        #all things with i index should be (n,1) arrays,
        #all things with j index should be (1,n) arrays
        
      
        # broadcasted (n,n) array
        delta_E = E_i - E_j

        # finite-difference ratio (n,n) array, ignoring warnings about division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (f_i - f_j) / delta_E

        # Find the locations where the energy difference is effectively zero
        zero_mask = np.isclose(delta_E, 0)

        # For these locations, calculate the analytical derivative    
        # der is (n,1) but will be properly stretched by NumPy's broadcasting through np.where()    
        der = self.fd_dist_der(E_i,self.mu)

        # Use the analytical derivative where the energy difference is zero,
        # and the finite-difference ratio everywhere else.
        # result is (n,n)
        
        
        return np.where(zero_mask, der, ratio)
        
        
     
    def cal_e_density(self, mu, N):  
        
        
        fd_Ep = self.fd_dist(Ep,mu)
        fd_Em =  self.fd_dist(Em,mu)
        
        
        return (np.sum(fd_Ep) + np.sum(fd_Em)) / N

    def calc_mu(self, Ep, Em, tol=1e-6, max_iter=100):
        
        Ep_flat = Ep.flatten()
        Em_flat = Em.flatten()
        N = Ep_flat.size
        
        # search boundaries for mu
        mu_low = np.min(Em_flat) 
        mu_high = np.max(Ep_flat) 

        # Bisection loop
        for i in range(max_iter):
            mu_mid = (mu_low + mu_high) / 2
            n_mid = self.cal_e_density(mu_mid, N)
            
            error = n_mid - self.N_e_unitcell
            
            if abs(error) < tol:
                return mu_mid
            
            if error < 0:
                # n_mid is too low. Shift search range up.
                mu_low = mu_mid
            else:
                # n_mid is too high. Shift search range down.
                mu_high = mu_mid
                
        print(f"Bisection did not converge for n_e_unitcell={self.N_e_unitcell} within {max_iter} iterations.")
        return mu_mid
       
        
        
        
    def scatterPlot(self, Ep, Em):
        # generic function plot band energy points colored by scattering rate
        
        # find scattering rates at each kx,ky
        rates = self.scatterRate(self.kx, self.ky, Ep, Em)
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,  projection="3d")

        
        # set up shared normalization across both bands
        norm = Normalize(vmin=rates.min(), vmax=rates.max())
        cmap = plt.cm.viridis
        # compute facecolors for both bands using the same normalization
        facecolors_Ep = cmap(norm(rates[:,:,0]))
        facecolors_Em = cmap(norm(rates[:,:,1]))
        
        
        # alpha controls transparency, linewidth=0 removes grid lines on surface
        # facecolors uses a colormap to color the surface based on scattering rates
        surf1 = ax.plot_surface(self.c1, self.c2, Ep, alpha=0.8, linewidth=0, facecolors=facecolors_Ep)
        surf2 = ax.plot_surface(self.c1, self.c2, Em, alpha=0.8, linewidth=0, facecolors=facecolors_Em)


        ax.set_xlabel(r'$c1$')
        ax.set_ylabel(r'$c2$')
        ax.set_title('Band Energies Colored by Scattering Rate')
        
        # add one colorbar for both surfaces
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([]) 
        fig.colorbar(sm, ax=ax, label="Scattering rate")
        plt.show()



# lattice constant, C-C distance, in Angstroms
a = 1.42
# hopping term
t = 2

#vacancy potential
V0 = 10
#renormalisation of hopping due to vacancy
delta_t = -100

#electrons per unit cell (max 2)
N_e_unitcell = 1.1

# primitive lattice vectors for graphene
a1 = a*np.array([3/2, np.sqrt(3)/2])
a2 = a*np.array([3/2, -np.sqrt(3)/2])

# reciprocal lattice vectors
A = np.column_stack((a1, a2))
B = 2*np.pi * np.linalg.inv(A).T
b1 = B[:,0]
b2 = B[:,1]

# k-space grid 
N =150 # number of k-points per dimension, total points = N^2

#Temperature 
T = 300

# Create a grid of coefficients c1, c2 from -0.5 to 0.5
c_range = np.linspace(-0.5, 0.5, N)
C1, C2 = np.meshgrid(c_range, c_range)

# calculate the Cartesian kx, ky values 
KX = C1*b1[0] + C2*b2[0]
KY = C1*b1[1] + C2*b2[1]

H = Hamiltonian(a,N**2,N_e_unitcell,t,a1,a2,b1,b2,KX,KY,C1,C2,T,V0,delta_t)


#band structure
Ep, Em = H.band()
H.bandPlot(Ep, Em)

# calculate the correct chemical potential mu
mu = H.calc_mu(Ep, Em)
print(f"Chemical Potential mu for electron per unit cell n_e_unitcell={H.N_e_unitcell}: {mu:.4f} eV")
# update the mu attribute in the Hamiltonian object
H.mu = mu

# band surface with scattering rates
H.scatterPlot(Ep, Em)
