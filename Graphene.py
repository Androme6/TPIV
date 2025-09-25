import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.path import Path


class Hamiltonian:
    def __init__(self, a, t, KX, KY, V0, delta_t):
        self.a = a
        self.t = t
        self.kx = KX
        self.ky = KY
        self.V0 = V0
        self.delta_t = delta_t
        
        mask = self.BrillouinZone()
        self.n_unitcell = np.sum(mask)
    
    # band structure methods
    
    def band(self):
        # 2D tight binding with nearest-neighbor hopping, two atoms per cell and on honeycomb lattice
        val = self.t*np.sqrt(1 + 4*np.cos(1.5*self.kx*self.a)*np.cos(np.sqrt(3)/2*self.ky*self.a) + 4*np.cos(np.sqrt(3)/2*self.ky*self.a)**2)
        
        # Plot within the first BZ
        mask = self.BrillouinZone()
        # creates masked arrays (of the same size!) to only show points inside the BZ
        # ma.masked_where(condition, array) returns an array where elements are masked if condition is True
        # ~ negates the condition (bitwise NOT)
        Ep = np.ma.masked_where(~mask, val)
        Em = np.ma.masked_where(~mask, -val)
        
        return Ep, Em
    
    def bandPlot(self, Ep, Em):
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,  projection="3d")

        

        # alpha controls transparency, linewidth=0 removes grid lines on surface
        surf1 = ax.plot_surface(self.kx, self.ky, Ep, alpha=0.8, linewidth=0)
        surf2 = ax.plot_surface(self.kx, self.ky, Em, alpha=0.8, linewidth=0)


        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_zlabel("Energy")
        ax.set_title("Graphene Band Structure in 1st BZ")
        plt.show()
            
    # Fermi surface methods   
            
    def fermiEnergy(self, Ep,Em, n):
        # general method to compute Fermi energy from of band energies, independent of the specific Hamiltonian or lattice
               
        mask = self.BrillouinZone()
        # creates a new array (of size np.sum(mask)) containing only the elements
        # from the original array where the mask was True.
        Ep_bz = Ep[mask]
        Em_bz = Em[mask]
        E_bz = np.concatenate((Ep_bz, Em_bz))
        E_bz_flat = E_bz.ravel()
        E__bz_sorted = np.sort(E_bz_flat)
        
        
        print(f"Number of k-points in 1st BZ: {self.n_unitcell}")

       
        # Check for valid electron filling
        # no spin degeneracy, max 2 electrons per unit cell
        if n > 2: 
            raise ValueError(f"Too many electrons! Max filling is 2.")

        
        EF = E__bz_sorted[int(self.n_unitcell*n-1)]

        return EF
  
    def fermiContour(self, Ep, Em, EF):
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        
                
    
        ax.contour(self.kx, self.ky, Ep, levels=[EF])
        ax.contour(self.kx, self.ky, Em, levels=[EF])
        
        
        # draw hexagonal outline of BZ
        self.plotBZBoundary(ax)
        
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_title("Fermi Contours")
        plt.show()

    # Fermi points method
    
    def fermiPoints(self, Ep,Em, EF):
        # finds points in k-space where the band energy is tol close to EF
        
        
        tol = 1e-3

        # mask to identify points in k space whose energy is close to EF
        mask_p = np.isclose(Ep, EF, tol)
        kx_p, ky_p = self.kx[mask_p], self.ky[mask_p]
        # conduction band label s=+1
        s_p = np.ones_like(kx_p)   

        mask_m = np.isclose(Em, EF, tol)
        kx_m, ky_m = self.kx[mask_m], self.ky[mask_m]
        # valence band label s=-1
        s_m = -np.ones_like(kx_m)  

        # Concatenate results from both bands
        kx_all = np.concatenate([kx_p, kx_m])
        ky_all = np.concatenate([ky_p, ky_m])
        s_all  = np.concatenate([s_p, s_m])

        return kx_all, ky_all, s_all

  
    # Scattering rate methods

    #formFactor and M in scatterRate are specific to vacancy scattering in graphene
    # scatter is fully generic

   
    def formFactor(self, kx, ky, a):
        #form factor for graphene to help compute scattering rates
        return np.exp(1j*a*kx) + 2*np.exp(-1j*(a/2)*kx)*np.cos((np.sqrt(3)/2)*a*ky)
        
    def scatterRate(self, kx, ky, sBand):
        #number of fermi surface points
        nFS = kx.size
        rates = np.zeros(nFS)
        
        #values of form factor at each fermi surface point
        f_vals = self.formFactor(kx, ky, self.a)
        
        # double loop over all pairs of fermi surface points
        for i in range(0, nFS):
            for j in range(0, nFS):
                # only consider scattering between different points
                if j != i:
                    #impurity on A
                    M = (self.V0/(2*self.n_unitcell)) - (self.delta_t/(2*self.n_unitcell))*(sBand[i]*np.abs(f_vals[i]) + sBand[j]*np.abs(f_vals[j]))
                    rates[i] += np.abs(M)**2
           
       
        return rates
    
    def scatterPlot(self, Ep, Em, EF):
        # generic function plot fermi surface points colored by scattering rate
        
        # find fermi surface points
        kx_FS, ky_FS, s_FS = self.fermiPoints(Ep, Em, EF)
        if kx_FS.size == 0:
            print("No Fermi surface points found at this energy. Cannot plot scatter rate.")
            return
        
        # find scattering rates at each fermi surface point
        rates = self.scatterRate(kx_FS, ky_FS, s_FS)
        
        # allow for very small numerical variation in rates
        max_rate, min_rate = rates.max(), rates.min()
        relative_diff = 100 * (max_rate - min_rate) / min_rate
        print(f"Relative difference in scattering rates: {relative_diff:.2f}%")
        print(f"Max scattering rate: {max_rate:.15f}, Min scattering rate: {min_rate:.15f}")
        if relative_diff < 1:
            rates = np.full_like(rates, np.mean(rates))
        
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        
        # s is marker size
        graph = ax.scatter(kx_FS, ky_FS, s=3, array=rates)
        
        # draw outline of BZ
        self.plotBZBoundary(ax)
        
        
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_title('Fermi Surface')
        fig.colorbar(graph, ax=ax, label="Scattering rate")
        plt.show()
        
        
    # Brillouin zone methods
    
    # functions to define and plot the BZ
    # only bz_vertices is lattice specific
    
    def bz_vertices(self):
        # function to get the BZ vertices (Dirac points) for the specific honeycomb lattice
        
        kx_vert = 2 * np.pi / (3 * self.a)
        ky_vert1 = 2 * np.pi / (3 * np.sqrt(3) * self.a)
        ky_vert2 = 4 * np.pi / (3 * np.sqrt(3) * self.a)
        
        vertices = np.array([
            [kx_vert, ky_vert1],
            [0, ky_vert2],
            [-kx_vert, ky_vert1],
            [-kx_vert, -ky_vert1],
            [0, -ky_vert2],
            [kx_vert, -ky_vert1],
        ])
        
        return vertices
    
    def BrillouinZone(self):
        # Given a generic set of vertices, creates an accurate mask for the first BZ
        
        bz_vertices = self.bz_vertices()
        # Create a Path object from the BZ vertices
        # Path is a matplotlib class that represents a series of possibly disconnected, possibly closed, line and curve segments.
        path = Path(bz_vertices)

        # Creates a grid of all k-points from kx and ky
        # vstack stacks 1D arrays as columns to create a 2D array
        # so each row of points is a point (kx, ky)
        points = np.vstack((self.kx.ravel(), self.ky.ravel())).T
        
        # Create the mask by checking which points are inside the path
        # Path.contains_points returns whether the area enclosed by the path contains the given points.
        mask = path.contains_points(points)
        
        # so the mask goes from (N_k,) to (N_kx, N_ky)
        return mask.reshape(self.kx.shape)

    def plotBZBoundary(self, ax):
        # plots the generic BZ boundary by connecting the vertices

        bz_vertices = self.bz_vertices()
        # Create a closed loop by appending the first vertex to the end
        bz_path = np.vstack([bz_vertices, bz_vertices[0]])
        # k is for dashed line, lw is line width
        # x and y coordinates of the vertices are in the first and second columns of bz_path
        ax.plot(bz_path[:,0], bz_path[:,1], "k--", lw=1.5, label="1st Brillouin Zone")
        ax.legend()




        
# lattice constant
a = 1
# hopping term
t = 2

#vacancy potential
V0 = 10
#renormalisation of hopping due to vacancy
delta_t = -100


# k-space grid based on BZ hexagonal shape
N = 5000  # number of k-points per dimension, total points = N^2
kx_max = 2 * np.pi / (3 * a)
ky_max = 4 * np.pi / (3 * np.sqrt(3) * a)
kx_vec = np.linspace(-kx_max * 1.2, kx_max * 1.2, N)
ky_vec = np.linspace(-ky_max * 1.2, ky_max * 1.2, N)
KX, KY = np.meshgrid(kx_vec, ky_vec)


H = Hamiltonian(a,t,KX,KY,V0,delta_t)

#band structure
Ep, Em = H.band()
H.bandPlot(Ep, Em)

# electrons per unit cell
# max 2 (no spin degeneracy)
n_electrons_unitcell = 1.1

# evaluate Fermi energy
EF = H.fermiEnergy(Ep, Em, n_electrons_unitcell)
print(f"Fermi Energy corresponding to {n_electrons_unitcell} electrons per unit cell= {EF}")

# plot Fermi contours
H.fermiContour(Ep, Em, EF)

# fermi surface with scattering rates
H.scatterPlot(Ep, Em,EF)
