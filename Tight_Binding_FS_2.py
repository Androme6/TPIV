import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure


class Hamiltonian:
    def __init__(self, N, a, t, KX, KY, KZ, V0, delta_t):
        self.N = N
        self.a = a
        self.t = t
        self.kx = KX
        self.ky = KY
        self.kz = KZ
        self.V0 = V0
        self.delta_t = delta_t
        
    def band(self):
        # 3D tight binding with nearest-neighbor hopping and on cubic lattice
        return  -2*self.t*(np.cos(self.kx*self.a) + np.cos(self.ky*self.a)+ np.cos(self.kz*self.a))
        #return (-3+np.cos(0.5*self.a*self.kx)*np.cos(0.5*self.a*self.ky)+np.cos(0.5*self.a*self.kz)*np.cos(0.5*self.a*self.ky)+np.cos(0.5*self.a*self.kx)*np.cos(0.5*self.a*self.kz))+0.0995*(-3+np.cos(self.kx*self.a) + np.cos(self.ky*self.a)+ np.cos(self.kz*self.a))
        #return (-3+np.cos(0.5*self.a*self.kx)*np.cos(0.5*self.a*self.ky)+np.cos(0.5*self.a*self.kz)*np.cos(0.5*self.a*self.ky)+np.cos(0.5*self.a*self.kx)*np.cos(0.5*self.a*self.kz))+0.0995*(-3+np.cos(self.kx*self.a) + np.cos(self.ky*self.a)+ np.cos(self.kz*self.a))
    def fermiEnergy(self, E, n):
        
        # general method to compute Fermi energy from of band energies, independent of the specific Hamiltonian or lattice
        E_flat = E.ravel()  
        E_sorted = np.sort(E_flat)
            
        half_filling =  self.N**3 /2
         
        if n > 2:
            raise ValueError(f"Too many electrons! Max = {2*half_filling}")
        
        print(E_flat.size)
      
               
        EF = E_sorted[int(half_filling*n)]
        
        return EF
    
    def fermiPoints(self, E, EF):
       
        # mask to identify points in k space whose energy is close to EF
        mask = np.isclose(E, EF, atol=1e-1)
        return self.kx[mask], self.ky[mask], self.kz[mask], E[mask]
    
    def scatter(self, E, EF):
        
        
        
        kx_FS, ky_FS, kz_FS, _ = self.fermiPoints(E, EF)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
         
        rates = self.scatterRate(kx_FS, ky_FS, kz_FS, self.V0, self.delta_t)
        graph = ax.scatter(kx_FS, ky_FS, kz_FS, s=40, array=rates)
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_zlabel(r'$k_z$')
        ax.set_title('3D Fermi Surface')
        fig.colorbar(graph, ax=ax, label="Scattering rate")
        plt.show()
       
        
        
        
    def surface(self, E, EF):
        # marching_cubes takes the 3D array E and gives vertices and triangles of the EF isosurface
        # faces is an array of shape (n_faces, 3) containing the indices of the vertices that make up each triangular face
        vertices, faces, _, _ = measure.marching_cubes(E, level=EF)  
        scale = k[1]-k[0]   
        scaled_vertices = vertices*scale - np.pi/a
        
        kx = scaled_vertices[:, 0]
        ky = scaled_vertices[:, 1]
        kz = scaled_vertices[:, 2]

        
        rates = self.scatterRate(kx, ky, kz, self.V0, self.delta_t)
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        
        
        #graph = ax.scatter(kx, ky, kz, s=40, cmap="plasma", array=rates)
        


      
        # faces is an array of shape (n_triangles, 3) containing the coordinates of the vertices that make up each triangular face
        verts = [list(zip(kx[tri], ky[tri], kz[tri])) for tri in faces]
        #for each triangle, compute average scattering rate of its vertices. avg_rates is a 1D array (n_triangles)
        avg_rates = np.mean(rates[faces], axis=1)
        #norm is a Normalize object to map avg_rates to [0,1] for colormap
        norm = plt.Normalize(np.min(avg_rates), np.max(avg_rates))
        cmap = plt.get_cmap("spring")
        #this is a list of RGBA colors for each triangle, mapped from avg_rates using the colormap
        face_colors = cmap(norm(avg_rates))
        #create a Poly3DCollection for the triangular faces
        poly = Poly3DCollection(verts, facecolors=face_colors, edgecolor='k', linewidths=0.05)
        #adds the colored triangular faces to the 3D plot
        ax.add_collection3d(poly)
        #this creates a ScalarMappable for the colorbar, using the same colormap and normalization
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(avg_rates)
        



        #graph = ax.plot_trisurf(kx, ky, kz, triangles=faces, cmap="spring", array=rates, shade=False)



        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_zlabel(r'$k_z$')
        ax.set_title('3D Fermi Surface')
        fig.colorbar(mappable, ax=ax, label="Scattering rate")
        plt.show()
        
        
      
        
    def scatterRate(self, kX, kY, kZ, V0, delta_t):
        nFS = kX.size
        rates = np.zeros(nFS)
        for i in range(0, nFS):
            for j in range(0, nFS):
                if j != i:
                    rates[i] += (V0 + 2*delta_t*(np.cos(self.a*(kX[i]-kX[j]))+np.cos(self.a*(kY[i]-kY[j]))+np.cos(self.a*(kZ[i]-kZ[j]))))**2
                    
            rates[i] /= nFS        
        print(rates)
        return rates
    

# lattice constant
a = np.pi
# number of sites per dimension
N = 20
# hopping term
t = 2

#vacancy potential
V0 = 0
delta_t = 10

# cubic lattice
k = np.linspace(-np.pi/a, np.pi/a, N)
KX, KY, KZ = np.meshgrid(k,k,k)


H = Hamiltonian(N,a,t,KX,KY,KZ,V0,delta_t)
E = H.band()


n_electrons_unitcell = 0.5

EF = H.fermiEnergy(E, n_electrons_unitcell)
print(f"Fermi Energy corresponding to {n_electrons_unitcell} electrons per unit cell= {EF}")

H.scatter(E,EF)
H.surface(E,EF)



