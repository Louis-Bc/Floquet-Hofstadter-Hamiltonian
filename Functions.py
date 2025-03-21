import numpy as np
from scipy.linalg import expm

# This file contains functions that build tight-binding Hamiltonians for a 2D square lattice subject to 
# a magnetic field. 

# The following functions find the energy eigenvalues for Hamiltonians in a real space basis.

def basis(x, y, size):
    """
    Takes in a point (x,y) and returns a basis vector associated with that point.
    
    Parameters
    ----------
    x : int
        The x-coordinate of a site on an N by N square lattice in units of the lattice parameter a.
    y : int
        The y-coordinate of a site on an N by N square lattice in units of the lattice parameter a.
    size : int
        The number of atoms along each side of the square lattice.
    
    Returns
    -------
    list
        The column of a size by size identity matrix representing the point (x,y) in a new basis.
    """
    r = np.identity(size**2)[:, size*y+x]
    return r

def H_real(size, alpha, t_nn = 1, t_nnn = 0.2, hop = 'nn', E_0 = 0):
    """
    Returns the single particle energy eigenvalues of the tight-binding Hamiltonian in real space for a 
    size by size square lattice subject to a perpendicular magnetic field.
    
    Parameters
    ----------
    size : int
        The number of atoms along each side of the square lattice.
    alpha : float
        A dimensionless measure of the magnetic flux through a unit cell of the lattice; must be a
        rational number for this model to be accurate.
    t_nn : float, optional
        The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
        lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be between 
        0 and 1.
    t_nnn : float, optional
        The next-nearest neighbor hopping integral, providing a measure for how likely an electron on 
        the lattice is to hop to next-nearest neighbor sites. Defaults to 0.2. Values passed in should be 
        between 0 and 1.
    hop : str, optional
        Should be either 'nn' to specify that only nearest neighbor hopping should be included or 'nnn' 
        to specify that both nearest and next-nearest neighbor hopping should be included. Defaults to
        'nn'.
    E_0 : float, optional
        The chemical potential energy associated with each site on the lattice. Defaults to 0.
        
    Returns
    -------
    ndarray
        A one dimensional numpy array containing the energy eigenvalues of the Hamiltonian.
    """
    Phi = 2*np.pi*alpha
    H_hop = np.zeros([size**2, size**2], dtype=np.complex128)
    H_stat = E_0*np.identity(size**2)
    
    if hop == 'nn':
        for n in range(size):
            for m in range(size):
                if m<size-1 and n<size-1:
                    #if not on edges, hopping is normal and periodic boundary conditions do not have to be applied
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(m+1,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,n+1,size)))
                elif m==size-1 and n<size-1: 
                    #if at the right edge, but not on the corner, hopping right loops back to x=0 and hopping up is normal
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(0,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,n+1,size)))
                elif n==size-1 and m<size-1:
                    #if at the top edge, but not on the corner, hopping right is normal and hopping up loops back to x=0
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(m+1,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,0,size))*np.exp(1j*Phi*size*m))
                elif m==size-1 and n==size-1:
                    #if on corner, hopping right loops back to x=0 and hopping up loops back to y=0
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(0,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,0,size))*np.exp(1j*Phi*size*m))

    if hop == 'nnn':
        for n in range(size):
            for m in range(size):
                if m<size-1 and 0<n<size-1:
                    #if not on edges, hopping is normal and periodic boundary conditions do not have to be applied
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(m+1,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,n+1,size)))
                    #next-nearest neighbor hopping
                    H_hop += t_nnn*(np.outer(basis(m,n,size), basis(m+1,n+1,size))*np.exp(-1j*(2*n+1)*Phi/2) + 
                                    np.outer(basis(m,n,size), basis(m+1,n-1,size))*np.exp(-1j*(2*n-1)*Phi/2))
                elif m==size-1 and 0<n<size-1: 
                    #if at the right edge, but not on the corner, hopping right loops back to x=0 and hopping up is normal
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(0,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,n+1,size)))
                    #next-nearest neighbor hopping
                    H_hop += t_nnn*(np.outer(basis(m,n,size), basis(0,n+1,size))*np.exp(-1j*(2*n+1)*Phi/2) + 
                                    np.outer(basis(m,n,size), basis(0,n-1,size))*np.exp(-1j*(2*n-1)*Phi/2))
                elif n==0 and m<size-1:
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(m+1,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,n+1,size)))
                    #next-nearest neighbor hopping
                    H_hop += t_nnn*(np.outer(basis(m,n,size), basis(m+1,n+1,size))*np.exp(-1j*(2*n+1)*Phi/2) + 
                                    np.outer(basis(m,n,size), basis(m+1,size-1,size))*np.exp(1j*(1-2*size-2*size*m)*Phi/2))
                elif n==0 and m==size-1:
                    #if at the right edge, but not on the corner, hopping right loops back to x=0 and hopping up is normal
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(0,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,n+1,size)))
                    #next-nearest neighbor hopping
                    H_hop += t_nnn*(np.outer(basis(m,n,size), basis(0,n+1,size))*np.exp(-1j*(2*n+1)*Phi/2) + 
                                    np.outer(basis(m,n,size), basis(0,size-1,size))*np.exp(1j*(1-2*size**2)*Phi/2))
                elif n==size-1 and m<size-1:
                    #if at the top edge, but not on the corner, hopping right is normal and hopping up loops back to x=0
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(m+1,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,0,size))*np.exp(1j*Phi*size*m))
                    #next-nearest neighbor hopping
                    H_hop += t_nnn*(np.outer(basis(m,n,size), basis(m+1,0,size))*np.exp(1j*(1+2*size*m)*Phi/2) + 
                                    np.outer(basis(m,n,size), basis(m+1,n-1,size))*np.exp(1j*(3-2*size)*Phi/2))
                elif m==size-1 and n==size-1:
                    #if on corner, hopping right loops back to x=0 and hopping up loops back to y=0
                    H_hop += t_nn*(np.outer(basis(m,n,size), basis(0,n,size))*np.exp(-1j*Phi*n) + 
                                   np.outer(basis(m,n,size), basis(m,0,size))*np.exp(1j*Phi*size*m))
                    #next-nearest neighbor hopping
                    H_hop += t_nnn*(np.outer(basis(m,n,size), basis(0,0,size))*np.exp(1j*(1-2*size)*Phi/2) + 
                                    np.outer(basis(m,n,size), basis(0,n-1,size))*np.exp(1j*(3-2*size)*Phi/2))
                
    H = H_stat - H_hop - np.transpose(np.conjugate(H_hop))
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    energies = eigenvalues
    return(energies)

def H_k(k_x, k_y, q, p = 1, hop = 'nn', t_nn = 1, t_nnn = 0.2):
    """
        Outputs the energy eigenvalues and eigenvectors of the q-dimensional Hamiltonian in the k-space 
        basis.
        
        Parameters
        ----------
        k_x : float
            The x-component of the wavevector k for which the Hamiltonian should be constructed and
            diagonalized.
        k_y : float
            The y-component of the wavevector k for which the Hamiltonian should be constructed and
            diagonalized.
        q : int
            The denominator of the dimensionless measure of magnetic flux alpha. Also sets the size of 
            the Hamiltonian in the k-space basis.
        p : int, optional
            The numerator of the dimensionless measure of magnetic flux alpha. Defaults to 1
        hop : str, optional
            Should be either 'nn' to specify that only nearest neighbor hopping should be included or 
            'nnn' to specify that both nearest and next-nearest neighbor hopping should be included. 
            Defaults to 'nn'.
        t_nn : float, optional
            The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
            lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be 
            between 0 and 1.
        t_nnn : float, optional
            The next-nearest neighbor hopping integral, providing a measure for how likely an electron on 
            the lattice is to hop to next-nearest neighbor sites. Defaults to 0.2. Values passed in 
            should be between 0 and 1.
        
        Returns
        -------
        tuple
            The first element is an 1D ndarray of the energy eigenvalues of the q by q Hamiltonian and 
            the second element is a 2D ndarray, containing the eigenvectors of the Hamiltonian. The 
            eigenvectors are oriented along the columns of the 2D array.
    """
    Phi = 2*np.pi*p/q
    H = np.zeros([q, q], dtype=np.complex128)
    
    if q == 2 or q == -2:
        if q == 2:
            sign = 1
        elif q == -2:
            sign = -1
        if hop == 'nn':
            H[0, 0] = -2*t_nn*np.cos(k_x)
            H[1, 1] = 2*t_nn*np.cos(k_x)
            H[0, 1] = -2*t_nn*np.cos(k_y)
            H[1, 0] = -2*t_nn*np.cos(k_y)
        elif hop == 'nnn':
            H[0, 0] = -2*t_nn*np.cos(k_x)
            H[1, 1] = 2*t_nn*np.cos(k_x)
            H[0, 1] = -2*(t_nn*np.cos(k_y) + 1j*2*sign*t_nnn*np.sin(k_x)*np.sin(k_y))
            H[1, 0] = -2*(t_nn*np.cos(k_y) - 1j*2*sign*t_nnn*np.sin(k_x)*np.sin(k_y))
    else:
        if hop == 'nn':
            for i in range(abs(q)):
                H[i, i] = -2*t_nn*np.cos(k_x - Phi*i)
                if i==0:
                    H[i, i+1] = -t_nn*np.exp(1j*k_y)
                    H[i, abs(q)-1] = -t_nn*np.exp(-1j*k_y)
                elif i==abs(q)-1:
                    H[i, i-1] = -t_nn*np.exp(-1j*k_y)
                    H[i, 0] = -t_nn*np.exp(1j*k_y)
                else:
                    H[i, i+1] = -t_nn*np.exp(1j*k_y)
                    H[i, i-1] = -t_nn*np.exp(-1j*k_y)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            return(eigenvalues, eigenvectors)
        
        elif hop == 'nnn':
            for i in range(abs(q)):
                H[i, i] = -2*t_nn*np.cos(k_x - Phi*i)
                if i==0:
                    H[i, i+1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*i + 1)*Phi/2)))*np.exp(1j*k_y)
                    H[i, abs(q)-1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*(abs(q)-1) + 1)*Phi/2)))*np.exp(-1j*k_y)
                elif i==abs(q)-1:
                    H[i, i-1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*(i-1) + 1)*Phi/2)))*np.exp(-1j*k_y)
                    H[i, 0] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*i + 1)*Phi/2)))*np.exp(1j*k_y)
                else:
                    H[i, i+1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*i + 1)*Phi/2)))*np.exp(1j*k_y)
                    H[i, i-1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*(i-1) + 1)*Phi/2)))*np.exp(-1j*k_y)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return(eigenvalues, eigenvectors)
    
def h_k_reduced(k_x, k_y, q_test, q_bands, hop = 'nn', p_test=1, t_nn=1, t_nnn=0.2):
    ''' 
        Returns a q_bands by q_bands matrix representing the Hamiltonian for a constant flux per plaquette of p/q
    
        Parameters
        ----------
            k_x : float
                The x-component of the wavevector k for which the Hamiltonian should be constructed and
                diagonalized.
            k_y : float
                The y-component of the wavevector k for which the Hamiltonian should be constructed and
                diagonalized.
            q_test : int
                The denominator of the dimensionless measure of static magnetic flux per unit cell alpha.
            q_bands : int
                The size of the Floquet Hamiltonian for flux switching between p_test/q_test and another flux p/q; q_bands is 
                given by lcm(q_test, q) if q_test and q are incommensurate or max(q_test, q) if they are commensurate.
            hop : str, optional
                Should be either 'nn' to specify that only nearest neighbor hopping should be included or 
                'nnn' to specify that both nearest and next-nearest neighbor hopping should be included. 
                Defaults to 'nn'.
            p_test : int, optional
                The numerator of the dimensionless measure of static magnetic flux per unit cell alpha. Default value is set to 1.
            t_nn : float, optional
                The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
                lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be 
                between 0 and 1.
            t_nnn : float, optional
                The next-nearest neighbor hopping integral, providing a measure for how likely an electron on 
                the lattice is to hop to next-nearest neighbor sites. Defaults to 0.2. Values passed in 
                should be between 0 and 1.

        Returns
        -------
        tuple
            The first element is an 1D ndarray of the energy eigenvalues of the q_bands by q_bands Hamiltonian and 
            the second element is a 2D ndarray, containing the eigenvectors of the Hamiltonian. The 
            eigenvectors are oriented along the columns of the 2D array.
    '''
    
    H = np.zeros([q_bands,q_bands], dtype=np.complex128)
    Phi = 2*np.pi/q_test
    
    if q_test == 2 or q_test == -2:
        if q_test == 2:
            sign = 1
        elif q_test == -2:
            sign = -1
        if hop == 'nn':
            H[0, 0] = -2*t_nn*np.cos(k_x)
            H[1, 1] = 2*t_nn*np.cos(k_x)
            H[0, 1] = -2*t_nn*np.cos(k_y)
            H[1, 0] = -2*t_nn*np.cos(k_y)
        elif hop == 'nnn':
            H[0, 0] = -2*t_nn*np.cos(k_x)
            H[1, 1] = 2*t_nn*np.cos(k_x)
            H[0, 1] = -2*(t_nn*np.cos(k_y) + 1j*2*sign*t_nnn*np.sin(k_x)*np.sin(k_y))
            H[1, 0] = -2*(t_nn*np.cos(k_y) - 1j*2*sign*t_nnn*np.sin(k_x)*np.sin(k_y))
    else:
        if hop == 'nn':    
            for i in range(abs(q_bands)):
                H[i, i] = -2*t_nn*np.cos(k_x - Phi*i)
                if i==0:
                    H[i, i+1] = -t_nn*np.exp(1j*k_y)
                    H[i, abs(q_bands)-1] = -t_nn*np.exp(-1j*k_y)
                elif i==abs(q_bands)-1:
                    H[i, i-1] = -t_nn*np.exp(-1j*k_y)
                    H[i, 0] = -t_nn*np.exp(1j*k_y)
                else:
                    H[i, i+1] = -t_nn*np.exp(1j*k_y)
                    H[i, i-1] = -t_nn*np.exp(-1j*k_y)
        
        elif hop == 'nnn':
            for i in range(abs(q_bands)):
                H[i, i] = -2*t_nn*np.cos(k_x - Phi*i)
                if i==0:
                    H[i, i+1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*i + 1)*Phi/2)))*np.exp(1j*k_y)
                    H[i, abs(q_bands)-1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*(q_bands-1) + 1)*Phi/2)))*np.exp(-1j*k_y)
                elif i==abs(q_bands)-1:
                    H[i, i-1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*(i-1) + 1)*Phi/2)))*np.exp(-1j*k_y)
                    H[i, 0] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*i + 1)*Phi/2)))*np.exp(1j*k_y)
                else:
                    H[i, i+1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*i + 1)*Phi/2)))*np.exp(1j*k_y)
                    H[i, i-1] = -(t_nn + (2*t_nnn*np.cos(k_x - (2*(i-1) + 1)*Phi/2)))*np.exp(-1j*k_y)
    
    eigenvals, eigenvecs = np.linalg.eigh(H)
    
    ind = np.argsort(eigenvals)
    eigenvalues = eigenvals[ind]
    eigenvectors = eigenvecs[:, ind]
    
    return(eigenvalues, eigenvectors)

def H_F_exact(k_x, k_y, q_a, q_b, q, T_a, T_b, hop = 'nn', t_nn=1, t_nnn = 0.2):
    '''
        Builds the q by q Floquet Hamiltonian for periodic switching between fluxes 1/q_a and 1/q_b
    
        Parameters
        ----------
            k_x : float
                The x-component of the wavevector k for which the Hamiltonian should be constructed and
                diagonalized.
            k_y : float
                The y-component of the wavevector k for which the Hamiltonian should be constructed and
                diagonalized.
            q_a : int
                The denominator of the dimensionless measure of magnetic flux per unit cell alpha_a. 
            q_b : int
                The denominator of the dimensionless measure of magnetic flux per unit cell alpha_b. 
            q : int
                The size of the Floquet Hamiltonian for flux switching between 1/q_a and 1/q_b; given by lcm(q_a, q_b) for 
                incommensurate q_a, q_b or max(q_a, q_b) for commensurate q_a, q_b.
            T_a : float
                The period for which alpha = 1/q_a for a flux switching routine with total period T = T_a + T_b.
            T_b : float
                The period for which alpha = 1/q_b for a flux switching routine with total period T = T_a + T_b.
            hop : str, optional
                Should be either 'nn' to specify that only nearest neighbor hopping should be included or 
                'nnn' to specify that both nearest and next-nearest neighbor hopping should be included. 
                Defaults to 'nn'.
            t_nn : float, optional
                The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
                lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be 
                between 0 and 1.
            t_nnn : float, optional
                The next-nearest neighbor hopping integral, providing a measure for how likely an electron on the 
                lattice is to hop to next-nearest neighbor sites. Defaults to 0.2. Values passed in should be 
                between 0 and 1.
                
        Returns
        -------
        tuple
            The first element is an 1D ndarray of the energy eigenvalues of the q by q Hamiltonian and 
            the second element is a 2D ndarray, containing the eigenvectors of the Hamiltonian. The 
            eigenvectors are oriented along the columns of the 2D array.
    '''
    T = T_a + T_b
    eigval_a, eigvec_a = h_k_reduced(k_x, k_y, q_a, q, hop, t_nn, t_nnn)
    eigval_b, eigvec_b = h_k_reduced(k_x, k_y, q_b, q, hop, t_nn, t_nnn)
        
    D_a = np.identity(q) * np.exp(-1j*T_a*eigval_a)
    D_b = np.identity(q) * np.exp(-1j*T_b*eigval_b)
    
    U_a = eigvec_a @ D_a @ np.transpose(np.conj(eigvec_a))
    U_b = eigvec_b @ D_b @ np.transpose(np.conj(eigvec_b))
    
    M = U_a @ U_b
    eigval_M, eigvec_M = np.linalg.eig(M)
    #Sorting eigenvalues and eigenvectors
    idx_M = np.argsort(eigval_M, stable="True")
    eigval_M = eigval_M[idx_M]
    eigvec_M = eigvec_M[:, idx_M]
    
    D_M = np.log(eigval_M) * np.identity(q)
    H_F = (1j/T) * eigvec_M @ D_M @ np.transpose(np.conj(eigvec_M))
    eigenvalues, eigenvectors = np.linalg.eigh(H_F)
    
    #Connecting bands that go out of the first Floquet zone
    if int(sum(eigenvalues)) > 0:
        eigenvalues[q-1] = eigenvalues[q-1] - 2*np.pi/T
    elif int(sum(eigenvalues)) < 0:
        eigenvalues[0] = eigenvalues[0] + 2*np.pi/T
        
    ind = np.argsort(eigenvalues, stable="True")
    quasienergies = eigenvalues[ind]
    eigenvectors = eigenvectors[:, ind]
    
    return(quasienergies, eigenvectors)
    
def h_k_approx_Floquet(k_x, k_y, q_1, q_2, t_1, t_2, p_1=1, p_2=1, t=1):
    """
        Builds the effective Floquet Hamiltonian approximated up to the first-order correction of the Baker-
        Campbell-Haussdorf Formula in the k-space basis when alpha switches between p_1/q_1 and p_2/q_2 in a
        square wave fashion.
        
        Parameters
        ----------
            k_x : float
                The x-component of the wavevector k for which the Hamiltonian should be constructed and
                diagonalized.
            k_y : float
                The y-component of the wavevector k for which the Hamiltonian should be constructed and
                diagonalized.
            q_1 : int
                The denominator of the dimensionless measure of magnetic flux per unit cell alpha_1.
            q_2 : int
                The denominator of the dimensionless measure of magnetic flux per unit cell alpha_2.
            t_1 : float
                The fraction T_1/T, where T is one full period T_1 + T_2 of the switching routine and T_1
                is the period during which the dimensionless flux per unit cell is alpha_1. Should be a 
                value between 0 and 1.
            t_2 : float
                The fraction T_2/T, where T_2 is the period during which the dimensionless flux per unit 
                cell is alpha_2. 
            p_1 : int, optional
                The numerator of the dimensionless measure of magnetic flux per unit cell alpha_1. Defaults
                to 1.
            p_2 : int, optional
                The numerator of the dimensionless measure of magnetic flux per unit cell alpha_2. Defaults
                to 1.
            t : int, optional
                The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
                lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be 
                between 0 and 1.
                
        Returns
        -------
        tuple
            The first element is an 1D ndarray of the energy eigenvalues of the q_bands by q_bands Hamiltonian and 
            the second element is a 2D ndarray, containing the eigenvectors of the Hamiltonian. The 
            eigenvectors are oriented along the columns of the 2D array.
    """
    # If q_1 and q_2 are incommensurate, q_bands is the least common multiple of the two integers,
    # otherwise, q_bands is the larger of the two integers.
    if q_1 == 0:
        Phi_1 = 0
        Phi_2 = 2*np.pi*p_2/q_2
        q_bands = q_2
    elif q_2 == 0:
        Phi_2 = 0
        Phi_1 = 2*np.pi*p_1/q_1
        q_bands = q_1
    else:
        Phi_1 = 2*np.pi*p_1/q_1
        Phi_2 = 2*np.pi*p_2/q_2
    
        if q_2>q_1:
            if (q_2/q_1).is_integer():
                q_bands = max(q_1, q_2)
            else:
                q_bands = np.lcm(q_1, q_2)
        elif q_1>q_2:
            if (q_1/q_2).is_integer():
                q_bands = max(q_1, q_2)
            else:
                q_bands = np.lcm(q_1, q_2)

    H_0 = np.zeros([q_bands,q_bands], dtype=np.complex128)
    
    #Zero-th order terms:
    
    #Edge case with periodic boundary conditions
    H_0[q_bands-1, 0] = (t_1 + t_2)*np.exp(1j*k_y)
    #Off-diagonal terms
    for i in range(q_bands-1):
        H_0[i, i+1] += (t_1 + t_2)*np.exp(1j*k_y)
    #Diagonal terms
    for i in range(q_bands):
        H_0[i, i] += t_1*np.cos(k_x + Phi_1*i) + t_2*np.cos(k_x + Phi_2*i)
    
    #Adding hermitian conjugate
    H_0 += np.transpose(np.conj(H_0))
    
    #Commutator terms
    H_com = np.zeros([q_bands,q_bands], dtype=np.complex128)
    for i in range(q_bands-1):
        H_com[i, i+1] += -((np.exp(-1j*Phi_1*(i+1)) - np.exp(-1j*Phi_2*(i+1)) - 
                           np.exp(-1j*Phi_1*i) + np.exp(-1j*Phi_2*i))*np.exp(1j*(k_x + k_y)) + 
                          (np.exp(1j*Phi_1*(i+1)) - np.exp(1j*Phi_2*(i+1)) - 
                           np.exp(1j*Phi_1*i) + np.exp(1j*Phi_2*i))*np.exp(1j*(k_y - k_x)))
    H_com[q_bands-1, 0] = ((np.exp(-1j*Phi_1*q_bands) - np.exp(-1j*Phi_2*q_bands))*np.exp(1j*(k_x - q_bands*k_y)) + 
                            (np.exp(1j*Phi_1*q_bands) - np.exp(1j*Phi_2*q_bands))*np.exp(-1j*(k_x + q_bands*k_y)))
    #Subtracting hermitian conjugate so that the commutator is antihermitian
    H_com += -np.transpose(np.conj(H_com))
    
    H = -t*H_0 - t**2*(t_1*t_2*1j/2)*H_com
    
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    ind = np.argsort(eigenvalues, stable="True")
    quasienergies = eigenvalues[ind]
    eigenvectors = eigenvectors[:, ind]
    
    return(quasienergies, eigenvectors)

def H_F_real(N, q_1, q_2, T_1, T_2, hop_type = 'nn', t_nn = 1, t_nnn = 0.2, p_1 = 1, p_2 = 1):
    """
        Builds the Floquet Hamiltonian in real space for an electron on an NxN square lattice with magnetic flux per plaquette which
        periodically switches between p_1/q_1 for time T_1 and p_2/q_2 for time T_2.
        
        Parameters
        ----------
        N : int
            The number of atoms along each side of the square lattice.
        q_1 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_1.
        q_2 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_2.
        T_1 : float
            The period for which alpha = 1/q_1 for a flux switching routine with total period T = T_1 + T_2.
        T_2 : float
            The period for which alpha = 1/q_2 for a flux switching routine with total period T = T_1 + T_2.
        hop_type : str, optional
            Should be either 'nn' to specify that only nearest neighbor hopping should be included or 
            'nnn' to specify that both nearest and next-nearest neighbor hopping should be included. 
            Defaults to 'nn'.
        t_nn : float, optional
            The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
            lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be 
            between 0 and 1.
        t_nnn : float, optional
            The next-nearest neighbor hopping integral, providing a measure for how likely an electron on the 
            lattice is to hop to next-nearest neighbor sites. Defaults to 0.2. Values passed in should be 
            between 0 and 1.
        p_1 : int, optional
            The numerator of the dimensionless measure of magnetic flux per unit cell alpha_1. Defaults
            to 1.
        p_2 : int, optional
            The numerator of the dimensionless measure of magnetic flux per unit cell alpha_2. Defaults
            to 1.
            
        Returns
        -------
        ndarray
            A one dimensional numpy array containing the quasienergies of the Floquet Hamiltonian in real space.
    """
    T = T_1 + T_2
    alpha_1 = p_1/q_1
    alpha_2 = p_2/q_2
    
    H_1 = np.array(H_real(N, alpha_1, t_nn, t_nnn, hop = hop_type))/1000
    H_2 = np.array(H_real(N, alpha_2, t_nn, t_nnn, hop = hop_type))/1000
    Trot_mtx = expm(-1j*H_1*T_1) @ expm(-1j*H_2*T_2)
    eigenvalues, eigenvectors = np.linalg.eig(Trot_mtx)
    quasiE = (1j/T)*np.log(eigenvalues)*1000
    return(quasiE)

def H_chiral(N, k_x, q, p=1, t=1):
    """
        Builds a Hamiltonian which simulates a system with periodic boundary conditions in the x-direction
        and open boundary conditions in the y-direction, allowing for chiral edge states to be hosted by
        the system.
        
        Parameters
        ----------
        N : int
            The number of atomic sites along each side of the simulated square lattice. The Hamiltonian 
            matrix built by this function is N by N.
        k_x : float
            The wavenumber of a plane wave in the Fourier series of the Fermionic creation and annihilation 
            operators in the Hamiltonian.
        q : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha.
        p : int, optional
            The numerator of the dimensionless measure of magnetic flux per unit cell alpha_1. Defaults to 1.
        t : float, optional
            The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
            lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be between 
            0 and 1.
            
        Returns
        -------
        ndarray
            A 2D array containing the elements of the N by N Hamiltonian for a square lattice which is 
            infinite along the x-direction and contains N sites along the y-direction.
    """
    H = np.zeros([N,N], dtype=np.complex128)
    if q == 0:
        Phi = 0
    else:
        Phi = 2*np.pi*p/q
        
    for n in range(N):
        H[n,n] += -t*np.exp(1j*(k_x - Phi*n)) - t*np.exp(-1j*(k_x - Phi*n))
        if n < N-1:
            H[n,n+1] += -t
            H[n+1,n] += -t
    
    #Only hopping down is allowed along the top
    H[q-1, q-2] += -t
    return(H)  

def H_F_k_chiral(N, k_x, q_1, q_2, t_1, t_2, p_1=1, p_2=1, t=1):
    """
        Builds the (approximate) effective Floquet Hamiltonian for flux switching between p_1/q_1 and p_2/q_2 for a square lattice
        with a periodic boundary condition only along the x-direction (a cylindrical square lattice).
        
        Parameters
        ----------
        N : int
            The number of atomic sites along each side of the simulated square lattice. The Hamiltonian 
            matrix built by this function is N by N.
        k_x : float
            The wavenumber of a plane wave in the Fourier series of the Fermionic creation and annihilation 
            operators in the Hamiltonian.
        q_1 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_1.
        q_2 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_2.
        t_1 : float
            The fraction T_1/T, where T is one full period T_1 + T_2 of the switching routine and T_1
            is the period during which the dimensionless flux per unit cell is alpha_1. Should be a 
            value between 0 and 1.
        t_2 : float
            The fraction T_2/T, where T_2 is the period during which the dimensionless flux per unit 
            cell is alpha_2. 
        p_1 : int, optional
            The numerator of the dimensionless measure of magnetic flux per unit cell alpha_1. Defaults to 1.
        p_2 : int, optional
            The numerator of the dimensionless measure of magnetic flux per unit cell alpha_2. Defaults to 1.
        t : float, optional
            The nearest neighbor hopping integral, providing a measure for how likely an electron on the 
            lattice is to hop to nearest neighbor sites. Defaults to 1. Values passed in should be between 
            0 and 1.
            
        Returns
        -------
        ndarray
            A 2D array containing the elements of the N by N Hamiltonian for a square lattice which is 
            infinite along the x-direction and contains N sites along the y-direction.
    """
    H_1 = t_1*H_chiral(N, k_x, q_1, p_1, t)
    H_2 = t_2*H_chiral(N, k_x, q_2, p_2, t)
    H_F = H_1 + H_2
    return(H_F)

def U_n(N, k, mu, n, q, q_1, q_2, T_1, T_2, Hamiltonian, hop_type='nn'):
    '''
        Outputs either the 1st or 2nd link variable for the mu-th band evaluated at some point in k-space.
        
        Parameters
        ----------
        N : int
            The number of atoms per side of the square lattice. Should be an integer multiple of q.
        k : list
            The wavevector [k_x, k_y] at which the link variable U_n should be calculated.
        mu : int
            The band index, with zero corresponding to the bottom band and q-1 corresponding to the top band.
        n : int
            Denotes which link variable should be calculated. Should be either 1 or 2.
        q : int
            If q_1 and q_2 are given, q is the size of the Floquet Hamiltonian for 1/q_1, 1/q_2 flux switching. If q_1 == None, this is 
            the denominator of static flux alpha = 1/q.
        q_1 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_1 for 1/q_1, 1/q_2 flux switching.
        q_2 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_2 for 1/q_1, 1/q_2 flux switching.
        T_1 : float
            The period for which alpha = 1/q_1 for a flux switching routine with total period T = T_1 + T_2. If not using Floquet Hamiltonian,
            set to None.
        T_2 : float
            The period for which alpha = 1/q_2 for a flux switching routine with total period T = T_1 + T_2. Ifnot using Floquet Hamiltonian,
            set to None
        Hamiltonian : function
            The function corresponding to the Hamiltonian matrix for which the Chern numbers of the energy bands should be calculated.
        hop_type : str, optional
            Should be either 'nn' to specify that only nearest neighbor hopping should be included or 
            'nnn' to specify that both nearest and next-nearest neighbor hopping should be included. 
            Defaults to 'nn'.
        
        Returns
        -------
        float
            Returns either the first or second link variable (given by n) to be used for calculating the Berry curvature.
    '''
    #Defining unit vectors in k-space
    e_1 = np.array([2*np.pi/(N), 0])
    e_2 = np.array([0, 2*np.pi/(abs(q)*N)])
    
    if q_1 == None:
        if n==1:
            k_t = np.array(k) + e_1
            E_k, u_k = Hamiltonian(k[0], k[1], q, hop=hop_type)
            E_k_t, u_k_t = Hamiltonian(k_t[0], k_t[1], q, hop=hop_type)
            U1 = np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu]))/np.abs(np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu])))
            return(U1)
        elif n==2:
            k_t = np.array(k) + e_2
            E_k, u_k = Hamiltonian(k[0], k[1], q_1, q_2, q, T_1, T_2, hop=hop_type)
            E_k_t, u_k_t = Hamiltonian(k_t[0], k_t[1], q_1, q_2, q, T_1, T_2, hop=hop_type)
            U2 = np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu]))/np.abs(np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu])))
            return(U2)
    else:
        if n==1:
            k_t = np.array(k) + e_1
            E_k, u_k = Hamiltonian(k[0], k[1], q_1, q_2, q, T_1, T_2, hop=hop_type)
            E_k_t, u_k_t = Hamiltonian(k_t[0], k_t[1], q_1, q_2, q, T_1, T_2, hop=hop_type)
            U1 = np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu]))/np.abs(np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu])))
            return(U1)
        elif n==2:
            k_t = np.array(k) + e_2
            E_k, u_k = Hamiltonian(k[0], k[1], q_1, q_2, q, T_1, T_2, hop=hop_type)
            E_k_t, u_k_t = Hamiltonian(k_t[0], k_t[1], q_1, q_2, q, T_1, T_2, hop=hop_type)
            U2 = np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu]))/np.abs(np.dot(np.conj(np.array(u_k[:, mu])).T, np.array(u_k_t[:, mu])))
            return(U2)
    
def Berry_curv(N, k, mu, q, q_1, q_2, T_1, T_2, Hamiltonian, hop_type='nn'):
    '''
        Outputs the Berry curvature at some point in k space specified by k = [k_x, k_y].
        
        Parameters
        ----------
        N : int
            The number of atoms per side of the square lattice. Should be an integer multiple of q.
        k : list
            The wavevector [k_x, k_y] at which the link variable U_n should be calculated.
        mu : int
            The band index, with zero corresponding to the bottom band and q-1 corresponding to the top band.
        q : int
            If q_1 and q_2 are given, q is the size of the Floquet Hamiltonian for 1/q_1, 1/q_2 flux switching. If q_1 == None, this is 
            the denominator of static flux alpha = 1/q.
        q_1 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_1 for 1/q_1, 1/q_2 flux switching. If not
            using Floquet Hamiltonian, set to None.
        q_2 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_2 for 1/q_1, 1/q_2 flux switching. If not
            using Floquet Hamiltonian, set to None.
        T_1 : float
            The period for which alpha = 1/q_1 for a flux switching routine with total period T = T_1 + T_2. If not using Floquet Hamiltonian,
            set to None.
        T_2 : float
            The period for which alpha = 1/q_2 for a flux switching routine with total period T = T_1 + T_2. Ifnot using Floquet Hamiltonian,
            set to None
        Hamiltonian : function
            The function corresponding to the Hamiltonian matrix for which the Chern numbers of the energy bands should be calculated.
        hop_type : str, optional
            Should be either 'nn' to specify that only nearest neighbor hopping should be included or 
            'nnn' to specify that both nearest and next-nearest neighbor hopping should be included. 
            Defaults to 'nn'.
        
        Returns
        -------
        float
            Returns the berry curvature at the point [k_x, k_y] in k-space.
    '''
    #Defining unit vectors in k-space
    e_1 = np.array([2*np.pi/(N), 0])
    e_2 = np.array([0, 2*np.pi/(abs(q)*N)])
    
    Berry_curv = np.log(U_n(N, [k[0], k[1]], mu, 1, q_1, q_2, q, T_1, T_2, Hamiltonian, hop_type)*U_n(N, np.array([k[0], k[1]]) + e_1, mu, 2, q_1, q_2, q, T_1, T_2, Hamiltonian, hop_type)/(U_n(N, [k[0], k[1]] + e_2, mu, 1, q_1, q_2, q, T_1, T_2, Hamiltonian, hop_type)*U_n(N, [k[0], k[1]], mu, 2, q_1, q_2, q, T_1, T_2, Hamiltonian, hop_type)))
    return(Berry_curv)

def Chern_numbers(N, q, q_1, q_2, T_1, T_2, Hamiltonian, hop = 'nn'):
    '''
        Outputs the Berry curvature at some point in k space specified by k = [k_x, k_y].
        
        Parameters
        ----------
        N : int
            The number of atoms per side of the square lattice. Should be an integer multiple of q.
        q : int
            If q_1 and q_2 are given, q is the size of the Floquet Hamiltonian for 1/q_1, 1/q_2 flux switching. If q_1 == None, this is 
            the denominator of static flux alpha = 1/q.
        q_1 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_1 for 1/q_1, 1/q_2 flux switching. If not
            using Floquet Hamiltonian, set to None.
        q_2 : int
            The denominator of the dimensionless measure of magnetic flux per unit cell alpha_2 for 1/q_1, 1/q_2 flux switching. If not
            using Floquet Hamiltonian, set to None.
        T_1 : float
            The period for which alpha = 1/q_1 for a flux switching routine with total period T = T_1 + T_2. If not using Floquet Hamiltonian,
            set to None.
        T_2 : float
            The period for which alpha = 1/q_2 for a flux switching routine with total period T = T_1 + T_2. Ifnot using Floquet Hamiltonian,
            set to None
        Hamiltonian : function
            The function corresponding to the Hamiltonian matrix for which the Chern numbers of the energy bands should be calculated.
        hop_type : str, optional
            Should be either 'nn' to specify that only nearest neighbor hopping should be included or 
            'nnn' to specify that both nearest and next-nearest neighbor hopping should be included. 
            Defaults to 'nn'.
        
        Returns
        -------
        list
            Returns a list containing the Chern numbers of the q bands in k-space.
    '''
    k_x_reduced = np.arange(-N//2,N//2)*(2*np.pi/(N))
    k_y_reduced = np.arange(-N//2,N//2)*(2*np.pi/(q*N))

    #Defining unit vectors in k-space
    e_1 = np.array([2*np.pi/(N), 0])
    e_2 = np.array([0, 2*np.pi/(q*N)])
    
    Chern_num_list = []
    for mu in range(q):
        c = 0
        for kx_val in k_x_reduced:
            for ky_val in k_y_reduced:
                c += Berry_curv(N, [kx_val, ky_val], mu, q_1, q_2, q, T_1, T_2, Hamiltonian, hop_type=hop)/(2j*np.pi)
        Chern_num_list.append(c)
    return(Chern_num_list)