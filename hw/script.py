#%%

import numpy as np
from scipy import integrate


def calculate_angle(z: float,  Omega_m: float, Omega_Lambda: float, Omega_r: float = 0, Omega_k: float = 0, l: float = 5, verbose: bool = False) -> float:
    """
    Calculate the angular size of a l kpc galaxy at redshift z
    
    Parameters:
    z : Redshift
    l : Physical size of the galaxy in kpc (default 5 kpc)
    Omega : Density fraction parameter

    Return:
    Angular size
    """

    l_cm = l * 3.086e21 # cm
    
    # Physical constants
    h = 0.674 # Planck 2018
    G = 6.674e-8  # cm^3 g^-1 s^-2
    C = 3e10  # cm/s
    rho_crit0 = 1.878e-29 * h**2  # g cm^-3


    A = l_cm * np.sqrt(8 * np.pi * G * rho_crit0 / 3) / C
    if verbose:
        print(f"A = {A:.3e} ")
    
    def integrand(z_prime):
        factor = (Omega_r * (1 + z_prime)**4 + 
                    Omega_m * (1 + z_prime)**3 + 
                    Omega_k * (1 + z_prime)**2 + 
                    Omega_Lambda)
        return 1 / np.sqrt(factor)
    integral_result, _ = integrate.quad(integrand, 0, z)
    
    theta_rad = A * (1 + z) / integral_result
    
    return theta_rad



if __name__ == "__main__":
    print("Question 4:")
    arcsec = 180 * 3600 / np.pi

    print("1. Matter Domained (Omega_m, Omega_L) = (1,0):")
    theta_01_matter = calculate_angle(0.1, 1.0, 0.0, verbose=True)
    theta_1_matter = calculate_angle(1.0, 1.0, 0.0)
    print(f"   z = 0.1: theta = {(theta_01_matter*arcsec):.6f} arcsec")
    print(f"   z = 1.0: theta = {(theta_1_matter*arcsec):.6f} arcsec")

    print("\n2. LCDM (Omega_m, Omega_L) = (0.3, 0.7):")
    theta_01_lambda = calculate_angle(0.1, 0.3, 0.7)
    theta_1_lambda = calculate_angle(1.0, 0.3, 0.7)
    print(f"   z = 0.1: theta = {(theta_01_lambda*arcsec):.6f} arcsec")
    print(f"   z = 1.0: theta = {(theta_1_lambda*arcsec):.6f} arcsec")



    # plot
    import matplotlib.pyplot as plt
    plt.style.use('default')
    z_array = np.logspace(-2, 1, 100)
    theta_matter = []
    theta_lambda = []

    for z in z_array:
        theta_m = calculate_angle(z, 1.0, 0.0)
        theta_l = calculate_angle(z, 0.3, 0.7)
        theta_matter.append(theta_m)
        theta_lambda.append(theta_l)

    plt.figure(figsize=(10, 6))
    plt.loglog(z_array, theta_matter, 'b-', label='Matter-dominated (Ωₘ=1, ΩΛ=0)', linewidth=2)
    plt.loglog(z_array, theta_lambda, 'r-', label='ΛCDM (Ωₘ=0.3, ΩΛ=0.7)', linewidth=2)
    plt.plot([0.1, 1.0], [theta_01_matter, theta_1_matter], 'bo', markersize=8)
    plt.plot([0.1, 1.0], [theta_01_lambda, theta_1_lambda], 'ro', markersize=8)

    plt.xlabel('Redshift z', fontsize=12)
    plt.ylabel('Angular Size (arcseconds)', fontsize=12)
    plt.title('Angular Size of 5 kpc Galaxy vs Redshift', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0.01, 10)

    plt.tight_layout()
    plt.show()
