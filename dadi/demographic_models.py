"""
Custom demographic model for our example.
"""

import numpy
import dadi



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Defines the different models of divergence used in our inferences with DADI."
"There are six categories of models: SI Strict Isolation, IM Isolation with Migration, AM Ancient Migration, PAM Periodic Ancient Migration, SC Secondary Contact and PSC: Periodic Secondary Contact."
"Each model comes in subcategories: no suffix: Reference model; ex: Exponential growth; 2N: Background selection with 2 categories of population size in the genome (we assume population size is reduced by a given factor in non-recombining regions); 2M2P: Selection against migrants with 2 categories of migration rate in the genome (we assume migration is null in barrier regions); and the combinations between subcategories: 2N2M2P; 2Nex; 2M2Pex; 2N2M2Pex."
"All models assume an initial ancestral population which is split in two daughter populations."
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""
"Strict Isolation models (SI)"
""""""""""""""""""""""""""""""

def SI(params,(n1,n2), pts):
    nu1, nu2, Ts = params
    """
    Model with split and strict isolation.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Ts: Time of divergence in strict isolation.
    O: Proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # Ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Split event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    # Correctly oriented spectrum
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs
    
    
def SIex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, Ts, Te = params
    """
    Model with split and strict isolation; exponential growth.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after exponential growth.
    nu2: Size of population 2 after exponential growth.
    Ts: Time of divergence in strict isolation.
    Te: Time of the exponential growth in strict isolation.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # Exponential growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=0, m21=0) 
    # Correctly oriented spectrum
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs
    
    
    
def SIbo(params, (n1,n2), pts):
    nu1a, nu2a, nu1B, nu2B, nu1, nu2, Ts, TB, Te = params
    """
    Model with split and strict isolation; exponential growth.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1B: Size of population 1 during bottleneck.
    nu2B: Size of population 2 during bottleneck.
    nu1: Size of population 1 after bottleneck.
    nu2: Size of population 2 after bottleneck.
    Ts: Time of divergence in strict isolation.
    TB: Time of the bottleneck in strict isolation.
    Te: Time of the exponential growth in strict isolation.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # Bottleneck
    phi = dadi.Integration.two_pops(phi, xx, TB, nu1B, nu2B, m12=0, m21=0) 
    # Exponential growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=0, m21=0) 
    # Correctly oriented spectrum
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs


    
    
    
    
""""""""""""""""""""""""""""""""""""""
"Isolation with Migration models (IM)"
""""""""""""""""""""""""""""""""""""""

def IM(params,(n1,n2), pts):
    nu1, nu2, m12, m21, Ts = params
    """
    Model with continuous migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1. pop1=JPA, pop2=JTR
    m21: Migration from population 1 to population 2.
    Ts: Time of divergence in continuous migration.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # Divergence in continuous migration
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs
    
    
def IMex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te = params
    """ 
    Model with continuous migration; exponential growth. 
	
    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after exponential growth.
    nu2: Size of population 2 after exponential growth.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Ts: Time of divergence in strict isolation.
    Te: Time of the exponential growth in strict isolation.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """ 
    xx = dadi.Numerics.default_grid(pts) 	
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=m12, m21=m21) 
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21) 
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs

    
   
    


""""""""""""""""""""""""
"Secondary Contact (SC)"
""""""""""""""""""""""""

def SC(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc = params
    """
    Model with split, strict isolation, and secondary contact.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Ts: Time of divergence in strict isolation.
    Tsc: Time of secondary contact.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    #Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    #Secondary contact event
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs


def SCex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te = params
    """
    Model with split, strict isolation, and secondary contact; exponential growth.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Ts: Time of divergence in strict isolation.
    Tsc: Time of secondary contact.
    Te: Time of the exponential growth in continuous migration.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21) 
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs

""""""""""""""""""""""""
"Ancient Migration (AM)"
""""""""""""""""""""""""

def AM(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Tam, Ts = params
    """
    Model with split, ancient migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Tam: Time of ancient migration.
    Ts: Time of divergence in strict isolation.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # Ancient migration event
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1, nu2, m12=m12, m21=m21)
    # Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs
    


def AMex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Tam, Ts, Te = params
    """
    Model with split, ancient migration; exponential growth.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Tam: Time of ancient migration.
    Ts: Time of divergence in strict isolation.
    Te: Time of the exponential growth in strict isolation.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1a, nu2a, m12=m12, m21=m21)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=0, m21=0) 
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs


    
    
    
""""""""""""""""""""""""
"Periodic Secondary Contact (PSC)"
""""""""""""""""""""""""

def PSC(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc = params
    """
    Model with split, strict isolation, and two periods of secondary contact.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Ts: Time of divergence in strict isolation.
    Tsc: Time of secondary contact.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    #Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    #Secondary contact event-1
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    #Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    #Secondary contact event-2
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
 
    return fs

def PSCex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te = params
    """
    Model with split, strict isolation, and two periods of secondary contact; exponential growth.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Ts: Time of divergence in strict isolation.
    Tsc: Time of secondary contact.
    Te: Time of the exponential growth in continuous migration.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21) 
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs




""""""""""""""""""""""""
"Periodic Ancient Migration PAM"
""""""""""""""""""""""""

def PAM(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Tam, Ts = params
    """
    Model with split, two periods of ancient migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Tam: Time of ancient migration.
    Ts: Time of divergence in strict isolation.
    n1,n2: Size of fs to generate.
    O: The proportion of accurate SNP orientation.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # Ancient migration event-1
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1, nu2, m12=m12, m21=m21)
    # Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    # Ancient migration event-2
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1, nu2, m12=m12, m21=m21)
    # Divergence in strict isolation
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs




def PAMex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Tam, Ts, Te = params
    """
    Model with split, two periods of ancient migration; exponential growth.

    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1.
    m21: Migration from population 1 to population 2.
    Tam: Time of ancient migration.
    Ts: Time of divergence in strict isolation.
    Te: Time of the exponential growth in strict isolation.
    O: The proportion of accurate SNP orientation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1a, nu2a, m12=m12, m21=m21)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1a, nu2a, m12=m12, m21=m21)
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=0, m21=0) 
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
 
    return fs
