import numpy as np
import sympy as sp
import math as math
from quadrature_S import Chebyshev_poly_1st_kind, Gegenbauer_poly, sphere_surface_area, integrate_bbox_func_with_quadrature, sphere_Quadrature, weighted_Gaussian_roots_in_interval, weighted_Gaussian_Qrule_GW, fast_evaluate_Gegenbauer_poly
import pdb

def comb(n,k):
    return int(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))

def norm(p: sp.Poly):
    return math.sqrt(np.dot(p.coeffs(),p.coeffs()))

def laplacian(variables, p: sp.Poly):
    return sum([sp.diff(sp.diff(p,var),var) for var in variables])

def harmonics_dimension(numvars:int, degree: int):
    n = numvars
    m = degree
    if degree==0:
        return 1
    elif degree == 1:
        return n
    elif degree>=2:
        return comb(n+m-1,n-1)-comb(n+m-3,n-1)

def zonal_poly_centered_at_y(degree : int, variables, vect_y):
    #Returns a polynomial in the variables x with the following properties:
    #It should be homogeneous of degree degree,
    #it should be harmonic 
    #It should have value dim(H_degree) when x=y
    #It should be invariant under the stabilizer of the point $y$
    numvars = len(variables)    
    assert numvars>=3, "There must be at least three variables to define zonal polynomials via this implementation"
    alpha=(numvars-2)/2
    t = sp.symbols("t")
    #normalization constants
    muS = sphere_surface_area(numvars)
    hdim = harmonics_dimension(numvars, degree)
    #The zonal polynomials are just suitably normalized Gegenbauers
    p = sp.Poly(Gegenbauer_poly(alpha,degree,t))
    zonal_in_t= (hdim/(muS*p.eval(1)))*p
    normsq_poly = np.dot(variables,variables)
    #Homogenization of the evaluation at vector y on the sphere
    place = np.dot(variables,vect_y)
    newmons = [(place**monom[0])*(normsq_poly**int((degree-monom[0])/2)) for monom in zonal_in_t.monoms()]
    zonal_value = sp.Poly(np.dot(zonal_in_t.coeffs(),newmons))
    return zonal_value

def zonal_func_centered_at_y(degree : int, variables, vect_y):
    assert np.abs(np.dot(vect_y,vect_y)-1)<1e-6, "Should be evaluated at points on the sphere only" 
    #Returns a non-homogeneous polynomial function in x 
    #which agrees with the zonal polynomial centered at y FOR x in THE UNIT SPHERE
    numvars = len(variables)    
    assert numvars>=2, "There must be at least two variables to define zonal polynomials"
    t = sp.symbols("t")

    if numvars == 2:
        #Zonal polynomials do not comefrom Gegenbauers in dimension two
        p = Chebyshev_poly_1st_kind(degree,t)
        if degree==0:
            #zonal_in_t = sp.Poly(p/(2*np.pi),t)
            zonal_in_t = p/(2*np.pi)
        else:
            #zonal_in_t = (2/(2*np.pi*p.eval(1)))*p
            zonal_in_t = (2/(2*np.pi*p.subs(t,1)))*p.expand()
    else:
        alpha=(numvars-2)/2
        #normalization constants
        muS = sphere_surface_area(numvars)
        hdim = harmonics_dimension(numvars, degree)
        #The zonal polynomials are just suitably normalized Gegenbauers
        p =Gegenbauer_poly(alpha,degree,t)
        zonal_in_t= (hdim/(muS*p.subs(t,1)))*p.expand()
        #zonal_in_t= (hdim/(muS*p.eval(1)))*p

    #pdb.set_trace()
    ##WARNING: Must be evaluated at x on the sphere...
    inner_prod = np.dot(variables, vect_y)
    zexpr = zonal_in_t.subs(t,inner_prod)    
    return zexpr

def single_harmonic_projection(target_func, degree_m, variables, quadrature_exactness_degree):
    #Given a black-box target_func it constructs the best L2 approximation
    #By integration with a quadrature rule of the given degree    
    #First we construct the desired quadrature rule:
    roots, weights = sphere_Quadrature(numvars, quadrature_exactness_degree)
    first = True
    for index, (root, weight) in enumerate(zip(roots,weights)):
        zonal_func = zonal_func_centered_at_y(degree_m, variables, vect_y = root)
        if first: 
            result = zonal_func*weight*target_func(root)
            first = False
        else:
            result += zonal_func*weight*target_func(root)
    return result.expand()

def L2norm_by_quadrature(target_func, quadrature_exactness_degree):
    roots, weights = sphere_Quadrature(numvars, quadrature_exactness_degree)
    s=sum([(target_func(root)**2)*weight for index, (root,weight) in enumerate(zip(roots,weights))])
    return math.sqrt(s)

def single_harmonic_projection_given_Quadrature(target_func, degree_m, variables, roots, weights):
    #Given a black-box target_func it constructs the best L2 approximation
    #By integration with a quadrature rule of the given degree    
    #First we construct the desired quadrature rule:
    #roots, weights = sphere_Quadrature(numvars, quadrature_exactness_degree)
    first = True
    for index, (root, weight) in enumerate(zip(roots,weights)):
        zonal_func = zonal_func_centered_at_y(degree_m, variables, vect_y = root)
        if first: 
            result = zonal_func*weight*target_func(root)
            first = False
        else:
            result += zonal_func*weight*target_func(root)
    return result

#Functions for computing the coefficients of equivariant filters (which determine 
#their action completely). KEY FUNCTION FOLLOWS:
def equivariant_filter_coefficient(univariate_filter_func,filter_degree, coefficient_degree : int, numvars):
    #Given a filter function h(t) implemented as a (possibly unnormalized) black box computes the coefficient
    #of the convolution \int h(<x,y>)f(y)dy=\gamma(f)(x)
    #in degree coefficient_degree
    variablesX = sp.symbols([f"x{val+1}" for val in range(numvars)])
    center_y = np.zeros(numvars)
    center_y[0] = 1.0
    z_deg_func = zonal_func_centered_at_y(degree=coefficient_degree, variables=variablesX, vect_y=center_y)

    def integrand_func(z):
        lista_subs = list(zip(variablesX, z))
        return z_deg_func.subs(lista_subs)*univariate_filter_func(np.dot(center_y,z))

    #exactness_degree = int(np.ceil((filter_degree + coefficient_degree)/2))
    exactness_degree = filter_degree + coefficient_degree
    roots, weights = sphere_Quadrature(numvars, exactness_degree=exactness_degree)
    unnormalized_coeff = integrate_bbox_func_with_quadrature(integrand_func,roots,weights)
    return unnormalized_coeff/harmonics_dimension(numvars=numvars, degree = coefficient_degree)

def equivariant_filter_coefficient_via_univariate_quadrature(univariate_filter_func,filter_degree, coefficient_degree : int, numvars, new_method = True):
    assert numvars>=3, "Gegenbauer polynomials give the orthogonal function only in dimension at least 3"
    #TODO: replace recursive gegenbauers for a fast evaluation algorithm
    alpha = (numvars-2)/2
    roots, weights = weighted_Gaussian_Qrule_GW(alpha, filter_degree+coefficient_degree)
    j = coefficient_degree
    t = sp.symbols("t")
    if new_method==False:
        gj = Gegenbauer_poly(alpha, j,t)

        def gegenbauer_func_sq(z):
            return gj.subs(t,z)**2

        def integrand_func(z):
            return gj.subs(t,z) * univariate_filter_func(z)
    else:
        def gegenbauer_func_sq(z):
            return fast_evaluate_Gegenbauer_poly(alpha,j,z)**2

        def integrand_func(z):
            return fast_evaluate_Gegenbauer_poly(alpha,j,z) * univariate_filter_func(z)

    normalization_factor = integrate_bbox_func_with_quadrature(gegenbauer_func_sq,roots,weights)
    projection_coeff = integrate_bbox_func_with_quadrature(integrand_func,roots,weights)/normalization_factor
    #Ultimately we wish to return the lambda_j for the Funk Hecke formula    
    return (projection_coeff * fast_evaluate_Gegenbauer_poly(alpha,j,1)*sphere_surface_area(numvars))/harmonics_dimension(numvars,j)

#Construction of optimal univariate filter polynomials (up to normalization):
def build_optimal_univariate_filter_func(numvars, half_filter_degree):
    #returns a black box implementation of the polynomial of degree filter degree
    #that has all roots except the biggest one in common with theorthogonal poly of degree one more 
    roots = weighted_Gaussian_roots_in_interval(numvars,half_filter_degree+1)
    roots_except_biggest = np.delete(roots, np.argmax(roots))
    assert len(roots)==half_filter_degree+1
    def univariate_filter_func(z):
        val = 1.0
        for root in roots_except_biggest:
            val = (z-root)**2 * val    
        return val
    return univariate_filter_func

def equivariant_filter_coefficients(numvars, half_filter_degree, univariate_filter_func, old_method = True):
    Coefficients_Vector = []
    for j in range(2*half_filter_degree+1):
        if old_method:
            res = equivariant_filter_coefficient(
                univariate_filter_func,
                filter_degree=2*half_filter_degree, 
                coefficient_degree = j, 
                numvars = numvars)
        else:
            res = equivariant_filter_coefficient_via_univariate_quadrature(
                univariate_filter_func,
                filter_degree=2*half_filter_degree, 
                coefficient_degree = j, 
                numvars = numvars)

        Coefficients_Vector.append(res)

    Coefficients_Vector = np.array(Coefficients_Vector)
    Coefficients_Vector = Coefficients_Vector/Coefficients_Vector[0] #Should be normalized so it is a probability dist
    return Coefficients_Vector

def optimal_equivariant_filter_coefficients(numvars, half_filter_degree, old_method=True):
    univariate_filter_func = build_optimal_univariate_filter_func(numvars, half_filter_degree)
    Coefficients_Vector = equivariant_filter_coefficients(
            numvars = numvars,
            half_filter_degree=half_filter_degree, 
            univariate_filter_func = univariate_filter_func,
            old_method=old_method)
    return Coefficients_Vector