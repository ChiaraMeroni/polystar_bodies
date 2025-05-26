import numpy as np
import math
from radial_fns import Polytope, add_plot_arbitrary_radial_func_3D
from quadrature_S import sphere_Quadrature
from harmonic_basis import optimal_equivariant_filter_coefficients
from triangulations_ext_3D import radial_func_to_povray
import json
import pdb
import simpy as sp

def even_radon_transform_coefficients_2D(maximum_degree):
    even_radon_coefficients = []
    for k in range(maximum_degree+1):
        if k%2==0:
            q=int(k/2)
            res = (-1)**q
            #res = 1
            for jindex in range(q):
                res = res * ((2*jindex+1)/(2*jindex+2))
            even_radon_coefficients.append(res)
    return even_radon_coefficients

# Compute quadrature in rectangular coordinates
def cartesian_to_spherical_2D(unit_vector):
    res = np.zeros(2) #phitheta
    assert len(unit_vector) == 3, "only available for vectors in 3dims" 
    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2] 
    phi = math.acos(z)
    normxy = math.sqrt(x**2+y**2)
    if np.abs(normxy)<1e-10:
            theta = 0.0 #default value in either north or south pole
    else:
        pix = x/normxy
        if y>0:
            theta = math.acos(pix)
        else:
            theta =2*np.pi-math.acos(pix)

    res[0] = theta
    assert theta >=-1e-5 and theta< 2*np.pi+1e-5, "theta in [0,2pi]"
    res[1] = phi
    return res

def fast_evaluate_AL_poly(l: int, m: int,t):
    assert m<=l and 0<=m, "the coefficient must be in [0,l]"
    results_array = np.zeros(l+1)
    results_array[0] = 1.0
    #we begin with the diagonal, using the diagonal recursion
    for k in range(1,m+1):
        results_array[k] =  (-1)*(2*(k-1)+1)*np.sqrt(1-t**2)*results_array[k-1]
    #next the one after the diagonal
    if l>m:
        #one off from diagonal, special recursive formula
        k=m+1
        results_array[k] = t*(2*(k-1)+1)*results_array[k-1]
        for k in range(m+2,l+1):
            #k=l+1 so l=k-1 in the recursion
            results_array[k] = (2*(k-1)+1)/(k-1-m+1)*t*results_array[k-1]-(k-1+m)/(k-1-m+1)*results_array[k-2]
    return results_array[l]

def compute_normalization_coefficients_dict(total_degree):
    #normalization factors for building spherical harmonics from the associated Legendre polyomials
    normalization_coefficients_dict = dict()
    for current_degree in range(total_degree+1):
        for internal_index in range(current_degree+1):
            l=current_degree
            m = internal_index
            if m==0:
                res = math.sqrt((2*l+1)/(4*math.pi))
            else:
                res = math.sqrt(2)*math.sqrt((2*l+1)/(4*math.pi))           

            for k in range(l-m+1,l+m+1):
                res = res*(1/math.sqrt(k))
            #when done
            normalization_coefficients_dict[(l,m)] = res
    return normalization_coefficients_dict

def fast_evaluate_spherical_harm(normalization_coefficients_dict, l: int, m: int, unit_vector):
    theta_phi = cartesian_to_spherical_2D(unit_vector=unit_vector)
    theta, phi = theta_phi
    mabs = int(abs(m))
    coeff = normalization_coefficients_dict[(l,mabs)]
    AL_poly_value = fast_evaluate_AL_poly(l,mabs,math.cos(phi))
    if m>=0:
        trig_part_value = math.cos(mabs*theta)        
    if m<0:
        trig_part_value = math.sin(mabs*theta)
    return coeff*AL_poly_value*trig_part_value

def compute_spherical_harmonic_values_at_point_array(roots,normalization_coefficients_dict, total_degree):
    spherical_harmonics_values_dict = dict()
    for current_degree in range(total_degree+1):
        for internal_index in range(-current_degree, current_degree+1):
            func_values = np.array([fast_evaluate_spherical_harm(normalization_coefficients_dict, current_degree, internal_index, unit_vector) for unit_vector in roots])
            spherical_harmonics_values_dict[(current_degree,internal_index)] = func_values
        print(f"spherical harmonic values computed in degree {current_degree}!")
    return spherical_harmonics_values_dict

def numerically_normalize_spherical_harmonic_values_dict(weights, spherical_harmonics_values_dict):
    normalized_spherical_harmonics_values_dict = dict()
    for current_degree in range(total_degree+1):
        for internal_index in range(-current_degree, current_degree+1):
            func_values = spherical_harmonics_values_dict[(current_degree,internal_index)]         
            norm_factor = math.sqrt(np.dot(func_values*func_values,weights))    
            print(f"Norm factor in degree {current_degree} index {internal_index} equals {norm_factor}")        
            normalized_spherical_harmonics_values_dict[(current_degree,internal_index)] = 1/norm_factor * spherical_harmonics_values_dict[(current_degree,internal_index)]
        print(f"spherical harmonic values normalized in degree {current_degree}!")
    return normalized_spherical_harmonics_values_dict

def extract_numerical_fourier_coefficient(current_degree,internal_index, radial_fn_values, spherical_harmonics_values_dict, weights):
    assert (current_degree, internal_index) in spherical_harmonics_values_dict
    assert (len(weights)==len(radial_fn_values)), "function must be sampled at quadrature points so lengths must coincide"
    func_values_vector = radial_fn_values * spherical_harmonics_values_dict[(current_degree,internal_index)]
    return np.dot(func_values_vector,weights)

def evaluation_func_from_fourier_coeffs_dict(fourier_coeffs_dict, normalization_coefficients_dict):
    #we want a function which given a unit_vector returns the value of the function given by these fourier_coeffs
    def evaluation_func(unit_vector):
        res = 0.0
        for (current_degree,internal_index) in fourier_coeffs_dict:
            coeff= fourier_coeffs_dict[(current_degree,internal_index)]
            value = fast_evaluate_spherical_harm(normalization_coefficients_dict, current_degree, internal_index,unit_vector) 
            res += coeff*value
        return res
    return evaluation_func



if __name__ == "__main__":
    numvars = 3
    #Up to which degree do we want the quadrature rule to be exact?
    exactness_degree = 60
    roots, weights = sphere_Quadrature(numvars = numvars, exactness_degree = exactness_degree)
    print(f"Number of roots: {len(roots)}")

    #Hyperplanes defining the cube
    hyperplane_coeffs_array = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0],[0,0,1], [0,0,-1]])
    P = Polytope(hyperplane_coeffs_array=hyperplane_coeffs_array, ambient_dimension=3)
    #Main parameter: up to which degree do we compute Fourier coefficients?
    total_degree = 20
    #The next settings are used for povray
    numpartsY = 100
    numpartsX = 2*numpartsY
    povray = False
    povfilename = "smoothed_cube"
    #Change scenario to see other examples
    scenario = "IBUnweighted"

    assert scenario in ["Unweighted", "Weighted","IBUnweighted", "IBWeighted"]
    print(scenario)

    if scenario == "Unweighted":
        #Scenario: Fourier approximation of the polytope P
        radial_fn_values = np.array([P.evaluate_radial_fn(unit_vector) for unit_vector in roots])

        #Next we want to compute the harmonic expansion up to a given target degree
        normalization_coefficients_dict = compute_normalization_coefficients_dict(total_degree)
        spherical_harmonics_values_dict = compute_spherical_harmonic_values_at_point_array(roots, normalization_coefficients_dict, total_degree)
        spherical_harmonics_values_dict = numerically_normalize_spherical_harmonic_values_dict(weights, spherical_harmonics_values_dict)

        #Once the evaluations have been computed we can do the integration to obtain Fourier coefficients via our quadrature
        fourier_coeffs_dict = dict()
        for current_degree in range(total_degree+1):
            for internal_index in range(-current_degree, current_degree+1):
                coeff = extract_numerical_fourier_coefficient(current_degree, internal_index, radial_fn_values, spherical_harmonics_values_dict,weights)
                fourier_coeffs_dict[(current_degree,internal_index)] = coeff
            print(f"Fourier coeffs computation completed in degree {current_degree}")

        #Now we plot the result
        expansion_function_evaluator = evaluation_func_from_fourier_coeffs_dict(fourier_coeffs_dict, normalization_coefficients_dict)
        
        #for povray:
        if povray:
            radial_func_to_povray(
                numpartsX = numpartsX, 
                numpartsY=numpartsY, 
                radial_func = expansion_function_evaluator,
                filename = povfilename
                )
            pdb.set_trace()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])
        add_plot_arbitrary_radial_func_3D(expansion_function_evaluator, ax=ax, numpoints = 2000, color ="blue")
        # plt.savefig("Fourier"+str(total_degree)+".png")
        plt.show()



    if scenario == "Weighted":
        #Scenario: weighted Fourier approximation of the polytope P
        radial_fn_values = np.array([P.evaluate_radial_fn(unit_vector) for unit_vector in roots])

        #Next we want to compute the harmonic expansion up to a given target degree
        normalization_coefficients_dict = compute_normalization_coefficients_dict(total_degree)
        spherical_harmonics_values_dict = compute_spherical_harmonic_values_at_point_array(roots, normalization_coefficients_dict, total_degree)
        spherical_harmonics_values_dict = numerically_normalize_spherical_harmonic_values_dict(weights, spherical_harmonics_values_dict)
        #Once the evaluations have been computed we can do the integration to obtain Fourier coefficients via our quadrature
        fourier_coeffs_dict = dict()
        for current_degree in range(total_degree+1):
            for internal_index in range(-current_degree, current_degree+1):
                coeff = extract_numerical_fourier_coefficient(current_degree, internal_index, radial_fn_values, spherical_harmonics_values_dict,weights)
                fourier_coeffs_dict[(current_degree,internal_index)] = coeff
            print(f"Fourier coeffs computation completed in degree {current_degree}")

        #Next we compute the optimal reweighting
        half_filter_degree = int(np.ceil(total_degree/2)) #Making sure we have enough coefficients
        half_filter_degree = total_degree
        weight_coefficients = optimal_equivariant_filter_coefficients(numvars, half_filter_degree=half_filter_degree, old_method = False)
        print("Computation of weight coefficients for optimal filtering completed")
        print(weight_coefficients)
        #Compute weighted fourier coefficients:
        weighted_fourier_coeffs_dict = dict()
        for (current_degree,internal_index) in fourier_coeffs_dict:
            weighted_fourier_coeffs_dict[(current_degree,internal_index)] = weight_coefficients[current_degree] *fourier_coeffs_dict[(current_degree,internal_index)]

        #To save Fourier coefficients:
        #First we make a serializable object so the json writer below can write it and save it
        #the problem is that the keys of our dictionary are tuples and not strings so we fix that with repr
        total_dict = {
            "unweighted": [[repr(k), v] for k, v in fourier_coeffs_dict.items()],
            "weighted": [[repr(k), v] for k, v in weighted_fourier_coeffs_dict.items()] 
        }
        with open('data30cube.json', 'w') as json_file: 
            json.dump(total_dict, json_file, indent=4) 

        #Finally we plot the result
        expansion_function_evaluator = evaluation_func_from_fourier_coeffs_dict(weighted_fourier_coeffs_dict, normalization_coefficients_dict)

        #for povray:
        if povray:
            radial_func_to_povray(
                numpartsX = numpartsX, 
                numpartsY=numpartsY, 
                radial_func = expansion_function_evaluator,
                filename = povfilename
                )
            pdb.set_trace()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])
        add_plot_arbitrary_radial_func_3D(expansion_function_evaluator, ax=ax, numpoints = 2000, color ="blue")
        # plt.savefig("Fourier_sm"+str(total_degree)+".png")
        plt.show()

    if scenario=="IBUnweighted":
        #Scenario: Intersection body of polytope P from Fourier approximation of its squared radial fn
        #Our target function is the square of the radial function
        radial_fn_values = np.array([P.evaluate_radial_fn_raised_to_dim(unit_vector) for unit_vector in roots])
        #Next we want to compute the harmonic expansion up to a given target degree
        normalization_coefficients_dict = compute_normalization_coefficients_dict(total_degree)
        spherical_harmonics_values_dict = compute_spherical_harmonic_values_at_point_array(roots, normalization_coefficients_dict, total_degree)
        #Once the evaluations have been computed we can do the integration to obtain Fourier coefficients via our quadrature
        fourier_coeffs_dict = dict()
        for current_degree in range(total_degree+1):
            for internal_index in range(-current_degree, current_degree+1):
                coeff = extract_numerical_fourier_coefficient(current_degree, internal_index, radial_fn_values, spherical_harmonics_values_dict,weights)
                fourier_coeffs_dict[(current_degree,internal_index)] = coeff
            print(f"Fourier coeffs computation completed in degree {current_degree}")

        #Next we compute the weights for the radon transform
        weight_coefficients = even_radon_transform_coefficients_2D(total_degree)
        print("Computation of weight coefficients for Radon transform completed")
        #Compute weighted fourier coefficients:
        weighted_fourier_coeffs_dict = dict()
        for (current_degree,internal_index) in fourier_coeffs_dict:
            if current_degree%2!=0:
                weighted_fourier_coeffs_dict[(current_degree,internal_index)] = 0.0
            else:
                half_current_degree = int(current_degree/2)
                weighted_fourier_coeffs_dict[(current_degree,internal_index)] = 2*np.pi*weight_coefficients[half_current_degree]*fourier_coeffs_dict[(current_degree,internal_index)]
        
        #Now we plot the result
        expansion_function_evaluator = evaluation_func_from_fourier_coeffs_dict(weighted_fourier_coeffs_dict, normalization_coefficients_dict)

        #for povray:
        if povray:
            radial_func_to_povray(
                numpartsX = numpartsX, 
                numpartsY=numpartsY, 
                radial_func = expansion_function_evaluator,
                filename = povfilename                
                )
            pdb.set_trace()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])
        add_plot_arbitrary_radial_func_3D(expansion_function_evaluator, ax=ax, numpoints = 2000, color ="blue")
        # plt.savefig("Fourier_IB_from_RF_squared_"+str(total_degree)+".png")
        plt.show()


    if scenario == "IBWeighted":
        #Scenario: Weighted approximation of the squared radial function of the polytope
        radial_fn_values = np.array([P.evaluate_radial_fn_raised_to_dim(unit_vector) for unit_vector in roots])

        #Next we want to compute the harmonic expansion up to a given target degree
        normalization_coefficients_dict = compute_normalization_coefficients_dict(total_degree)
        spherical_harmonics_values_dict = compute_spherical_harmonic_values_at_point_array(roots, normalization_coefficients_dict, total_degree)
        #Once the evaluations have been computed we can do the integration to obtain Fourier coefficients via our quadrature
        fourier_coeffs_dict = dict()
        for current_degree in range(total_degree+1):
            for internal_index in range(-current_degree, current_degree+1):
                coeff = extract_numerical_fourier_coefficient(current_degree, internal_index, radial_fn_values, spherical_harmonics_values_dict,weights)
                fourier_coeffs_dict[(current_degree,internal_index)] = coeff
            print(f"Fourier coeffs computation completed in degree {current_degree}")

        #Next we compute the optimal reweighting
        half_filter_degree = int(np.ceil(total_degree/2)) #Making sure we have enough coefficients
        half_filter_degree = total_degree
        weight_coefficients = optimal_equivariant_filter_coefficients(numvars, half_filter_degree=half_filter_degree, old_method = False)
        print("Computation of weight coefficients for optimal filtering completed")
        print(weight_coefficients)
        #Compute weighted fourier coefficients:
        weighted_fourier_coeffs_dict = dict()
        for (current_degree,internal_index) in fourier_coeffs_dict:
            weighted_fourier_coeffs_dict[(current_degree,internal_index)] = weight_coefficients[current_degree] *fourier_coeffs_dict[(current_degree,internal_index)]

        #Finally we compute the weights for the radon transform applied to the weighted version
        fourier_coeffs_dict = weighted_fourier_coeffs_dict
        weight_coefficients = even_radon_transform_coefficients_2D(total_degree)
        print("Computation of weight coefficients for Radon transform completed")
        #Compute weighted fourier coefficients:
        weighted_fourier_coeffs_dict = dict()
        for (current_degree,internal_index) in fourier_coeffs_dict:
            if current_degree%2!=0:
                weighted_fourier_coeffs_dict[(current_degree,internal_index)] = 0.0
            else:
                half_current_degree = int(current_degree/2)
                weighted_fourier_coeffs_dict[(current_degree,internal_index)] = 2*np.pi*weight_coefficients[half_current_degree]*fourier_coeffs_dict[(current_degree,internal_index)]
        
        #To save Fourier coefficients:
        #First we make a serializable object so the json writer below can write it and save it
        #the problem is that the keys of our dictionary are tuples and not strings so we fix that with repr
        total_dict = {
            "unweighted": [[repr(k), v] for k, v in fourier_coeffs_dict.items()],
            "weighted": [[repr(k), v] for k, v in weighted_fourier_coeffs_dict.items()] 
        }
        with open('dataIB30cube.json', 'w') as json_file: 
            json.dump(total_dict, json_file, indent=4) 

        #Now we plot the result
        expansion_function_evaluator = evaluation_func_from_fourier_coeffs_dict(weighted_fourier_coeffs_dict, normalization_coefficients_dict)    

        if povray:
            radial_func_to_povray(
                numpartsX = numpartsX, 
                numpartsY=numpartsY, 
                radial_func = expansion_function_evaluator,
                filename = povfilename                
                )
            pdb.set_trace()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])
        add_plot_arbitrary_radial_func_3D(expansion_function_evaluator, ax=ax, numpoints = 2000, color ="blue")
        # plt.savefig("Fourier_RF_sm"+str(total_degree)+".png")
        plt.show()






