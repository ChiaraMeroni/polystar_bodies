import numpy as np
from matplotlib import cm

class Polytope:
    def __init__(self, hyperplane_coeffs_array: np.array, ambient_dimension: int):
        sp = hyperplane_coeffs_array.shape    
        assert sp[1]==ambient_dimension, f"Expected number of columns {ambient_dimension}, but got {sp[1]}"
        self.hyperplane_coeffs_array = hyperplane_coeffs_array
        self.ambient_dimension = ambient_dimension
    def evaluate_radial_fn(self, point):
        #Given a point in the ambient space R^a evaluate the radial function
        res = np.inf
        for coefficient_vector in self.hyperplane_coeffs_array:
            alpha = np.dot(coefficient_vector, point)
            if alpha > 0 and (1/alpha)<res:
                res = 1/alpha
        return res
    def evaluate_radial_fn_raised_to_dim(self,point):
        res = self.evaluate_radial_fn(point)
        return res**(self.ambient_dimension-1)

def add_plot_low_dim_radial_fn(polytope : Polytope, ax, numpoints=400, color = "blue"):
    #if dimension is one or two this returns a plot of the radial function
    assert polytope.ambient_dimension in [2,3], f"Radial function implemented only in dimensions 2,3"
    x = np.linspace(0, 2*np.pi, numpoints)  # 400 points from -10 to 10
    #We sample the circle uniformly
    def g(x):
        return polytope.evaluate_radial_fn([np.cos(x),np.sin(x)])
    
    y = [g(p) for p in x]
    ax.plot(x, y, label='radial_function', color = color)

def add_plot_arbitrary_radial_func_2D(radial_func_2D, ax, numpoints=400, color = "blue"):
    #if dimension is one or two this returns a plot of the radial function
    x = np.linspace(0, 2*np.pi, numpoints)  # 400 points from -10 to 10
    #We sample the circle uniformly
    def g(x):
        return radial_func_2D([np.cos(x),np.sin(x)])

    xs = [g(p)*np.cos(p) for p in x]
    ys = [g(p)*np.sin(p) for p in x]
    ax.plot(xs, ys, label='polygon', color = color)

def add_plot_arbitrary_radial_func_3D(radial_func_3D, ax, numpoints=400, color = "blue"):
    assert color in ["red", "blue", "green"], "Only red,blue,green colormaps are implemented"
    #We triangulate a sphere with spherical coordinates theta phi
    num_phi_points = int(np.ceil(np.sqrt(numpoints/2)))
    num_theta_points = 2*num_phi_points
    Theta = np.linspace(0.0, 2*np.pi+2*np.pi/num_theta_points, num_theta_points)
    Phi = np.linspace(0.0, np.pi+np.pi/num_phi_points, num_phi_points)
    Theta, Phi = np.meshgrid(Theta, Phi)
    X = np.zeros(Theta.shape)
    Y = np.zeros(Theta.shape)
    Z = np.zeros(Theta.shape)
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            X[i,j] = np.sin(Phi[i,j])*np.cos(Theta[i,j])
            Y[i,j] = np.sin(Phi[i,j])*np.sin(Theta[i,j])
            Z[i,j] = np.cos(Phi[i,j])
            R = radial_func_3D([X[i,j],Y[i,j],Z[i,j]])
            X[i,j] = R*X[i,j]
            Y[i,j] = R*Y[i,j]
            Z[i,j] = R*Z[i,j]
    #Now we add the plot
    if color == "blue":
        ax.plot_surface(X, Y, Z, cmap=cm.Blues)
    if color == "green":
        ax.plot_surface(X, Y, Z, cmap=cm.Greens)
    if color == "red":
        ax.plot_surface(X, Y, Z, cmap=cm.Oranges)

def draw_polytope_using_radial_func(polytope : Polytope, ax, numpoints=400, color = "blue"):
    assert polytope.ambient_dimension in [2,3], f"Radial function plotting implemented only in dimension 2"
    if polytope.ambient_dimension == 2:
        x = np.linspace(0, 2*np.pi, numpoints)  # 400 points from -10 to 10
        #We sample the circle uniformly
        def g(x):
            return polytope.evaluate_radial_fn([np.cos(x),np.sin(x)])
        
        xs = [g(p)*np.cos(p) for p in x]
        ys = [g(p)*np.sin(p) for p in x]
        ax.plot(xs, ys, label='polygon', color = color)

    if polytope.ambient_dimension == 3:
        assert color in ["red", "blue", "green"], "Only red,blue,green colormaps are implemented"
        #We triangulate a sphere with spherical coordinates theta phi
        num_phi_points = int(np.ceil(np.sqrt(numpoints/2)))
        num_theta_points = 2*num_phi_points
        Theta = np.linspace(0.0, 2*np.pi+2*np.pi/num_theta_points, num_theta_points)
        Phi = np.linspace(0.0, np.pi+np.pi/num_phi_points, num_phi_points)
        Theta, Phi = np.meshgrid(Theta, Phi)
        X = np.zeros(Theta.shape)
        Y = np.zeros(Theta.shape)
        Z = np.zeros(Theta.shape)
        for i in range(Theta.shape[0]):
            for j in range(Theta.shape[1]):
                X[i,j] = np.sin(Phi[i,j])*np.cos(Theta[i,j])
                Y[i,j] = np.sin(Phi[i,j])*np.sin(Theta[i,j])
                Z[i,j] = np.cos(Phi[i,j])
                R = polytope.evaluate_radial_fn([X[i,j],Y[i,j],Z[i,j]])
                X[i,j] = R*X[i,j]
                Y[i,j] = R*Y[i,j]
                Z[i,j] = R*Z[i,j]
        #Now we make the plot
        if color == "blue":
            ax.plot_surface(X, Y, Z, cmap=cm.Blues)
        if color == "green":
            ax.plot_surface(X, Y, Z, cmap=cm.Greens)
        if color == "red":
            ax.plot_surface(X, Y, Z, cmap=cm.Oranges)