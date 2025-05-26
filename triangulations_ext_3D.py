import pdb
import numpy as np
import math
from radial_fns import Polytope

class Mesh:
    def __init__(self, triangles_list, namestring) -> None:
        for triangle in triangles_list:
            assert isinstance(triangle,Triangle), "Every element should be of class triangle"
        self.triangles = triangles_list
        self.name = namestring
    def pov_ray_download_string(self):
        result_string = f"# declare {self.name} = mesh"+"{ \n"
        for triangle in self.triangles:
            result_string += triangle.pov_ray_download_string() 
        #finally we add the material
        result_string += "\t texture {\n" 	
        result_string += "\t\t pigment { color rgb <0.9, 0.1, 0.1> }\n"
        result_string += "\t\t finish { phong 1  }\n"
        result_string += "\t\t }\n"
        #and we close
        result_string+= "\t }\n"
        return result_string
    def to_print(self):
        for triangle in self.triangles:
            print(triangle.to_print())

class Triangle:
    def __init__(self,triangle_corners,ambient_dimension) -> None:
        assert ambient_dimension in [2,3], "triangles currently allowed only in two and three dimensions"
        assert len(triangle_corners)==3, "Every triangle must have three vertices"
        for vertex in triangle_corners:
            assert(len(vertex)==ambient_dimension), "Every corner must be a vector in dimension {}"
        self.triangle_corners = triangle_corners
        self.ambient_dimension = ambient_dimension
    
    def pov_ray_download_string(self):
        assert self.ambient_dimension ==3 ,"Option not implemented for 2D"
        result_string ="\t triangle {\n"
        v1 = self.triangle_corners[0]
        v2 = self.triangle_corners[1]
        v3 = self.triangle_corners[2]
        result_string +=f"\t<{v1[0]:.9f},{v1[1]:.9f},{v1[2]:.9f}>, <{v2[0]:.9f},{v2[1]:.9f},{v2[2]:.9f}>, <{v3[0]:.9f},{v3[1]:.9f},{v3[2]:.9f}>\n" 
        result_string +="\t}\n"
        return(result_string)

    def to_print(self):
        Res_string = ""
        for vertex in self.triangle_corners:
            Res_string += str(vertex) + "\n"
        return Res_string

def build_triangulated_rectangle_piece_2D(numPartsX,numPartsY, rangeX, rangeY, indexX, indexY):
    #returns two triangles which cover the rectangle with corners (indexX,indexY) to (indexX+1,indexY+1)
    #when this is possible
    deltaX = (rangeX[1]-rangeX[0])/numPartsX
    deltaY = (rangeY[1]-rangeY[0])/numPartsY
    X0=rangeX[0]
    Y0=rangeY[0]
    assert indexX < numPartsX, "to build triangles you cannot start on the edge of the covered area"
    assert indexY < numPartsY, "to build triangles you cannot start on the edge of the covered area"
    A = np.array([X0+indexX*deltaX, Y0+indexY*deltaY])
    B = np.array([X0+(indexX+1.0)*deltaX, Y0+indexY*deltaY])
    C = np.array([X0+indexX*deltaX, Y0+(indexY+1.0)*deltaY])
    D = np.array([X0+(indexX+1.0)*deltaX, Y0+(indexY+1.0)*deltaY])
    ambient_dimension = 2
    return Triangle([A,B,D],ambient_dimension=ambient_dimension), Triangle([A,C,D],ambient_dimension=ambient_dimension)

def build_triangulated_rectangle_region_2D(numPartsX,numPartsY, rangeX, rangeY):
    triangles_list = []
    for i in range(numPartsX):
        for j in range(numPartsY):
            triangles_list.extend(build_triangulated_rectangle_piece_2D(
                numPartsX = numPartsX,
                numPartsY = numPartsY,
                rangeX = rangeX,
                rangeY = rangeY,
                indexX = i,
                indexY = j
                ))
    return triangles_list

def build_3D_triangles_list_from_spherical_coords_and_radial_func(triangles_list_2D, radial_func):
    #TODO further develop allowing for smooth-triangles
    triangles_list = []
    for triangle_2D in triangles_list_2D:
        new_triangle_corners = []
        for corner_2D in triangle_2D.triangle_corners:        
            theta = corner_2D[0]
            phi = corner_2D[1]
            unit_vector = np.array([math.sin(phi)*math.cos(theta),math.sin(phi)*math.sin(theta),math.cos(phi)])
            scaling = radial_func(unit_vector)
            new_triangle_corners.append(scaling*unit_vector)
        triangles_list.append(Triangle(triangle_corners=new_triangle_corners, ambient_dimension=3))
        #add a new triangle
    return triangles_list

def radial_func_to_povray(numpartsX,numpartsY, radial_func, filename ):
    #Given how many angles in theta and in phi it samples the values and creates a scene
    triangles_list_2D = build_triangulated_rectangle_region_2D(
        numPartsX = numpartsX,
        numPartsY = numpartsY,
        rangeX = [0,2*math.pi],
        rangeY = [0,math.pi]
    )
    print("2D triangles list ready")
    triangles_list_3D = build_3D_triangles_list_from_spherical_coords_and_radial_func(
        triangles_list_2D=triangles_list_2D,
        radial_func = radial_func)
    print("3D triangles list ready")

    mesh_name = "polytope_mesh_attempt"
    M = Mesh(triangles_list_3D, mesh_name)
    scene_string = M.pov_ray_download_string()
    filename = "Renders/"+filename+".pov"
    #finally we add camera and light
    scene_string += """
        camera {
            location <2.5 * cos(clock * 2 * pi), 2, -2.5 * sin(clock * 2 * pi)>   // Camera moves in a circle
            look_at <0, 0, 0>   // Camera always looks at the center
        }

        #version 3.7;
        global_settings {
            assumed_gamma 2.2
            max_trace_level 5
            radiosity {
                count 100
                nearest_count 5
                error_bound 1.8
                recursion_limit 2
                low_error_factor 0.5
                gray_threshold 0.0
                pretrace_start 0.08
                pretrace_end 0.01
                brightness 1
                adc_bailout 0.01/2
            }
        }

        light_source {
            <-2, 4, -3>
            color rgb <1, 1, 1>
            fade_distance 10
            fade_power 2
        }

        #declare Ambient_Light = rgb <0.1, 0.1, 0.1>;

        background {
            color rgb <0.5, 0.7, 1.0>
        }

        light_source {
            <5, 5, -5>
            color rgb <1, 1, 1>
        }

        object {
            polytope_mesh_attempt
        }
        """
    #and write everything to a file
    try:
        with open(filename, "x") as f:
            f.write(scene_string)
        
    except FileExistsError:
            print("ERROR: File already exists.")


if __name__ == "__main__":
    import numpy as np

    #Next we construct a polytope via its radial function
    hyperplane_coeffs_array = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0],[0,0,1], [0,0,-1]])
    P=Polytope(hyperplane_coeffs_array=hyperplane_coeffs_array, ambient_dimension=3)
    #We make a triangulated theta, phi region and push it onto the polytope for drawing it
    triangles_list_2D = build_triangulated_rectangle_region_2D(
        numPartsX = 500,
        numPartsY = 250,
        rangeX = [0,2*math.pi],
        rangeY = [0,math.pi]
    )
    triangles_list_3D = build_3D_triangles_list_from_spherical_coords_and_radial_func(
        triangles_list_2D=triangles_list_2D,
        radial_func = P.evaluate_radial_fn)
    mesh_name = "polytope_mesh_attempt"
    M = Mesh(triangles_list_3D, mesh_name)
    #Create povray scene and download to a file in /Renders
    radial_func_to_povray(
        numpartsX=200, 
        numpartsY=100, 
        radial_func=P.evaluate_radial_fn_raised_to_dim,
        filename="intento"
        )

