import torch
import numpy as np



"THis is not SE2 group. Its p4. It doesn't work for grid size =/= 4"

#Translation group
class Rn:
    name = 'R^2'
    #Dimension:
    n = 2
    #Identity element:
    e = torch.tensor([0., 0.], dtype=torch.float32)



## The sub-group H:
class H:
    name = 'SO(2)'
    #Dimension:
    n = 1
    #Identity element:
    e = torch.tensor([0.], dtype=torch.float32)

    ## TODO: So far just for multiples of 90 degrees. No interpolation required, Remove the Grid
    ##Works for general theta not just 90
    def left_representation_on_Rn(h, xx):
        if h == H.e:
            return xx
        else:
            xx_new = torch.rot90(xx, k=int(torch.round((1./(np.pi/2.)*h)).item()), dims=[-2, -1])
        return xx_new

    def left_representation_on_G(h, fx):
        h_inv_weight = H.left_representation_on_Rn(h, fx)

        h_inv_weight = torch.roll(h_inv_weight, shifts=int(torch.round((1. / (np.pi / 2.) * h)).item()), dims=2)
        return h_inv_weight

    ## Essential in the group convolutions
    # Define the determinant (of the matrix representation) of the group element
    def absdet(h):
        return 1.

    ## Grid class
    class grid_global:  # For a global grid
        # Should a least contain:
        #	N     - specifies the number of grid points
        #	scale - specifies the (approximate) distance between points, this will be used to scale the B-splines
        # 	grid  - the actual grid
        #	args  - such that we always know how the grid was constructed
        # Construct the grid
        def __init__(self, N):
            # Store N
            self.N = N
            # Define the scale (the spacing between points)
            self.scale = [2 * np.pi / N]
            # Generate the grid
            if self.N == 0:
                h_list = torch.tensor([], dtype=torch.float32)
            else:
                h_list = torch.from_numpy(np.array([np.linspace(0, 2*np.pi - 2*np.pi/N,N)], dtype=np.float32).transpose())
            self.grid = h_list
            # -------------------


## Group G = R^n \rtimes H.
class G:
    # Label for the group G
    name = 'SE(2)'
    # Dimension of the group G
    n = Rn.n + H.n
    # The identity element
    e = torch.cat([Rn.e, H.e], dim=-1)

    # Function that returns the classes for R^n and H
    @staticmethod
    def Rn_H():
        return Rn, H
