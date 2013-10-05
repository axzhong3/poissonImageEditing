## Gradient domain blending
## Cloned and updated from https://github.com/z-o-e

#!/usr/bin/env python
import matplotlib.pylab as plt
import Image
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg


F = 'foreground.jpg'
B = 'background.jpg'
M = 'matte.png'


class GradientDomainCloning:
    def __init__(self, F, B, M):
        # foreground
        self.f = np.asarray(Image.open(F),dtype=int) 

        # background
        self.b = np.asarray(Image.open(B),dtype=int)       
        # mask
        self.m = np.asarray(Image.open(M),dtype=int)

        # width and height
        self.h = self.b.shape[0]
        self.w = self.b.shape[1]
        # new image after gradient domain cloning
        self.new = Image.new('RGB',(self.h,self.w))        
        # map coordinate of pixels to be calculated to index_map according to mask
        self.idx_map = []

        # map coordinates of neigbourhoods to mask indices
        ngb_map = []

        # map coordinates to mask indices
        self.pMap = [[-1 for i in range (self.w)] for j in range(self.h)]
        counter = 0;
        
        for i in range(self.h):
            for j in range(self.w):
                if self.m[:,:,0][i][j]==255:
                    self.idx_map.append([i,j])
                    ngb_map.append([self.m[:,:,0][i-1][j]==255, 
                                    self.m[:,:,0][i+1][j]==255, 
                                    self.m[:,:,0][i][j-1]==255, 
                                    self.m[:,:,0][i][j+1]==255])
                    self.pMap[i][j] = counter
                    counter = counter+1
        
        # nxn matrix A, nx1 vector b are used to solve poisson equation Au=b
        # for nx1 unknown pixel color vector u
        # r, g, b, 3 channels are calculated seperately
        n = len(self.idx_map)
        self.b_r = np.zeros(n)
        self.b_g = np.zeros(n)
        self.b_b = np.zeros(n)
        self.A = [[0 for i in range (n)] for j in range(n)]

                
        # set up sparse matrix A, 4's on main diagnal, -1's and 0's off main diagnal
        for i in range(n):
            self.A[i][i] = 4;
            xx = self.idx_map[i][0]
            yy = self.idx_map[i][1]
            if (ngb_map[i][0] == True):
                self.A[i][self.pMap[xx-1][yy]] = -1
            if (ngb_map[i][1] == True):
                self.A[i][self.pMap[xx+1][yy]] = -1
            if (ngb_map[i][2] == True):
                self.A[i][self.pMap[xx][yy-1]] = -1
            if (ngb_map[i][3] == True):
                self.A[i][self.pMap[xx][yy+1]] = -1
        self.A = sparse.lil_matrix(self.A, dtype=int)
    
                
    # count within-clone-region-neighbor of a pixel in the clone region                 
    def count_neighbor(self, pix_idx):       
        count = 0
        boundary_flag = [0,0,0,0]
        y = pix_idx[0]
        x = pix_idx[1]
        # has left neighbor or not
        if (y >= 0 and y < self.h):
            if (y==0 or self.pMap[y-1][x] == -1):
                boundary_flag[0] = 1
            else:
                count +=1
            if (y == self.h-1 or self.pMap[y+1][x] == -1):
                boundary_flag[1] = 1
            else:
                count +=1
        if ( x >= 0 and x < self.w):
            if (x==0 or self.pMap[y][x-1] == -1):
                boundary_flag[2] = 1
            else:
                count +=1
            if (x == self.w-1 or self.pMap[y][x+1] == -1):
                boundary_flag[3] = 1
            else:
                count +=1
        return count,boundary_flag
    
    # set up b and solve discrete poisson equation    
    def poisson_solver(self):
        # split into r, g, b 3 channels and
        # iterate through all pixels in the cloning region indexed in idx_map
        for i in range(len(self.idx_map)):
            neighbors, flag = self.count_neighbor(self.idx_map[i])
            x, y = self.idx_map[i]
            if neighbors == 4:
                # degraded form if neighbors are all within clone region
                self.b_r[i] = 4*self.f[x,y,0] - (self.f[x-1,y,0] +self.f[x+1,y,0] + self.f[x,y-1,0] + self.f[x,y+1,0])
                self.b_g[i] = 4*self.f[x,y,1] - (self.f[x-1,y,1] +self.f[x+1,y,1] + self.f[x,y-1,1] + self.f[x,y+1,1])
                self.b_b[i] = 4*self.f[x,y,2] - (self.f[x-1,y,2] +self.f[x+1,y,2] + self.f[x,y-1,2] + self.f[x,y+1,2])
            # have neighbor(s) on the clone region boundary, include background terms  
            else: 
                self.b_r[i] = 4*self.f[x,y,0] - (self.f[x-1,y,0] +self.f[x+1,y,0] + self.f[x,y-1,0] + self.f[x,y+1,0])
                self.b_g[i] = 4*self.f[x,y,1] - (self.f[x-1,y,1] +self.f[x+1,y,1] + self.f[x,y-1,1] + self.f[x,y+1,1])
                self.b_b[i] = 4*self.f[x,y,2] - (self.f[x-1,y,2] +self.f[x+1,y,2] + self.f[x,y-1,2] + self.f[x,y+1,2])
                self.b_r[i] += flag[0] * self.b[x-1,y,0] + flag[1] * self.b[x+1,y,0] + flag[2] * self.b[x,y-1,0] + flag[3] * self.b[x,y+1,0]
                self.b_g[i] += flag[0] * self.b[x-1,y,1] + flag[1] * self.b[x+1,y,1] + flag[2] * self.b[x,y-1,1] + flag[3] * self.b[x,y+1,1]
                self.b_b[i] += flag[0] * self.b[x-1,y,2] + flag[1] * self.b[x+1,y,2] + flag[2] * self.b[x,y-1,2] + flag[3] * self.b[x,y+1,2]


        # use conjugate gradient to solve for u
        u_r = splinalg.cg(self.A, self.b_r)[0]
        u_g = splinalg.cg(self.A, self.b_g)[0]
        u_b = splinalg.cg(self.A, self.b_b)[0]   

        return u_r, u_g, u_b
    
    # combine
    def combine(self):
        self.new = np.array(self.new,dtype=int)
        u_r,u_g,u_b = self.poisson_solver()

                        
        # naive copy
        for i in range(3):
            self.new[:,:,i] = self.b[:,:,i];     

        # fix cloning region
        for i in range(len(self.idx_map)):
            x, y = self.idx_map[i]
            self.new[x,y,0] = min(255,u_r[i])
            self.new[x,y,1] = min(255,u_g[i])
            self.new[x,y,2] = min(255,u_b[i])
        self.new = np.asarray(self.new, dtype='uint8')
           

if __name__ == "__main__":
    
    test = GradientDomainCloning(F, B, M)
    
    test.combine()

    result = Image.fromarray(test.new)
    
    result.save('ouptut.png')
    
    
