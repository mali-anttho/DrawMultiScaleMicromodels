import os
import h5py
import numpy as np
from skimage import morphology as morph
from skimage.morphology import disk, ellipse
from skimage.filters import median
import random

# This part adds 2 levels of micro-porosities
# Load the binary image, perform morphological operations, and ...
# add micro-porosities in between solids

#%%

class CircleImageGenerator:
    
    def __init__(self,  stride=40, offset=30, x_dim=256, y_dim=256, xydevmax=15, raddevmax=5):

        self.stride = stride
        self.offset = offset
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.xdevmax = xydevmax
        self.ydevmax = xydevmax
        self.raddevmax = raddevmax

    def gen_coor(self, rad):
        x_coor = []
        y_coor = []
        r_coor = []
        
        for j in range(0, self.x_dim + self.offset, self.stride):
            counter = 0
            for k in range(0, self.y_dim + self.offset, self.stride):
                if (counter % 2) == 0:
                    x_coor.append(j + random.randint(-self.xdevmax, self.xdevmax))
                    y_coor.append(k + random.randint(-self.ydevmax, self.ydevmax))
                    r_coor.append(rad + random.randint(-self.raddevmax, self.raddevmax))
                    x_coor.append(j + random.randint(-self.xdevmax, self.xdevmax))
                    y_coor.append(k + self.stride + random.randint(-self.ydevmax, self.ydevmax)) # self.stride
                    r_coor.append(rad + random.randint(-self.raddevmax, self.raddevmax))
                else:
                    x_coor.append(j + self.offset + random.randint(-self.xdevmax, self.xdevmax)) # +self.offset
                    y_coor.append(k + self.offset + random.randint(-self.ydevmax, self.ydevmax)) # +self.offset
                    r_coor.append(rad + random.randint(-self.raddevmax, self.raddevmax))
                    x_coor.append(j - self.offset + random.randint(-self.xdevmax, self.xdevmax)) # -self.offset
                    y_coor.append(k - self.offset + random.randint(-self.ydevmax, self.ydevmax)) # -self.offset
                    r_coor.append(rad + random.randint(-self.raddevmax, self.raddevmax))
                    
                    
                counter += 1
                
                
        return x_coor, y_coor, r_coor
    
    def draw_circles(self, x_coor, y_coor, r_coor):
        
        img=255*np.ones((self.x_dim, self.y_dim))
        
        for i in range(len(x_coor)):
            x_o, y_o, r_o = x_coor[i], y_coor[i], r_coor[i]
            
            for j in range(x_o - r_o, x_o + r_o, 1):
                for k in range(y_o - r_o, y_o + r_o, 1):
                    if j < self.x_dim and k < self.y_dim:
                        if (j - x_o) ** 2 + (k - y_o) ** 2 <= r_o ** 2:
                            img[j, k] = 0
                        else:
                            continue
                        
                    else:
                        continue
                                

        return img

#%% Create Mask


class addMicroPorosity:
    
    def __init__(self, depth=64, solid_thresh=0.33, bins=[-100, 100, 200]):
        
        self.depth=depth
        self.thresh=solid_thresh
        self.bins=bins
        
    def createMask(self, img, r_coor):
        
        min_rad=min(r_coor)+1
        
        mask1=(morph.opening(img, disk(min_rad-1)))
        mask2=(morph.closing(img, disk(min_rad-1)))
        
        solid=np.count_nonzero(mask2==0)/(256*256)
        
            
        while solid >self.thresh:
            
            mask2=(morph.dilation(mask2, ellipse(2, 1)))
            mask2=(morph.closing(mask2, ellipse(round(0.5*min_rad), round(0.25*min_rad))))
            solid=np.count_nonzero(mask2==0)/(256*256)
            
        
        mask=(mask2-mask1)-0.5*img
        
        mask = median(mask, disk(2))
        
        return mask
    
    
    def superImpose(self, img2):
        
        stats=np.zeros((1,4))
        
        regions = np.digitize(img2, self.bins)
        
        region2=regions==2
        region3=regions==3
        
        micro1=100*np.count_nonzero(region2)/(256*256)
        micro2=100*np.count_nonzero(region3)/(256*256)
        
        max_it=0
        
        while np.std([micro1,micro2]) > 5 and max_it<=100:
            
            if micro1>micro2:
                
                region2=morph.erosion(region2, disk(1))
                region3=morph.dilation(region3, disk(1))
                
            else:
                
                region2=morph.dilation(region2, disk(1))
                region3=morph.erosion(region3, disk(1))
                
                
            micro1=100*np.count_nonzero(region2)/(256*256)
            micro2=100*np.count_nonzero(region3)/(256*256)
        
            max_it=max_it+1
        
        final_img=100*(region2)+200*(region3)
        
        final_img[regions==1]=255
        final_img[regions==0]=0        
        final_img = median(final_img, ellipse(3,2))
        final_img=final_img.astype('uint8')
        regions2 = np.digitize(final_img, bins=[50, 128, 225])
        
        final_img =0*(regions2==0)+1*(regions2==1)+2*(regions2==2)+3*(regions2==3)
        final_img2=0*(regions2==0)+1*(regions2==1)+1*(regions2==2)+255*(regions2==3)
        gray_area=(final_img2==1)*np.random.default_rng().normal(150, 25, (256,256)).astype(int)-1
        final_img2=final_img2+gray_area
        final_img3=np.repeat(final_img[np.newaxis, :, :], self.depth, axis=0)
        
        stats[0, 0]=np.count_nonzero(final_img==0)/(256*256)
        stats[0, 1]=np.count_nonzero(final_img==1)/(256*256)
        stats[0, 2]=np.count_nonzero(final_img==2)/(256*256)
        stats[0, 3]=np.count_nonzero(final_img==3)/(256*256)
        
        return final_img, final_img2, final_img3, stats

#%%
class export_data:
    
    def __init__(self,  stride=20, offset=20, xydevmax=10, raddevmax=5):
        self.stride = stride
        self.offset = offset
        self.xdevmax = xydevmax
        self.ydevmax = xydevmax
        self.raddevmax = raddevmax
        self.target_direc = os.getcwd() 

    def save_to_hdf5(self, img, final_img, final_img2, stats, rad, x_coor, y_coor, r_coor, idx, raddevmax, xdevmax):
        
        idx=idx+200
        filename = self.target_direc + "/example_images/"+ "rad" + str(rad * 10) + "_raddev" + str(raddevmax * 10) + "_coordev" + str(xdevmax * 10) + "_" + f"{idx:03}" + ".hdf5"
        #g = h5py.File(filename, 'w')
        with h5py.File(filename, 'w') as g:
            g.create_dataset('x_coor', data=np.array(x_coor), dtype="float", compression="gzip")
            g.create_dataset('y_coor', data=np.array(y_coor), dtype="float", compression="gzip")
            g.create_dataset('rad', data=np.array(r_coor), dtype="uint16", compression="gzip")
            g.create_dataset('binary_image', data=img, dtype="uint8", compression="gzip")
            g.create_dataset('phase4_image', data=final_img, dtype="uint8", compression="gzip")
            g.create_dataset('phase3_image', data=final_img2, dtype="uint8", compression="gzip")
            g.attrs['Pores'] = stats[0,0] * 100
            g.attrs['Micro Pores 1'] = stats[0,1] * 100
            g.attrs['Micro Pores 2'] = stats[0,2] * 100
            g.attrs['Solid'] = stats[0,3] * 100
            g.attrs['rad'] = rad * 10
            g.attrs['stride'] = self.stride * 10
            g.attrs['offset'] = self.offset * 10
            g.attrs['xdevmax'] = self.xdevmax * 10
            g.attrs['ydevmax'] = self.ydevmax * 10
            g.attrs['raddevmax'] = self.raddevmax * 10

    
    def save_as_rawData(self, img, rad, idx, raddevmax, xdevmax):
        
        idx=idx+200
        savename = self.target_direc + '/example_images/' + "rad" + str(rad * 10) + "_raddev" + str(raddevmax * 10) + "_coordev" + str(xdevmax * 10) + "_r" + f"{idx:03}" + '_3d.raw'
        img.astype('int8').tofile( savename)

#%%

generator = CircleImageGenerator()
mod_img=addMicroPorosity()
saveData=export_data()

for rad in range(10,15,1):


    for idx in range(0, 1, 1):
    
        x_coor, y_coor, r_coor = generator.gen_coor(rad)   # Generate coordinates and 
        
        img=generator.draw_circles(x_coor, y_coor, r_coor) # Generate the image
        
        mask=mod_img.createMask(img, r_coor)
        
        final_img, final_img2, final_img3, stats=mod_img.superImpose(mask)
        
        saveData.save_to_hdf5(img, final_img, final_img2, stats, rad, x_coor, y_coor, r_coor, idx, generator.raddevmax, generator.xdevmax)
        
        saveData.save_as_rawData(final_img3, rad, idx, generator.raddevmax, generator.xdevmax)
        
        print('rad=' + str(rad) + ', image =' +str( idx))