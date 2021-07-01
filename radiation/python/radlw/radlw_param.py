
from radiation.python.phys_const import *

# num of total spectral bands
nbands = 16         
# num of total g-points
ngptlw = 140       
# lookup table dimension
ntbl = 10000   
# max num of absorbing gases
macgas = 7      
# num of halocarbon gasees
maxxsec = 4     
# num of ref rates of binary species
nrates = 6    
# dim for plank function table
nplnk  = 181

nbdlw = nbands

# \name Number of g-point in each band
NG01=10, NG02=12, NG03=16, NG04=14, NG05=16, NG06=8
NG07=12, NG08=8, NG09=12, NG10=6, NG11=8, NG12=8
NG13=4, NG14=2, NG15=2, NG16=2

# \name Begining index of each band
NS01=0, NS02=10, NS03=22, NS04=38, NS05=52, NS06=68
NS07=76, NS08=88, NS09=96, NS10=108, NS11=114,        
NS12=122, NS13=130, NS14=134, NS15=136, NS16=138

# band indices for each g-point
NGB = [10*1, 12*2, 16*3, 14*4, 16*5, 8*6, 12*7,  8*8, 12*9,
       6*10, 8*11, 8*12, 4*13, 2*14, 2*15, 2*16]  

# Band spectrum structures (wavenumber is 1/cm
wvnlw1 = [10.,  350.,  500.,  630.,  700.,  820.,  980., 1080.,
          1180., 1390., 1480., 1800., 2080., 2250., 2380., 2600.]
wvnlw2 = [350.,  500.,  630.,  700.,  820.,  980., 1080., 1180.,
          1390., 1480., 1800., 2080., 2250., 2380., 2600., 3250.]

delwave = [340., 150., 130.,  70., 120., 160., 100., 100., 210.,
           90., 320., 280., 170., 130., 220., 650.]
