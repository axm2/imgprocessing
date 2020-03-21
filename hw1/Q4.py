from laspy import file
import numpy as np
from PIL import Image

source = "input/17258975.las"
las = file.File(source, mode="r")  # reading of las file
mmmin = las.header.min  # return [longmin,latmin,zmin]
mmmax = las.header.max  # return [longmax,latmax,zmax]

longitude = (int) ((mmmax[0] - mmmin[0]) / 8)
latitude = (int) ((mmmax[1] - mmmin[1]) / 8)
temp = np.zeros((longitude, latitude, 2))
# now we fill data points into temp arr
print("filling data points")
for x, y, z in np.nditer([las.x, las.y, las.z]): # x = long y = lat z = alt
    long = (int) ((x-mmmin[0]) / 8) - 1
    lat = (int) ((y-mmmin[1]) / 8) - 1
    alt = z
    temp[long][lat][0] += z # adding altitude
    temp[long][lat][1] += 1 # counter for how many points at these coords

# now we take the average of each pixel
print("taking averages")
temp2 = np.zeros((longitude, latitude))
for i in range(longitude):
    for j in range(latitude):
        if temp[i][j][1] == 0:
            temp2[i][j] = 0
        else:
            temp2[i][j] = temp[i][j][0] / temp[i][j][1]

# now we convert it to an image
print("converting to image")
factor = 255 / max(temp2.flatten())
img = temp2*factor
im = Image.fromarray(img)
im = im.convert("L")
im.save('output/Q4_raster.png')
