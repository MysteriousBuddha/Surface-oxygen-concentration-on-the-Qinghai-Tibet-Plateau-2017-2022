import numpy as np
import pandas as pd
from osgeo import gdal, osr
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


###################################
# Construct the oxygen concentration estimation model

# Read data and select the data to be estimated
df = pd.read_excel('Surface oxygen concentration on the Qinghai-Tibet Plateau (2018-2020).xlsx', parse_dates=['Time'])
data = df[(df['Time'].dt.month>=6) & (df['Time'].dt.month<=8)]

# Normalized the factors
nE = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['Elevation (m)'].values.reshape(-1, 1))
nT = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['Temperature (°C)'].values.reshape(-1, 1))
nL = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['LAI'].values.reshape(-1, 1))

# Calculate the estimated temporary variable
tmp = []
for i in range(len(nE)):
    tmp.append(-0.3958 * nE[i][0] + 0.3550 * nT[i][0] + 0.2492 * nL[i][0])
oc = data['Oxygen concentration (%)'].values.tolist()
toc = pd.DataFrame(data=np.array([tmp, oc], dtype=float).T, columns=['Tmp', 'OC'])

# Construct the oxygen concentration estimation model
result = []
for m in range(3, 366):
    a = []
    b = []
    rmse = []
    for t in range(50):
        toc_train = toc.sample(m)
        toc_test = toc[~toc.index.isin(toc_train.index)]  # Test set
        x_train = toc_train['Tmp'].values
        y_train = toc_train['OC'].values
        f = LinearRegression()
        x_train = x_train.reshape(-1, 1)
        f.fit(x_train, y_train)
        a.append(f.coef_[0])
        b.append(f.intercept_)
        x_test = toc_test['Tmp'].values
        y_test = toc_test['OC'].values
        x_test = x_test.reshape(-1, 1)
        y_predict = f.predict(x_test)
        rmse.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    result.append([i, np.mean(a), np.mean(b), np.mean(rmse), np.std(rmse)])
dr = pd.DataFrame(columns=['num', 'a', 'b', 'rmse_mean', 'rmse_std'], data=result)

# Select the most robust model
ma = dr.loc[dr.idxmin()['rmse_std']]['a']
mb = dr.loc[dr.idxmin()['rmse_std']]['b']


###################################
# Estimate the oxygen concentration distribution data

# Read Tiff
def readTiff(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    im_data = band.ReadAsArray(0, 0, im_width, im_height).astype(float)
    del dataset
    return im_data

# Save Tiff
def writeTiff(op_data, lu_lon, lu_lat, op_size, op_nodata, op_filename):
    driver = gdal.GetDriverByName('GTiff')
    d_type = gdal.GDT_Float32
    x_size = op_data.shape[-1]
    y_size = op_data.shape[-2]
    dataset = driver.Create(op_filename, x_size, y_size, 1, d_type)
    transform = (lu_lon, op_size, 0, lu_lat, 0, -op_size)
    dataset.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(op_data)
    dataset.GetRasterBand(1).SetNoDataValue(op_nodata)
    dataset.FlushCache()

# Read raster data
dem = readTiff('raster//dem.tif')  # Read dem data
dem = np.where(dem == dem[0][0], np.nan, dem)
tem = readTiff('raster//tem.tif')  # Read temperature data
tem = np.where(tem == tem[0][0], np.nan, tem)
lai = readTiff('raster//lai.tif')  # Read LAI data
lai = np.where(lai == lai[0][0], np.nan, lai)

# Normalized the factors
ne = np.empty(shape=dem.shape)
nt = np.empty(shape=tem.shape)
nl = np.empty(shape=lai.shape)
dem_min = np.nanmin(dem)
dem_max = np.nanmax(dem)
tem_min = np.nanmin(tem)
tem_max = np.nanmax(tem)
lai_min = np.nanmin(lai)
lai_max = np.nanmax(lai)
for i in range(len(ne)):
    for j in range(len(ne[0])):
        ne[i][j] = (dem[i][j] - dem_min) / (dem_max - dem_min)
        nt[i][j] = (tem[i][j] - tem_min) / (tem_max - tem_min)
        nl[i][j] = (lai[i][j] - lai_min) / (lai_max - lai_min)

# Estimate the oxygen concentration distribution data
ocs = np.empty(shape=dem.shape)
for i in range(len(ocs)):
    for j in range(len(ocs[0])):
        tmps = -0.3958 * ne[i][j] + 0.3550 * nt[i][j] + 0.2492 * nl[i][j]
        ocs[i][j] = ma * tmps + mb

# Save the result
dataset = gdal.Open('raster//dem.tif')
transform = dataset.GetGeoTransform()
filename = 'oxygen concentration.tif'
writeTiff(ocs, transform[0], transform[3], transform[1], np.nan, filename)