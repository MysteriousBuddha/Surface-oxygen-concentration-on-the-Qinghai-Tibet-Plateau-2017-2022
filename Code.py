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
data = pd.read_excel('Surface oxygen concentration on the Qinghai-Tibet Plateau (2018-2020).xlsx', parse_dates=['Time'])

# Normalized the factors
E_MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
T_MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
L_MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
nE = E_MinMaxScaler.fit_transform(data['Elevation (m)'].values.reshape(-1, 1))
nT = T_MinMaxScaler.fit_transform(data['Temperature (°C)'].values.reshape(-1, 1))
nL = L_MinMaxScaler.fit_transform(data['LAI'].values.reshape(-1, 1))

# Calculate the estimated temporary variable
tmp = []
for i in range(len(nE)):
    tmp.append(-0.3958 * nE[i][0] + 0.3550 * nT[i][0] + 0.2492 * nL[i][0])
oc = data['Oxygen concentration (%)'].values.tolist()
toc = pd.DataFrame(data=np.array([tmp, oc], dtype=float).T, columns=['Tmp', 'OC'])

# Construct the oxygen concentration estimation model
result = []
for m in range(3, len(data)-2):
    a = []
    b = []
    rmse = []
    for t in range(50000):
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
        y_mean = np.mean(y_test)
        ssr = np.sum(np.power(y_predict - y_mean, 2))
        sse = np.sum(np.power(y_test - y_predict, 2))
        Fstats = ssr / (sse / (len(toc_test) - 2))
        p_value = stats.f.sf(Fstats, 1, len(toc_test) - 2)
        rmse.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        r2.append(f.score(x_test, y_test))
        p.append(p_value)
    result.append([m, np.mean(a), np.mean(b), np.mean(rmse), np.std(rmse), np.mean(r2), np.mean(p)])
dr = pd.DataFrame(columns=['num', 'a', 'b', 'rmse_mean', 'rmse_std', 'r2_mean', 'p_mean'], data=result)
dr.to_excel('estimation model.xlsx')

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
for i in tqdm(range(len(ne))):
    for j in range(len(ne[0])):
        ne[i][j] = E_MinMaxScaler.transform(dem[i][j].reshape(-1, 1))
        nt[i][j] = T_MinMaxScaler.transform(tem[i][j].reshape(-1, 1))
        nl[i][j] = L_MinMaxScaler.transform(lai[i][j].reshape(-1, 1))

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
