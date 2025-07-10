import matplotlib.pyplot as plt
import dataretrieval.nwis as nwis
import pandas as pd
import ee
import geemap

# Trigger the authentication flow. This is a one-time setup per environment.
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()
# --- 1. Define Parameters for Data Retrieval ---
# Site for LAKE TUSCALOOSA
site_id = '02464800'
# Parameter code for Gage Height (in feet)
parameter_cd = '00065'
# Define your desired date range
start_date = '2015-01-01'
end_date = '2024-12-31'
statisticCodes = ["00003"] # 00001 = max, 00002 = min , 00003 = mean
print(f"Attempting to retrieve data for site {site_id} from {start_date} to {end_date}.")

try:
    # --- 2. Correctly Retrieve and Unpack Data ---
    # nwis.get_dv() returns a tuple: (DataFrame, metadata)
    # We unpack it into two separate variables.
    df_insitu, urlstr = nwis.get_dv(
        sites=site_id,
        parameterCd=parameter_cd,
        statCd=statisticCodes,
        start=start_date,
        end=end_date
    )

    # --- 3. Inspect the DataFrame ---
    print("\nSuccessfully retrieved data. DataFrame info:")
    df_insitu['water_level_m'] = df_insitu['00065_Mean'] * 0.3048  # converting gauge height from feet to meter
    display(df_insitu.head())

except Exception as e:
    print(f"\nAn error occurred during data retrieval or processing: {e}")
  
    """
    For a single Sentinel-2 image, calculate_area_and_volume calculates the AWEI water mask,
    surface area, and water volume.
    Returns: ee.Feature with date, area, and volume.
    """
# a function to mask out cloudy pixels
def mask_s2_clouds(image):
    """Masks clouds and cloud shadows in a Sentinel-2 SR image."""
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    # Both flags should be set to 0, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    # Return the masked image, scaled to reflectance values.
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])
def calculate_area_and_volume(image):
    global store_num
    global store_num2
    store_num= []
    store_num2= []

    # a. Calculate the AWEI water mask from the original S2 image.
    awei = image.expression(
        'B2 + 2.5 * B3 - 1.5 * (B8 + B11) - 0.25 * B12', {
            'B2': image.select('B2'), 'B3': image.select('B3'), 'B8': image.select('B8'),
            'B11': image.select('B11'), 'B12': image.select('B12')
        }
    )
  # applying a threshold of -0.2 to extract water body
    water_mask = awei.gt(-0.2)

    # b. Calculate surface area.
    area_image = water_mask.multiply(ee.Image.pixelArea())
    area_stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,      # Match bathymetry scale
        maxPixels=1e12
    )
    
    # Extract the number and convert to sq km.
    water_area_sq_km = ee.Number(area_stats.get('B2',0)).divide(1e6)
    
    # c. Calculate water volume.
  # Loading globathy datasset
    globathy_img = ee.Image("projects/sat-io/open-datasets/GLOBathy/GLOBathy_bathymetry")
  #apply the water mask extracted from AWEI to globathy image
    lake_depth = globathy_img.updateMask(water_mask)
  # multiplying depth*pixel_area for each image, so it would result to volume data for each sinle pixel.
    pixel_volume_img = ee.Image.pixelArea().multiply(lake_depth)
  # summing all the pixel values in the image to get total volume
    volume_stats = pixel_volume_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,  
        maxPixels=1e12
    )
    # Extract the number and convert to cubic km.
    total_volume_km3 = ee.Number(volume_stats.get('area',0)).divide(1e9)

    # d. Return a single feature with all properties as simple values.
    # This ensures the output of the .map() operation is a valid FeatureCollection.
    return ee.Feature(None, {
        'date': image.date().format('YYYY-MM-dd'),
        'area_km2': water_area_sq_km,
        'volume_km3': total_volume_km3
    })

# defining the ROI of LAKE TUSCALOOSA
aoi = ee.Geometry.Polygon(-87.520999, 33.263885, -87.498121,33.264235,-87.502874, 33.294227,-87.518918, 33.319746,-87.542136, 33.339118,-87.552643, 33.324916,-87.538770, 33.303648,-87.518877, 33.282364)  # Define the area of interest of the lake
 # select and preprocess sentinel-2 surface reflectance harmonized dataset
s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(aoi) \
                .filterDate(start_date,end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',10)) \
                .map(mask_s2_clouds)                                       

results_time_series = s2_collection.map(calculate_area_and_volume)
# Convert the final results to a pandas DataFrame for better data handling
results_time_series_feat = ee.FeatureCollection(results_time_series)
results_df = geemap.ee_to_df(results_time_series_feat)
results_df['date'] = pd.to_datetime(results_df['date'])
print(results_df.head())
# Calculate the mean volume for the entire period as a baseline
mean_volume = results_df['volume_km3'].mean()

# Calculate volume change relative to the mean
results_df['volume_change_km3'] = results_df['volume_km3'] - mean_volume

# Calculate the change from the previous time step
results_df['volume_change_monthly_km3'] = results_df['volume_km3'].diff()

# Display the final results
print(results_df.head())

#=======================================================================================================
# Visualization
#=======================================================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Assuming 'results_df' is your final, cleaned DataFrame.
results_df = results_df.sort_values(by='date').reset_index(drop=True)

# --- 1. Calculate the Difference (Daily Change) ---
results_df['area_diff'] = results_df['area_km2'].diff()
results_df['volume_diff'] = results_df['volume_km3'].diff()

# --- 2. Standardize the Data ---

# Columns to standardize
cols_to_standardize = ['area_km2', 'volume_km3', 'area_diff', 'volume_diff']
# Create a copy for the standardized values to keep original data safe
df_standardized = results_df.copy()
# To fit the scaler ONLY on the non-NaN data and transform it
# The [1:] index skips the first row with NaN values
df_standardized = df_standardized.iloc[1:,:]

# Initialize the scaler
scaler = StandardScaler()

df_standardized.loc[:, cols_to_standardize] = scaler.fit_transform(df_standardized.loc[:, cols_to_standardize])


# --- 3. Create the Subplots ---
# Create a figure with 1 row and 2 columns of plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=150)

# --- Plot 1: Standardized Area and Volume ---
ax1.plot(df_standardized['date'], df_standardized['area_km2'], color='royalblue', label='Standardized Area')
ax1.plot(df_standardized['date'], df_standardized['volume_km3'], color='darkred', label='Standardized Volume')
ax1.set_title('Standardized Area and Volume Time Series', fontsize=16)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Standardized Value (Z-score)', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle=':')

# --- Plot 2: Standardized Changes in Area and Volume ---
ax2.plot(df_standardized['date'], df_standardized['area_diff'], color='orange', label='Standardized Daily Area Change')
ax2.plot(df_standardized['date'], df_standardized['volume_diff'], color='darkgreen', label='Standardized Daily Volume Change')
ax2.set_title('Standardized Daily Changes', fontsize=16)
ax2.set_xlabel('Date', fontsize=12)
# ax2 does not need a y-label as it shares the axis with ax1
ax2.legend()
ax2.grid(True, linestyle=':')

# --- 4. Customize Ticks for Both Plots ---
for ax in [ax1, ax2]:
    # Set x-axis labels to be vertical
    ax.tick_params(axis='x', labelrotation=90)
    # Improve date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # A tick every 3 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

def water_mask_func (image,threshold):
    awei = image.expression('B2 + 2.5 * B3 - 1.5 * (B8 + B11) - 0.25 * B12',
                   {'B2': image.select('B2'), 'B3': image.select('B3'), 'B8': image.select('B8'),
                    'B11': image.select('B11'), 'B12': image.select('B12')})
    return awei.gt(threshold)

# Selecting a sample image from image collection
single_image = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
print("Using image:", single_image.id().getInfo())
# loading the globathy dataset
globathy_img = ee.Image("projects/sat-io/open-datasets/GLOBathy/GLOBathy_bathymetry")
# extracting water mask from sample single image
threshold = -0.1
water_mask_single = water_mask_func(single_image,threshold)
rgb_vis_params = {'min':0,'max':0.3,'bands':['B4','B3','B2'],}
mask_vis_params = {'palette': ['black', 'cyan']}
depth_vis_params = {'min': 0, 'max': 30, 'palette': ['blue', 'cyan', 'yellow', 'red']}

Map = geemap.Map(basemap='HYBRID')
Map.addLayer(single_image, rgb_vis_params, 'True Color Image')
Map.addLayer(globathy_img, depth_vis_params, 'GLOBathy Depth')
Map.addLayer(water_mask_single.selfMask(), mask_vis_params, f'Water Mask (AWEI > {threshold})')
Map.centerObject(aoi, 13)
display(Map)



# --- 5. Show the Final Figure ---
# Adjust layout to prevent labels from overlapping
plt.tight_layout()
plt.show()

# ============================================================================================
# Validation
#+============================================================================================
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# --- 1. Prepare and Merge DataFrames ---
# This part remains the same.
# Ensure the index of the in-situ data is datetime for merging
df_insitu.index = pd.to_datetime(df_insitu.index)

# Ensure the 'date' column of the results data is datetime
results_df['date'] = pd.to_datetime(results_df['date'])

# Set the 'date' column as the index for the satellite results
df_satellite = results_df.set_index('date')

# Merge the two dataframes based on their date index.
merged_df = pd.merge(
    df_insitu[['water_level_m']],
    df_satellite[['volume_km3']],
    left_index=True,
    right_index=True,
    how='inner'
)

# --- 2. Standardize Data for Comparison ---
# This part remains the same.
scaler = StandardScaler()
merged_df[['water_level_scaled', 'volume_scaled']] = scaler.fit_transform(merged_df[['water_level_m', 'volume_km3']])

# --- 3. Calculate RMSE ---
# This part remains the same.
rmse = np.sqrt(mean_squared_error(merged_df['water_level_scaled'], merged_df['volume_scaled']))

print(f"Validation DataFrame (first 5 rows):\n{merged_df.head()}\n")
print(f"Root Mean Square Error (RMSE) between standardized gauge height and satellite volume: {rmse:.4f}")

# --- 4. Create the Scatter Plot and Regression Line with Matplotlib and Sklearn ---
plt.figure(figsize=(12, 8), dpi=100)

# Create the scatter plot
plt.scatter(
    merged_df['water_level_scaled'],
    merged_df['volume_scaled'],
    alpha=0.6,
    color='royalblue',
    label='Data Points'
)

# --- Calculate and Plot the Regression Line ---
# a. Prepare data for scikit-learn's LinearRegression model
X = merged_df[['water_level_scaled']] # Independent variable (needs to be 2D)
y = merged_df['volume_scaled']      # Dependent variable

# b. Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# c. Generate points for the regression line
# We'll create a line based on the min and max x-values to draw across the plot
x_fit = np.array([X.min(), X.max()]).reshape(-1, 1)
y_fit = model.predict(x_fit)

# d. Plot the regression line
plt.plot(x_fit, y_fit, color='darkred', linestyle='--', label='Linear Regression')

# --- Finalize the Plot ---
# Add titles and labels for clarity
plt.title('Validation: In-Situ Gauge Height vs. Satellite-Derived Volume', fontsize=16, pad=20)
plt.xlabel('Standardized In-Situ Water Level (Z-score)', fontsize=12)
plt.ylabel('Standardized Satellite-Derived Volume (Z-score)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# Add the RMSE value as text on the plot for context
plt.text(
    0.05, 0.95,
    f'RMSE = {rmse:.4f}',
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
)

# Show the plot
plt.show()
