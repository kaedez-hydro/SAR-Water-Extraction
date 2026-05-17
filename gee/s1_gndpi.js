// ============================================================
// S1-GNDPI: Sentinel-1 Generalized Normalized Difference
// Polarimetric Index for Water Body Extraction
// ============================================================
// Reference: This script implements the S1-GNDPI algorithm for
// water body extraction from Sentinel-1 GRD imagery.
// The index is designed to enhance the contrast between water
// and land surfaces using dual-polarization (VV+VH) SAR data.
//
// Platform: Google Earth Engine (JavaScript API)
// Data: COPERNICUS/S1_GRD (Sentinel-1 Ground Range Detected)
//
// Usage:
//   1. Copy this script into the GEE Code Editor
//   2. Adjust the ROI and date range as needed
//   3. Run the script to visualize results
// ============================================================

// ================= 1. Basic Configuration =================
var roi = ee.Geometry.Rectangle([116.3, 38.9, 117.1, 39.1]);
Map.centerObject(roi, 11);
var startDate = '2023-08-04';
var endDate = '2023-08-16';

// ================= 2. Data Loading & Preprocessing =================
var s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'));

// Mosaic and clip to ROI
var s1 = s1_collection.mosaic().clip(roi);

// Speckle filtering (critical step: reduces granular noise
// while preserving edges)
// Radius = 20 meters
var s1_filtered = s1.focal_median(20, 'circle', 'meters');

// ================= 3. S1-GNDPI Calculation =================
// Formula: GNDPI = 0.5 * ln(VV_linear + VH_linear) + 1.90
//
// The constant 1.90 is a calibration offset derived from the
// statistical distribution of the non-coherent sum, ensuring
// that typical water pixels (~ -20 dB) map to ~ -0.4 and
// typical urban pixels (~ 0 dB) map to ~ 1.9.

// Convert dB to linear: 10^(dB/10)
var vv_lin = ee.Image(10).pow(s1_filtered.select('VV').divide(10));
var vh_lin = ee.Image(10).pow(s1_filtered.select('VH').divide(10));

// Non-coherent summation
var sum_lin = vv_lin.add(vh_lin);

// Logarithmic transform and normalization
var gndpi = sum_lin.log().multiply(0.5).add(1.90).rename('S1_GNDPI');

// ================= 4. Visualization =================
// Based on the corrected formula:
//   Water (~ -20 dB)  -> GNDPI ~ -0.4 (dark)
//   Land  (~ -10 dB)  -> GNDPI ~  0.8 (gray)
//   Urban (~   0 dB)  -> GNDPI ~  1.9 (bright)

var min_val = -0.5;
var max_val =  2.5;

// Layer 1: Original VV (for comparison)
var vis_vv = {min: -25, max: -5, palette: ['black', 'white']};
Map.addLayer(s1_filtered.select('VV'), vis_vv, 'Fig(a) Original VV');

// Layer 2: Original VH
var vis_vh = {min: -30, max: -12, palette: ['black', 'white']};
Map.addLayer(s1_filtered.select('VH'), vis_vh, 'Fig(b) Original VH');

// Layer 3: S1-GNDPI (grayscale)
Map.addLayer(gndpi, {min: min_val, max: max_val, palette: ['black', 'white']},
             'Fig(c) S1-GNDPI (this study)');

// Layer 4: S1-GNDPI (pseudo-color, optional heatmap)
var vis_color = {
  min: min_val,
  max: max_val,
  palette: ['000080', '0000FF', '00FFFF', 'FFFF00', 'FF0000']
};
Map.addLayer(gndpi, vis_color, 'S1-GNDPI (pseudo-color)');

// ================= 5. Statistics (Optional) =================
// Verify the value range in the Console
print('S1-GNDPI Percentile Statistics:', gndpi.reduceRegion({
  reducer: ee.Reducer.percentile([2, 98]),
  geometry: roi,
  scale: 100,
  bestEffort: true
}));

// ================= 6. Export (Optional) =================
// Export.image.toDrive({
//   image: gndpi,
//   description: 'S1_GNDPI_export',
//   scale: 10,
//   region: roi,
//   maxPixels: 1e9
// });
