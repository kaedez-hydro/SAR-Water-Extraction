#!/usr/bin/env python3
"""
Generate synthetic SAR-like demo data for pipeline testing.

Creates:
  - demo_sar.tif   : 100x100 pixel simulated SAR backscatter (dB)
  - demo_labels.shp: polygon labels (0=non-water, 1=water)

No real satellite data — fully synthetic, zero compliance risk.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Point, Polygon
import geopandas as gpd
import os

np.random.seed(42)
SIZE = 100
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_sar():
    """Generate a 100x100 simulated SAR image with a water channel."""
    # Background land: ~ -10 dB mean, speckle noise
    land = np.random.normal(-10, 2.5, (SIZE, SIZE))

    # Diagonal river channel: low backscatter (~ -20 dB)
    water_mask = np.zeros((SIZE, SIZE), dtype=bool)
    for i in range(SIZE):
        for j in range(SIZE):
            # A meandering band
            center = 50 + 12 * np.sin(j * 0.08) + (j - 50) * 0.3
            if abs(i - center) < 6:
                water_mask[i, j] = True

    # Add water pixels
    water = np.random.normal(-20, 1.5, (SIZE, SIZE))
    sar = np.where(water_mask, water, land)

    # Add a small urban/bright patch
    urban = np.random.normal(-2, 2.0, (SIZE, SIZE))
    urban_mask = np.zeros((SIZE, SIZE), dtype=bool)
    urban_mask[15:30, 60:75] = True
    sar = np.where(urban_mask, urban, sar)

    # Save as GeoTIFF (WGS84, arbitrary origin)
    transform = from_origin(116.5, 39.0, 0.001, 0.001)
    tif_path = os.path.join(OUT_DIR, "demo_sar.tif")
    with rasterio.open(tif_path, "w", driver="GTiff", height=SIZE, width=SIZE,
                       count=1, dtype=rasterio.float32, crs="EPSG:4326",
                       transform=transform) as dst:
        dst.write(sar.astype(np.float32), 1)
    print(f"[OK] {tif_path} — {SIZE}x{SIZE} px, range [{sar.min():.1f}, {sar.max():.1f}] dB")
    return tif_path


def make_labels():
    """Create label polygons corresponding to the synthetic SAR scene."""
    # Image extent: X=[116.500, 116.600], Y=[38.900, 39.000]
    # Water polygon (roughly covering the diagonal river band)
    water_poly = Polygon([
        (116.505, 38.995), (116.515, 38.990), (116.525, 38.985),
        (116.535, 38.980), (116.545, 38.975), (116.555, 38.970),
        (116.565, 38.965), (116.575, 38.960), (116.585, 38.955),
        (116.590, 38.950), (116.585, 38.945), (116.575, 38.942),
        (116.565, 38.940), (116.555, 38.942), (116.545, 38.945),
        (116.535, 38.950), (116.525, 38.955), (116.515, 38.960),
        (116.505, 38.965), (116.500, 38.972), (116.500, 38.982),
        (116.505, 38.990),
    ])

    # Non-water polygon (upper-left land area)
    land_left = Polygon([
        (116.500, 38.995), (116.520, 38.995), (116.520, 38.980),
        (116.500, 38.975),
    ])

    # Non-water polygon (lower-right land area + urban bright spot)
    land_right = Polygon([
        (116.550, 38.940), (116.580, 38.935), (116.600, 38.920),
        (116.600, 38.905), (116.550, 38.900), (116.550, 38.940),
    ])

    gdf = gpd.GeoDataFrame({
        'label': [1, 0, 0],
        'geometry': [water_poly, land_left, land_right]
    }, crs="EPSG:4326")

    shp_path = os.path.join(OUT_DIR, "demo_labels.shp")
    gdf.to_file(shp_path)
    print(f"[OK] {shp_path} — {len(gdf)} polygons, labels: {list(gdf['label'])}")
    return shp_path


if __name__ == "__main__":
    make_sar()
    make_labels()
    print("\nDemo data ready. Run the pipeline with:\n"
          "  cd ../src\n"
          "  python main.py --config ../example/demo_config.yaml")
