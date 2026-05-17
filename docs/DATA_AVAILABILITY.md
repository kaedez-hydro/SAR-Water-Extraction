# Data Availability Statement

## Summary

This study develops a cross-sensor water body extraction framework for SAR imagery. The repository contains all code necessary to reproduce the methodology using publicly available Sentinel-1 data.

## Data Access Policy

### Code — Fully Open Source

All code in this repository is released under the MIT License and can be freely used, modified, and redistributed:

- Feature extraction pipeline (GLCM, LBP, Wavelet)
- Machine learning training and classification (Random Forest, SVM, XGBoost)
- SHAP model interpretability analysis
- Google Earth Engine S1-GNDPI implementation

### Sentinel-1 Data — Publicly Available

Sentinel-1 GRD data is freely and openly distributed by the European Space Agency (ESA) through:

- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- [Google Earth Engine](https://earthengine.google.com/) (COPERNICUS/S1_GRD collection)
- [Alaska Satellite Facility (ASF)](https://asf.alaska.edu/)

The GEE script in `gee/s1_gndpi.js` can be used to reproduce the Sentinel-1 water extraction results without downloading data locally.

### GF-3 Data — Restricted

In accordance with the **Surveying and Mapping Law of the People's Republic of China** (《中华人民共和国测绘法》) and the **Data Security Law of the People's Republic of China** (《中华人民共和国数据安全法》), the following data used in the associated study **cannot** be publicly released:

1. **GF-3 (Gaofen-3) SL SAR imagery** — China's civilian high-resolution SAR satellite data is subject to distribution controls for sensitive geographic locations.
2. **High-resolution derivative products over flood detention areas** — Classification maps and enhanced imagery over critical water infrastructure and flood detention basins contain spatially explicit information that may fall under regulated categories.

Researchers affiliated with Chinese institutions may apply for GF-3 data access through the **China Centre for Resources Satellite Data and Application (CRESDA)** ([data.cresda.cn](https://data.cresda.cn/#/home)).

## Reproducibility Guidance

To fully reproduce the methodology without access to GF-3 data:

1. **Acquire Sentinel-1 GRD data** over your study area from any of the sources listed above.
2. **Run the S1-GNDPI algorithm** (`gee/s1_gndpi.js`) to generate enhanced water index imagery.
3. **Generate training labels** by thresholding the S1-GNDPI output or using existing water body datasets (e.g., JRC Global Surface Water).
4. **Train and classify** using the Python pipeline in `src/` with your Sentinel-1 derived data.

The cross-sensor generalization capability demonstrated in the paper can be validated by training on Sentinel-1 data and testing on Sentinel-1 data from different dates or adjacent regions.

## Contact

For questions regarding data access or reproduction of results, please contact the author or open a GitHub Issue.
