"""
pipeline/elevation_features.py
Real DEM elevation extraction for all 243 BBMP wards using bengaluru_dem.tif.

FIX 3: Now uses rasterio instead of PIL — preserves affine transform so
pixel↔lon/lat conversions use actual DEM metadata, not hardcoded constants.
"""

import json, math
import numpy as np


def load_dem(dem_path: str) -> tuple:
    """
    Load DEM using rasterio. Returns (array, transform).
    FIX 3: Replaces PIL Image.open (which discarded georeferencing).
    """
    import rasterio                          # type: ignore
    with rasterio.open(dem_path) as src:
        arr       = src.read(1).astype(np.float32)
        transform = src.transform
        nodata    = src.nodata if src.nodata is not None else -9999.0
    arr[arr == nodata] = np.nan
    arr[arr < 0]       = np.nan
    return arr, transform


def lonlat_to_pixel(lon: float, lat: float, transform=None) -> tuple:
    """Convert lon/lat to DEM pixel (row, col) using the affine transform."""
    if transform is None:
        raise ValueError("transform required — pass rasterio transform object")
    col = int((lon - transform.c) / transform.a)
    row = int((lat - transform.f) / transform.e)
    return row, col


def sample_polygon_elevation(dem: np.ndarray, ring: list, transform=None) -> dict:
    """
    Sample DEM within a ward polygon ring.
    Returns mean, min, variance of elevation.
    transform: rasterio Affine (required for FIX 3 coordinate conversion)
    """
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    h, w = dem.shape

    r0, c0 = lonlat_to_pixel(lon_min, lat_max, transform)
    r1, c1 = lonlat_to_pixel(lon_max, lat_min, transform)

    r0 = max(0, r0); r1 = min(h - 1, r1)
    c0 = max(0, c0); c1 = min(w - 1, c1)

    if r1 <= r0 or c1 <= c0:
        # Ward too small for bbox — use centroid sample
        lon_c = sum(lons) / len(lons)
        lat_c = sum(lats) / len(lats)
        rc, cc = lonlat_to_pixel(lon_c, lat_c, transform)
        rc = max(0, min(h - 1, rc))
        cc = max(0, min(w - 1, cc))
        v = dem[rc, cc]
        return {"mean": float(v), "min": float(v), "variance": 0.0}

    patch = dem[r0:r1+1, c0:c1+1]
    valid = patch[~np.isnan(patch)]
    if len(valid) == 0:
        return {"mean": 883.0, "min": 850.0, "variance": 100.0}

    return {
        "mean":     float(np.mean(valid)),
        "min":      float(np.min(valid)),
        "variance": float(np.var(valid)),
    }


def extract_all_ward_elevations(geojson_path: str, dem_path: str) -> dict:
    """
    Returns dict: ward_name → {mean_elevation, min_elevation, elevation_variance}
    FIX 3: Uses rasterio transform for correct coordinate mapping.
    """
    geo = json.load(open(geojson_path))
    dem, transform = load_dem(dem_path)

    results = {}
    for feat in geo["features"]:
        name = feat["properties"]["KGISWardName"]
        ring = feat["geometry"]["coordinates"][0]
        elev = sample_polygon_elevation(dem, ring, transform)
        results[name] = {
            "mean_elevation":     elev["mean"],
            "min_elevation":      elev["min"],
            "elevation_variance": elev["variance"],
        }
    return results


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(__file__))
    geo  = os.path.join(base, "../../data/data/gis/BBMP.geojson")
    dem  = os.path.join(base, "../../data/data/gis/bengaluru_dem.tif")
    elevs = extract_all_ward_elevations(geo, dem)
    vals = list(elevs.values())
    means = [v["mean_elevation"] for v in vals]
    print(f"Extracted elevation for {len(elevs)} wards")
    print(f"Elevation range: {min(means):.1f}m – {max(means):.1f}m")
    print(f"Mean city elevation: {sum(means)/len(means):.1f}m")
