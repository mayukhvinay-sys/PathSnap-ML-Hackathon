from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


def project_xy(lon, lat, center_lon=None, center_lat=None):
    """Project WGS84 coordinates to trip-centred azimuthal equidistant metres."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if center_lon is None:
        center_lon = float(np.nanmean(lon))
    if center_lat is None:
        center_lat = float(np.nanmean(lat))
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    return np.asarray(x_m), np.asarray(y_m)
