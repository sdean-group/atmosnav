import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


def make_map_axis(polar=False, central_longitude=0, ncol=1, nrow=1, pos=1):
    t1 = time.time()
    if polar:
        proj = ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)
    else:
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax = plt.subplot(ncol, nrow, pos, projection=proj)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none"
    )
    ax.add_feature(states_provinces, edgecolor="gray")
    t2 = time.time()
    # print("making map axis took %fs"%((t2-t1)*1.))
    return ax

from datetime import datetime

def plot_altitude(r):
    dates = [datetime.fromtimestamp(t) for t in r['t']] 
    plt.plot(dates, r['h'])
    plt.plot(dates, r['lbnd'], '--', color='black', alpha=0.5)
    plt.plot(dates, r['ubnd'], '--', color='black', alpha=0.5)
    plt.show()

def plot_lat_lon(r):
    plt.scatter(r['lon'], r['lat'])
    plt.show()

def plot_on_map(r, filename=None):
    dates = [datetime.fromtimestamp(t) for t in r['t']] 
    plt.figure(figsize=(10,6))
    ax1 = make_map_axis(ncol=2, nrow=1, pos=1) #oops, ncol and nrows are flipped here
    ax1.plot(r['lon'],r['lat'])
    ax2 = plt.subplot(2,1,2)

    ax2.plot(dates, r['h'])
    ax2.plot(dates, r['lbnd'], '--', color='black', alpha=0.5)
    ax2.plot(dates, r['ubnd'], '--', color='black', alpha=0.5)

    ax2.grid()
    ax2.set_ylabel('Altitude (km)')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def set_map_axis(polar=False):
    ax = make_map_axis(polar)
    # t1 = time.time()
    plt.sca(ax)
    # t2 = time.time()
    # print("stetting map axis took %fms"%((t2-t1)*1000))


def plot_on_map2(rec, axes=None, reduction=np.mean, day_ticks=False, **kwargs):
    if axes == None:
        axes = plt.gca()
    tt = rec.t[0, :] - rec.t[0, 0]
    d1_ind = np.argmin(np.abs(tt - np.arange(0, 8)[:, np.newaxis] * 60 * 60 * 24), axis=1)
    h6_ind = np.argmin(np.abs(tt - np.arange(0, 8 * 6)[:, np.newaxis] * 60 * 60 * 6), axis=1)
    if reduction is not None:
        rlon, rlat = reduction(rec.lon, axis=0), reduction(rec.lat, axis=0)
        axes.plot(rlon, rlat, transform=ccrs.PlateCarree(), **kwargs)
        if day_ticks:
            axes.plot(rlon[d1_ind], rlat[d1_ind], ".", transform=ccrs.PlateCarree(), **kwargs, markeredgecolor="black")

    else:
        for i in range(rec.lon.shape[0]):
            axes.plot(rec.lon[i, :], rec.lat[i, :], transform=ccrs.PlateCarree(), **kwargs)
            if day_ticks:
                axes.plot(
                    rec.lon[i, :][d1_ind],
                    rec.lat[i, :][d1_ind],
                    ".",
                    transform=ccrs.PlateCarree(),
                    **kwargs,
                    markeredgecolor="black",
                )


from collections.abc import Iterable



# comment this back in to support polygon stuff 
#import alphashape as ap
#from descartes import PolygonPatch

def plot_alpha_set(pts, ax=None):
    if ax is None:
        ax = plt.gca()
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    # print(len(lons),len(lats))
    # ax.plot(lons,lats,'*',color='black')
    α = 1.0
    while True:
        shape = ap.alphashape(pts, α)
        if not isinstance(shape, Iterable):
            break
        α /= 2
    ax.add_patch(PolygonPatch(shape, alpha=0.4))