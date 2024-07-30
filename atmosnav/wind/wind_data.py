import os
import jax
import math
import numpy as np
from jax import lax
import jax.numpy as jnp
from dataclasses import dataclass
import json
from line_profiler import profile
from .wind import Wind
from ..utils import *

@dataclass
class WindCfg:
    legacy: bool=False
    num_levels: int = 0
    alt_min: float = 0.0
    alt_d: float = 0.0
    num_lats: int = 0
    lat_min: float = 0.0
    lat_d: float = 0.0
    num_lons: int = 0
    lon_min: float = 0.0
    lon_d: float = 0.0

class WindFromData(Wind):
    
    def _load_data(dname):
        """
        Load the data given a start time and config file name.

        Args:
            dname (string): The name of the wind config file

        Returns:
            wind_ts (jnp.Array): The list of wind times
            wind_data (jnp.Array): The wind data stored in the form of a memory map
            wind_legacy_levels (jnp.Array): The list of wind levels
            wind_cfg (WindCfg object): A WindCfg instance
        """

        assert os.path.exists(dname)
        assert os.path.exists(os.path.join(dname, "config.json")), "No config json found in data directory"

        with open(os.path.join(dname, "config.json")) as f:
            cfg = json.load(f)
        
        wind_legacy_levels = jnp.asarray(json.loads(cfg["LEVELS"]), dtype=jnp.int32)
        wcfg = WindCfg()
        wcfg.legacy = 1
        wcfg.num_levels = cfg["NUM_LEVELS"]
        wcfg.num_lats = cfg["NUM_LATS"]
        wcfg.lat_min = cfg["LAT_MIN"]
        wcfg.lat_d = cfg["LAT_D"]
        wcfg.num_lons = cfg["NUM_LONS"]
        wcfg.lon_min = cfg["LON_MIN"]
        wcfg.lon_d = cfg["LON_D"]
        wind_cfg = wcfg

        wind_ts = jnp.load(os.path.join(dname, "wind_ts.npy"), mmap_mode='r')
        wind_data = jnp.load(os.path.join(dname, "wind_data.npy"), mmap_mode = 'r')
        
        return wind_ts, wind_data, wind_legacy_levels, wind_cfg


    def from_data(data_path, integration_time_step):
        """
        Construct the Wind instance from the data path, start time, and the integration time step.
        """
        wind_ts, wind_data, wind_legacy_levels, wind_cfg = WindFromData._load_data(data_path)
        return WindFromData(wind_data, wind_ts, wind_legacy_levels, wind_cfg, integration_time_step / (2.0 * math.pi * 6371008.0 / 360.0) )

    def __init__(self, wind_data, wind_ts, wind_legacy_levels, wind_cfg, idlat):
        self.wind_data = wind_data
        self.wind_ts = wind_ts
        self.wind_legacy_levels = wind_legacy_levels
        self.wind_cfg = wind_cfg
        self.idlat = idlat
    
    @profile
    def get_wind(self, f: int, pt, lev: int) -> int:
        idx = (self.wind_cfg.num_levels * 2 * (self.wind_cfg.num_lons * pt[0] + (pt[1] % self.wind_cfg.num_lons)) + 2 * lev)
        return 0.01 * jax.lax.dynamic_slice(self.wind_data, (f, idx), (1, 2))[0]
    
    def get_base_neighbor(self, lat: float, lon: float):
        assert -10.0 <= self.wind_cfg.lat_d <= 10.0, f"lat_d out of range: {self.wind_cfg.lat_d}"
        assert -360.0 <= self.wind_cfg.lat_min <= 360.0, f"lat_min out of range: {self.wind_cfg.lat_min}"
        lat0 = ((lat - self.wind_cfg.lat_min) // self.wind_cfg.lat_d).astype(int)
        lon0 = ((lon - self.wind_cfg.lon_min) // self.wind_cfg.lon_d).astype(int)
        return lat0, lon0
    
    @profile
    def get_index(self, t: int) -> int:
        assert len(self.wind_ts) != 0, "wind_ts should not be length 0, data is not loaded properly"
        return jnp.searchsorted(self.wind_ts, t)

    @profile
    def get_level(self, altitude: float) -> int:
        p = alt2p(altitude)
        return jnp.searchsorted(self.wind_legacy_levels, p)
    
    @profile
    def get_direction(self, time, state):
        file_idx = self.get_index(time)
        file_idx = lax.cond(file_idx == 0, lambda _: 1, lambda idx: idx, operand=file_idx)
        file_idx -= 1
        
        tl = self.wind_ts[file_idx]
        th = self.wind_ts[file_idx + 1]
       
        h = jnp.clip(state[2], 0, 22)

        level_idx = self.get_level(h)
        
        level_idx = jax.lax.cond(
            level_idx == self.wind_cfg.num_levels,
            lambda _: self.wind_cfg.num_levels - 1,
            lambda _: level_idx,
            operand=None
        )
        
        h = jax.lax.cond(
            level_idx == self.wind_cfg.num_levels - 1,
            lambda idx: 1.0 * p2alt(self.wind_legacy_levels[idx]),
            lambda _: 1.0 * h,
            operand=level_idx
        )

        hh = p2alt(self.wind_legacy_levels[level_idx - 1])
        hl = p2alt(self.wind_legacy_levels[level_idx])
        sgn = -1


        vlat = state[0]
        vlon = state[1]
        nlon = 0

        def adjust_longitude(val):
            vlon, nlon = val
            dn = jnp.where(vlon < 0, 1, -1)
            vlon = vlon + 360 * dn
            nlon = nlon + dn
            
            return vlon, nlon
        
        vlon, nlon = jax.lax.while_loop(lambda v: (v[0] < 0) | (v[0] >= 360), adjust_longitude, (vlon, nlon))

        ilat, ilon = self.get_base_neighbor(vlat, vlon)
        
        θ_lat = (state[0] - (self.wind_cfg.lat_min + self.wind_cfg.lat_d * ilat)) / self.wind_cfg.lat_d
        θ_lon = (state[1] - (self.wind_cfg.lon_min + self.wind_cfg.lon_d * ilon) + (360 * nlon)) / self.wind_cfg.lon_d
        θ_t = (time - tl) / (th - tl)
        θ_h = (h - hl) / (hh - hl)
        
        v0 = jnp.array([[[[self.get_wind(file_idx + 0, (ilat + 0, ilon + 0), level_idx + sgn * 0), self.get_wind(file_idx + 0, (ilat + 0, ilon + 0), level_idx + sgn * 1)], [self.get_wind(file_idx + 0, (ilat + 0, ilon + 1), level_idx + sgn * 0), self.get_wind(file_idx + 0, (ilat + 0, ilon + 1), level_idx + sgn * 1)]], [[self.get_wind(file_idx + 0, (ilat + 1, ilon + 0), level_idx + sgn * 0), self.get_wind(file_idx + 0, (ilat + 1, ilon + 0), level_idx + sgn * 1)], [self.get_wind(file_idx + 0, (ilat + 1, ilon + 1), level_idx + sgn * 0), self.get_wind(file_idx + 0, (ilat + 1, ilon + 1), level_idx + sgn * 1)]]], [[[self.get_wind(file_idx + 1, (ilat + 0, ilon + 0), level_idx + sgn * 0), self.get_wind(file_idx + 1, (ilat + 0, ilon + 0), level_idx + sgn * 1)], [self.get_wind(file_idx + 1, (ilat + 0, ilon + 1), level_idx + sgn * 0), self.get_wind(file_idx + 1, (ilat + 0, ilon + 1), level_idx + sgn * 1)]], [[self.get_wind(file_idx + 1, (ilat + 1, ilon + 0), level_idx + sgn * 0), self.get_wind(file_idx + 1, (ilat + 1, ilon + 0), level_idx + sgn * 1)], [self.get_wind(file_idx + 1, (ilat + 1, ilon + 1), level_idx + sgn * 0), self.get_wind(file_idx + 1, (ilat + 1, ilon + 1), level_idx + sgn * 1)]]]])
        v1 = v0[:, :, :, 0] * (1 - θ_h) + v0[:, :, :, 1] * θ_h
        v2 = v1[:, :, 0] * (1 - θ_lon) + v1[:, :, 1] * θ_lon
        v3 = v2[:, 0] * (1 - θ_lat) + v2[:, 1] * θ_lat
        v4 = v3[0] * (1 - θ_t) + v3[1] * θ_t
        u, v = v4
    
        dv = v * self.idlat
        du = u * (self.idlat / jnp.cos((state[0] + dv * 0.5) * (math.pi / 180.0)))
        du = jnp.min(jnp.array([jnp.max(jnp.array([du, -120]), axis=None), 120]), axis=None)

        return dv, du
    
    def tree_flatten(self):
        children = (self.wind_data, self.wind_ts, self.wind_legacy_levels)  # arrays / dynamic values
        aux_data = {"wind_cfg":self.wind_cfg, "idlat": self.idlat}  # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
