import os
import jax
import math
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
import json

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
    
    def _load_data(dname, start_time):
        """
        Load the data given a start time and config file name.

        Args:
            dname (string): The name of the wind config file
            start_time (jnp.float32): The start time of the simulation

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

        t1, t2 = int(start_time), int(2e9) # min and max times
        ls = [int(x.split(".")[0]) for x in os.listdir(dname) if x.endswith(".bin")]
        fs = sorted([x for x in ls if t1 <= x <= t2])

        print('Start loading wind data')
        WIND_DATA_LENGTH = 16115040
        wind_data = np.zeros((len(fs), WIND_DATA_LENGTH), dtype=np.int16)
        wind_ts = np.zeros((len(fs), ), dtype=np.int32)
        for i, t in enumerate(fs):
            path = os.path.join(dname, str(t) + ".bin")
            mp = np.memmap(path, dtype=np.int16, mode="r", offset=4)
            wind_ts[i] = t
            wind_data[i] =mp#

        print('Done loading wind data')
        return jnp.asarray(wind_ts), jnp.asarray(wind_data), wind_legacy_levels, wind_cfg

    def from_data(path, start_time, integration_time_step):
        """
        Construct the Wind instance from the data path, start time, and the integration time step.
        """
        wind_ts, wind_data, wind_legacy_levels, wind_cfg = WindFromData._load_data(path, start_time)
        return WindFromData(wind_data, wind_ts, wind_legacy_levels, wind_cfg, integration_time_step / (2.0 * math.pi * 6371008.0 / 360.0) )

    def __init__(self, wind_data, wind_ts, wind_legacy_levels, wind_cfg, idlat):
        self.wind_data = wind_data
        self.wind_ts = wind_ts
        self.wind_legacy_levels = wind_legacy_levels
        self.wind_cfg = wind_cfg
        self.idlat = idlat
    
    def get_wind(self, f: int, pt, lev: int) -> int:
        
        def adjust_pt(pt, num_lons):
            def cond_fun(state):
                pt, _ = state
                return (pt[1] >= num_lons) | (pt[1] < 0)

            def body_fun(state):
                pt, _ = state
                pt = jax.lax.cond(pt[1] >= num_lons,
                                lambda x: (x[0], x[1] - num_lons),
                                lambda x: x,
                                pt)
                pt = jax.lax.cond(pt[1] < 0,
                                lambda x: (x[0], x[1] + num_lons),
                                lambda x: x,
                                pt)
                return (pt, None)

            state = (pt, None)
            state = jax.lax.while_loop(cond_fun, body_fun, state)
            pt, _ = state
            return pt

        pt = adjust_pt(pt, self.wind_cfg.num_lons)

        idx = (self.wind_cfg.num_levels * 2 * (self.wind_cfg.num_lons * pt[0] + pt[1]) + 2 * lev)
        return jnp.array([0.01* self.wind_data[f][idx], 0.01 * self.wind_data[f][idx + 1]])
    
    def get_base_neighbor(self, lat: float, lon: float):
        assert -10.0 <= self.wind_cfg.lat_d <= 10.0, f"lat_d out of range: {self.wind_cfg.lat_d}"
        assert -360.0 <= self.wind_cfg.lat_min <= 360.0, f"lat_min out of range: {self.wind_cfg.lat_min}"
        lat0 = ((lat - self.wind_cfg.lat_min) // self.wind_cfg.lat_d).astype(int)
        lon0 = ((lon - self.wind_cfg.lon_min) // self.wind_cfg.lon_d).astype(int)
        return lat0, lon0

    def get_index(self, t: int) -> int:
        assert len(self.wind_ts) != 0, "wind_ts should not be length 0, data is not loaded properly"
        return bisect_left_jax(self.wind_ts, t)

    def get_level(self, altitude: float) -> int:
        if self.wind_cfg.legacy:
            p = alt2p(altitude)
            return bisect_left_jax(self.wind_legacy_levels, p)
        else:
            return int((altitude - self.wind_cfg.alt_min) / self.wind_cfg.alt_d)

    def get_direction(self, time, state):
        """
        Gets the direction of the wind movement at the state and a given time.

        Args:
            time (jnp.float32): The time now since the simulation has started
            state (jnp.Array): The state vector, composed of [lat, lon, h]
        
        Returns:
            dv (jnp.float32): The direction along latitude
            du (jnp.float32): The diretcion along longitude
        """
        file_idx = self.get_index(time)
        file_idx = lax.cond(file_idx == 0, lambda _: 1, lambda idx: idx, operand=file_idx)
        file_idx -= 1
        
        tl = self.wind_ts[file_idx]
        th = self.wind_ts[file_idx + 1]
       
        h = jnp.clip(state[2], 0, 22)

        if self.wind_cfg.legacy:
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
        else:
            level_idx = self.get_level(h)
            hl = self.wind_cfg.alt_d * level_idx
            hh = self.wind_cfg.alt_d * (level_idx + 1)
            sgn = 1

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
        
        p = jnp.zeros((2, 2, 2, 2, 2))

        
        def body_for_16(i, p):
            i0 = (i >> 0) & 1
            i1 = (i >> 1) & 1
            i2 = (i >> 2) & 1
            i3 = (i >> 3) & 1
            w = self.get_wind(file_idx + i3, (ilat + i2, ilon + i1), level_idx + sgn * i0)
            return p.at[(i3,i2, i1, i0, 0)].set(w[0]).at[(i3,i2, i1, i0, 1)].set(w[1])

        p = jax.lax.fori_loop(0, 16, body_for_16, p)

        def body_for_8(i, p):
            i1 = (i >> 0) & 1
            i2 = (i >> 1) & 1
            i3 = (i >> 2) & 1
            p = p.at[(i3, i2, i1, 0, 0)].set(p[i3][i2][i1][0][0] * (1 - θ_h) + p[i3][i2][i1][1][0] * θ_h)
            return p.at[(i3, i2, i1, 0, 1)].set(p[i3][i2][i1][0][1] * (1 - θ_h) + p[i3][i2][i1][1][1] * θ_h)

        p = jax.lax.fori_loop(0, 8, body_for_8, p)
        
        def body_for_4(i, p):
            i2 = (i >> 0) & 1
            i3 = (i >> 1) & 1
            p = p.at[(i3, i2, 0, 0, 0)].set(p[i3][i2][0][0][0] * (1 - θ_lon) + p[i3][i2][1][0][0] * θ_lon)
            return p.at[(i3, i2, 0, 0, 1)].set(p[i3][i2][0][0][1] * (1 - θ_lon) + p[i3][i2][1][0][1] * θ_lon)

        p = jax.lax.fori_loop(0, 4, body_for_4, p)

        def body_for_2(i, p):
            i3 = (i >> 0) & 1
            return p.at[(i3,0,0,0,0)].set(p[i3][0][0][0][0] * (1 - θ_lat) + p[i3][1][0][0][0] * θ_lat)\
                .at[(i3,0,0,0,1)].set( p[i3][0][0][0][1] * (1 - θ_lat) + p[i3][1][0][0][1] * θ_lat)
        
        p = jax.lax.fori_loop(0, 2, body_for_2, p)

        p = p.at[(0,0,0,0,0)].set(p[0][0][0][0][0] * (1 - θ_t) + p[1][0][0][0][0] * θ_t)\
            .at[(0,0,0,0,1)].set(p[0][0][0][0][1] * (1 - θ_t) + p[1][0][0][0][1] * θ_t)
    
        u = p[0][0][0][0][0]
        v = p[0][0][0][0][1]

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

