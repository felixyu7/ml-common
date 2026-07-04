"""Detector-volume geometry for physics-informed labels.

Replicates the containment surface that ``I3EventLabeler`` (icecube.ml_suite) uses to
define IceCube ``morphology`` and the ``vertex_*`` (detector-entry) truth:
``I3Surfaces::ExtrudedPolygon(gcd, sig_padding=50 m)`` -- the 2-D convex hull of the
string XY positions, extruded over the DOM z-range, inflated outward by ``padding_m``.

The stored ``vertex_*`` is the trajectory's first intersection with that surface (the
interaction point when contained; a point on the +50 m surface for through-going tracks),
so a signed distance to *this* surface is what matches the labels -- NOT a raw 3-D convex
hull of the DOMs (that is the Prometheus-path volume in the converter's core/geometry.py
and does not reproduce the label geometry).

Frame note: ``resources/icecube.geo`` is in raw IceCube coordinates (z-center ~ -1942 m);
the mmap ``vertex_*`` is detector-centered (z ~ +-500 m). ``from_geo_file`` applies
``z += z_shift_m`` (default 1942) to bring the geofile into the vertex frame. Validated:
cascade/starting vertices sit ~137 m inside, through-going entry points at margin ~= 0.
"""

import numpy as np

# Muon energy loss in ice, dE/dx = a + b*E (a: ionization, b: radiative), scaled to ice
# density (~0.917). CSDA range = (1/b) * ln(1 + (b/a) * E). Constants are physical, not
# fitted; used only to estimate an in-detector track length for the cascade<->track ramp.
_MU_A_GEV_PER_M = 0.238      # ~0.259 GeV/mwe * 0.917
_MU_B_PER_M = 3.33e-4        # ~0.363e-3 /mwe * 0.917


def muon_range_m(energy_gev):
    """Approximate muon CSDA range in ice [m] for energy [GeV]. Vectorized; 0 at E<=0."""
    e = np.maximum(np.asarray(energy_gev, dtype=np.float64), 0.0)
    return np.log1p((_MU_B_PER_M / _MU_A_GEV_PER_M) * e) / _MU_B_PER_M


class ExtrudedPolygon:
    """Padded extruded-polygon detector volume; ``margin(points)`` is signed distance to
    the padded surface in meters (positive inside), matching the labeler's surface."""

    def __init__(self, edge_normals, edge_offsets, z_min, z_max, padding_m):
        # edge_normals: [E,2] outward unit normals; edge_offsets: [E] with n.x = offset on
        # the edge, so a point is inside the (unpadded) polygon iff n.x <= offset for all edges.
        self.edge_normals = np.asarray(edge_normals, dtype=np.float64)
        self.edge_offsets = np.asarray(edge_offsets, dtype=np.float64)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.padding_m = float(padding_m)

    @classmethod
    def from_geo_file(cls, path, z_shift_m=1942.0, padding_m=50.0):
        from scipy.spatial import ConvexHull
        pts = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue  # header / metadata lines
                pts.append((x, y, z))
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] < 3:
            raise ValueError(f"geo file {path!r} yielded {pts.shape[0]} points; need >= 3")
        pts[:, 2] += z_shift_m

        hull = ConvexHull(pts[:, :2])
        verts = pts[hull.vertices, :2]           # polygon vertices (hull order)
        centroid = verts.mean(axis=0)
        normals = np.zeros((len(verts), 2))
        offsets = np.zeros(len(verts))
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            edge = b - a
            nrm = np.array([edge[1], -edge[0]], dtype=np.float64)
            nrm /= np.linalg.norm(nrm)
            if nrm.dot(centroid - a) > 0:         # force outward
                nrm = -nrm
            normals[i] = nrm
            offsets[i] = nrm.dot(a)
        return cls(normals, offsets, pts[:, 2].min(), pts[:, 2].max(), padding_m)

    def margin(self, points):
        """Signed distance [m] to the padded surface, positive inside. points: [...,3]."""
        p = np.asarray(points, dtype=np.float64)
        d = p[..., :2] @ self.edge_normals.T - self.edge_offsets   # [...,E]; + outside edge
        s_xy = -np.max(d, axis=-1)                                 # + inside laterally
        s_z = np.minimum(p[..., 2] - self.z_min, self.z_max - p[..., 2])
        return np.minimum(s_xy, s_z) + self.padding_m
