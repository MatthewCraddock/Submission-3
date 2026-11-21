"""Map model for the mech bot: terrain, buildings, trees, a walkable grid, openness classes, a
time-budgeted A* planner and line-of-sight tests. Standard library only.

Everything is derived from the two files the server mounts in /arena (protocol.md section 11):
  v2/world.json   buildings/stadium/bridges (prisms), trees, bounds
  v2/terrain.i16  ground height on a 1 m grid
The slow parts (walkable grid, openness) are precomputed into mapmodel.bin by precompute.py and
checked against a signature of world.json; if the signature differs they are recomputed here.
"""
import array
import base64
import heapq
import json
import math
import os
import sys
import time
import zlib

CELL = 2.0                  # nav / building raster cell, metres (aligned to terrain vertices)
OCELL = 8.0                 # openness cell, metres
SQ2 = math.sqrt(2.0)
BODY_TOP = 9.0              # a mech is 9 m tall: obstacles that start above this don't block walking
STEP_UP = 1.2               # ledges up to this are walked over
MAX_SLOPE = math.tan(math.radians(50.0))


class MapModel:
    def __init__(self, world_path, terrain_path, pre_path=None, log=None):
        self.log = log or (lambda *a: None)
        with open(world_path, "rb") as f:
            raw = f.read()
        self.sig = "%08x-%d" % (zlib.crc32(raw) & 0xFFFFFFFF, len(raw))
        w = json.loads(raw)
        T = w["terrain"]
        self.tx0, self.ty0 = T["minX"], T["minY"]
        self.tn, self.tm = T["nx"], T["ny"]
        a = array.array("h")
        with open(terrain_path, "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder == "big":
            a.byteswap()
        if len(a) != self.tn * self.tm:
            raise ValueError("terrain size mismatch")
        self.h = a
        b = w["bounds"]
        self.bmin = (b["minX"], b["minY"])
        self.bmax = (b["maxX"], b["maxY"])
        self.nx, self.ny = self.tn // 2, self.tm // 2
        self.cells = {}
        for p in w["prisms"]:
            try:
                self._fill(p)
            except Exception as e:  # a bad prism must not lose the map
                self.log("prism skipped:", repr(e))
        self._trees(w.get("trees") or {})
        pre = self._load_pre(pre_path)
        if pre:
            self.nav = pre["nav"]
            self.open_med, self.open_p25 = pre["med"], pre["p25"]
            self.onx, self.ony = pre["onx"], pre["ony"]
        else:
            self.log("no matching precomputed model: building walkable grid")
            self.nav = self._build_nav()
            self.open_med = self.open_p25 = None
            self.onx = self.ony = 0

    # ------------------------------------------------------------------ loading
    def _load_pre(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("sig") != self.sig or d.get("nx") != self.nx or d.get("ny") != self.ny:
                return None
            nav = bytearray(zlib.decompress(base64.b64decode(d["nav"])))
            med = bytes(zlib.decompress(base64.b64decode(d["med"])))
            p25 = bytes(zlib.decompress(base64.b64decode(d["p25"])))
            if len(nav) != self.nx * self.ny or len(med) != d["onx"] * d["ony"]:
                return None
            return {"nav": nav, "med": med, "p25": p25, "onx": d["onx"], "ony": d["ony"]}
        except Exception as e:
            self.log("precomputed model unusable:", repr(e))
            return None

    # ------------------------------------------------------------------ buildings
    def _fill(self, p):
        base, top = p["base"], p["top"]
        edges = []
        for poly in [p["ring"]] + list(p.get("holes") or []):
            pts = list(zip(poly[0::2], poly[1::2]))
            for k in range(len(pts)):
                x1, y1 = pts[k]
                x2, y2 = pts[(k + 1) % len(pts)]
                if y1 != y2:
                    edges.append((x1, y1, x2, y2) if y1 < y2 else (x2, y2, x1, y1))
        if not edges:
            return
        ymin = min(e[1] for e in edges)
        ymax = max(e[3] for e in edges)
        nx, ny = self.nx, self.ny
        j0 = int(math.ceil((ymin - self.ty0) / CELL - 0.5))
        j1 = int(math.floor((ymax - self.ty0) / CELL - 0.5))
        for j in range(max(j0, 0), min(j1, ny - 1) + 1):
            y = self.ty0 + (j + 0.5) * CELL
            xs = sorted(e[0] + (y - e[1]) * (e[2] - e[0]) / (e[3] - e[1]) for e in edges if e[1] <= y < e[3])
            for k in range(0, len(xs) - 1, 2):
                i0 = int(math.ceil((xs[k] - self.tx0) / CELL - 0.5))
                i1 = int(math.floor((xs[k + 1] - self.tx0) / CELL - 0.5))
                for i in range(max(i0, 0), min(i1, nx - 1) + 1):
                    self.cells.setdefault(j * nx + i, []).append((base, top))

    def solid(self, x, y, z):
        """Is (x, y, z) inside a building / stadium / bridge solid?"""
        i = int((x - self.tx0) // CELL)
        j = int((y - self.ty0) // CELL)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return False
        lst = self.cells.get(j * self.nx + i)
        if lst:
            for lo, hi in lst:
                if lo <= z <= hi:
                    return True
        return False

    # ------------------------------------------------------------------ terrain
    def ground(self, x, y):
        fx = x - self.tx0
        fy = y - self.ty0
        if fx < 0.0:
            fx = 0.0
        elif fx > self.tn - 1.001:
            fx = self.tn - 1.001
        if fy < 0.0:
            fy = 0.0
        elif fy > self.tm - 1.001:
            fy = self.tm - 1.001
        i = int(fx)
        j = int(fy)
        u = fx - i
        v = fy - j
        h = self.h
        n = self.tn
        k = j * n + i
        return 0.01 * ((h[k] * (1 - u) + h[k + 1] * u) * (1 - v) + (h[k + n] * (1 - u) + h[k + n + 1] * u) * v)

    # ------------------------------------------------------------------ trees (vision only)
    def _trees(self, tr):
        self.tb = {}
        f = tr.get("fields")
        d = tr.get("data")
        if not f or not d:
            return
        k = len(f)
        ix = {n: f.index(n) for n in ("x", "y", "z", "height", "crownR", "crownBase") if n in f}
        if len(ix) < 6:
            return
        cx = f.index("crownX") if "crownX" in f else ix["x"]
        cy = f.index("crownY") if "crownY" in f else ix["y"]
        for t in range(len(d) // k):
            r = d[t * k:(t + 1) * k]
            z0 = r[ix["z"]]
            crown = (r[cx], r[cy], r[ix["crownR"]], z0 + r[ix["crownBase"]], z0 + r[ix["height"]])
            key = (int(crown[0] // 16), int(crown[1] // 16))
            self.tb.setdefault(key, []).append(crown)

    def canopy_len(self, a, b):
        """Approximate metres of dense crown along a->b (crowns modelled as upright cylinders).
        The server blocks sight when this gets past about 4 m (canopyDensity 0.35, blockT 0.25)."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = max(2, int(math.hypot(dx, dy) / 2.0))
        seg = math.sqrt(dx * dx + dy * dy + dz * dz) / n
        inside = 0
        tb = self.tb
        for k in range(1, n):
            f = k / n
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            for t in tb.get((int(x // 16), int(y // 16)), ()):
                if t[3] <= z <= t[4] and (x - t[0]) ** 2 + (y - t[1]) ** 2 <= t[2] * t[2]:
                    inside += 1
                    break
        return inside * seg

    # ------------------------------------------------------------------ line of sight
    def los(self, a, b, trees=False):
        """Clear line a->b against buildings and ground (shots ignore trees; sight does not)."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        n = int(math.hypot(dx, dy) / 1.5)
        solid = self.solid
        ground = self.ground
        for k in range(1, n):
            f = k / n
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            if solid(x, y, z):
                return False
            if k % 3 == 0 and z < ground(x, y) + 0.3:
                return False
        if trees and self.tb and self.canopy_len(a, b) > 4.0:
            return False
        return True

    # ------------------------------------------------------------------ walkable grid
    def _build_nav(self):
        nx, ny, tn, h = self.nx, self.ny, self.tn, self.h
        blocked = bytearray(nx * ny)
        for J in range(ny):
            j = 2 * J + 1
            if j + 1 >= self.tm:
                for I in range(nx):
                    blocked[J * nx + I] = 1
                continue
            row = j * tn
            for I in range(nx):
                i = 2 * I + 1
                x = self.tx0 + i
                y = self.ty0 + j
                if i + 1 >= tn or x < self.bmin[0] or x > self.bmax[0] or y < self.bmin[1] or y > self.bmax[1]:
                    blocked[J * nx + I] = 1
                    continue
                dzx = (h[row + i + 1] - h[row + i - 1]) * 0.005
                dzy = (h[row + tn + i] - h[row - tn + i]) * 0.005
                if dzx * dzx + dzy * dzy > MAX_SLOPE * MAX_SLOPE:
                    blocked[J * nx + I] = 1
        for idx, lst in self.cells.items():
            I, J = idx % nx, idx // nx
            gz = h[(2 * J + 1) * tn + 2 * I + 1] * 0.01
            for lo, hi in lst:
                if lo < gz + BODY_TOP and hi > gz + STEP_UP:
                    blocked[idx] = 1
                    break
        out = bytearray(blocked)  # dilate by one cell (the mech is 2.5 m in radius)
        for idx in range(nx * ny):
            if blocked[idx]:
                I, J = idx % nx, idx // nx
                for dj in (-1, 0, 1):
                    jj = J + dj
                    if 0 <= jj < ny:
                        for di in (-1, 0, 1):
                            ii = I + di
                            if 0 <= ii < nx:
                                out[jj * nx + ii] = 1
        return out

    def cell(self, x, y):
        return int((x - self.tx0) // CELL), int((y - self.ty0) // CELL)

    def center(self, idx):
        I, J = idx % self.nx, idx // self.nx
        return self.tx0 + (I + 0.5) * CELL, self.ty0 + (J + 0.5) * CELL

    def free(self, x, y):
        I = int((x - self.tx0) // CELL)
        J = int((y - self.ty0) // CELL)
        if I < 0 or J < 0 or I >= self.nx or J >= self.ny:
            return False
        return self.nav[J * self.nx + I] == 0

    def line_free(self, ax, ay, bx, by, half=1.0):
        """True when a body ~2*half m wide can walk the straight line (samples every 0.5 m, plus both sides)."""
        L = math.hypot(bx - ax, by - ay)
        n = int(L * 2.0) + 1
        nav, nx, ny, tx0, ty0 = self.nav, self.nx, self.ny, self.tx0, self.ty0
        if L < 1e-6:
            return True
        px, py = -(by - ay) / L * half, (bx - ax) / L * half
        for k in range(1, n + 1):
            f = k / n
            cx = ax + (bx - ax) * f
            cy = ay + (by - ay) * f
            for sx, sy in ((0.0, 0.0), (px, py), (-px, -py)):
                I = int((cx + sx - tx0) // CELL)
                J = int((cy + sy - ty0) // CELL)
                if I < 0 or J < 0 or I >= nx or J >= ny or nav[J * nx + I]:
                    return False
        return True

    def near_free(self, x, y, r=6):
        """Index of the free cell nearest (x, y) within r cells, or None."""
        I0, J0 = self.cell(x, y)
        nx, ny, nav = self.nx, self.ny, self.nav
        best, bd = None, 1e9
        for dj in range(-r, r + 1):
            J = J0 + dj
            if J < 0 or J >= ny:
                continue
            for di in range(-r, r + 1):
                I = I0 + di
                if 0 <= I < nx and nav[J * nx + I] == 0:
                    d = di * di + dj * dj
                    if d < bd:
                        best, bd = J * nx + I, d
        return best

    # ------------------------------------------------------------------ openness
    def openness(self, x, y):
        """Median free distance (m) to a building in all directions around (x, y); 60 if unknown."""
        if not self.open_med:
            return 60.0
        I = int((x - self.tx0) // OCELL)
        J = int((y - self.ty0) // OCELL)
        if I < 0 or J < 0 or I >= self.onx or J >= self.ony:
            return 60.0
        v = self.open_med[J * self.onx + I]
        return float(v) if v else 60.0

    def region(self, x, y):
        o = self.openness(x, y)
        return "tight" if o < 40.0 else ("mid" if o < 100.0 else "open")


class PathJob:
    """Weighted A* on the 2 m grid that works in slices: step(deadline) until done."""

    def __init__(self, m, start, goal, costfn=None, weight=1.8, max_exp=25000):
        self.m = m
        self.done = False
        self.path = None
        self.partial = False
        s = m.near_free(start[0], start[1], 4)
        g = m.near_free(goal[0], goal[1], 8)
        self.costfn = costfn
        self.weight = weight
        self.max_exp = max_exp
        self.exp = 0
        if s is None or g is None:
            self.done = True
            return
        self.s, self.g = s, g
        nx = m.nx
        self.gx, self.gy = g % nx, g // nx
        self.gs = {s: 0.0}
        self.came = {}
        self.openq = [(self._h(s), 0.0, s)]
        self.best, self.best_h = s, self._h(s)
        self.closed = set()

    def _h(self, idx):
        nx = self.m.nx
        dx = abs(idx % nx - self.gx)
        dy = abs(idx // nx - self.gy)
        return (dx + dy + (SQ2 - 2.0) * min(dx, dy)) * CELL

    def step(self, deadline):
        if self.done:
            return True
        m = self.m
        nav, nx, ny = m.nav, m.nx, m.ny
        gs, came, openq, closed = self.gs, self.came, self.openq, self.closed
        w = self.weight
        cf = self.costfn
        goal = self.g
        pop, push = heapq.heappop, heapq.heappush
        perf = time.perf_counter
        n = 0
        while openq:
            n += 1
            if n & 15 == 0 and perf() > deadline:
                return False
            f, g0, cur = pop(openq)
            if cur in closed:
                continue
            closed.add(cur)
            self.exp += 1
            if cur == goal:
                self._finish(cur)
                return True
            hcur = f - g0 * 0.0  # (kept simple; best-node tracking below uses h)
            I, J = cur % nx, cur // nx
            hh = self._h(cur)
            if hh < self.best_h:
                self.best, self.best_h = cur, hh
            if self.exp >= self.max_exp:
                self.partial = True
                self._finish(self.best)
                return True
            for di, dj, c in ((1, 0, CELL), (-1, 0, CELL), (0, 1, CELL), (0, -1, CELL),
                              (1, 1, CELL * SQ2), (1, -1, CELL * SQ2), (-1, 1, CELL * SQ2), (-1, -1, CELL * SQ2)):
                ii, jj = I + di, J + dj
                if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                    continue
                nb = jj * nx + ii
                if nav[nb] or nb in closed:
                    continue
                if di and dj and (nav[J * nx + ii] or nav[jj * nx + I]):
                    continue  # no corner cutting
                if cf is not None:
                    mult = cf(nb)
                    if mult is None:
                        continue
                    c = c * mult
                ng = g0 + c
                if ng < gs.get(nb, 1e18):
                    gs[nb] = ng
                    came[nb] = cur
                    push(openq, (ng + w * self._h(nb), ng, nb))
        self._finish(self.best)
        self.partial = True
        return True

    def _finish(self, end):
        seq = [end]
        came = self.came
        while seq[-1] in came:
            seq.append(came[seq[-1]])
        seq.reverse()
        self.path = [self.m.center(i) for i in seq]
        self.done = True


def follow(m, path, pi, x, y, reach=3.0, look=26.0):
    """Pick a steering point on `path`. Returns (new_index, (tx, ty) or None when finished)."""
    n = len(path)
    if n == 0:
        return pi, None
    # advance to the closest upcoming point
    best, bd = pi, 1e18
    for k in range(pi, min(n, pi + 14)):
        d = (path[k][0] - x) ** 2 + (path[k][1] - y) ** 2
        if d < bd:
            best, bd = k, d
    pi = best
    while pi < n - 1 and (path[pi][0] - x) ** 2 + (path[pi][1] - y) ** 2 < reach * reach:
        pi += 1
    if pi >= n - 1 and (path[-1][0] - x) ** 2 + (path[-1][1] - y) ** 2 < reach * reach:
        return pi, None
    # farthest point within `look` metres that the straight line can reach
    far = min(n - 1, pi + int(look / CELL))
    for k in range(far, pi - 1, -1):
        px, py = path[k]
        if k == pi or m.line_free(x, y, px, py):
            return pi, (px, py)
    return pi, path[pi]
