"""Duelist v3: a bot that knows the map.

* map model (mapmodel.py): terrain, buildings, trees, walkable grid, tight/mid/open areas, A* paths
* memory: last seen enemies, weapons, cubes, crates, drones, gunfire it heard, supply drops
* weapons chosen by area type and by what it has seen opponents carry
* fights with matchup-aware distance, strafing, laser dodging and mortar-shell dodging
* heals out of sight, stays quiet while hiding, stages near supply drops and grabs them only when clear
"""
import gc
import math
import os
import random
import sys
import time
from collections import deque

import mechfight as mf
import mapmodel as mm

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- tunables: judgement calls, change ONE at a time and compare practice scores ----------------
HIDE_BELOW = 0.45        # go and heal out of sight under this fraction of max HP
HEAL_TO = 0.85           # ...and come back at this one
FARM_DRONES = True       # shoot NPC drones for their 2 cubes when nothing else is around
FARM_CRATES = True       # shoot crates (cube + 50% weapon) when nothing else is around
DODGE_BULLETS = True     # unified dodge planner: bullets, mortar shells, laser telegraphs and beams
DROP_BOOST = True         # hurry to a supply drop that would land before we arrive (only when nobody is near)
DROP_EARLY = 2.0         # start walking into the ring this many seconds before it lands
PREDODGE = True          # also dodge the shots an enemy is about to fire (their gun is pointed at us)
BACKWALL = True          # fight with a wall behind us so nothing can hit our back (rear hits do x1.5)
FACE_ATTACKER = True     # whoever just hit us becomes the preferred target, so we turn our front to them
DUEL = True              # keep fighting (instead of hiding) when we would win the duel: time-to-kill vs time-to-die
FINISH_W = 28.0          # target bonus for an enemy under 35% HP (kills end the threat and drop cubes)
SOFT_BUDGET = 0.012      # s: optional work (planning, cover search) stops after this
# weapon value by area type: tight = lightning country, open = laser/mortar country (judgement)
WEAPON_VALUE = {
    "tight": {"lightning": 3.2, "laser": 1.7, "mortar": 0.8, "gun": 1.0},
    "mid":   {"lightning": 2.5, "laser": 2.7, "mortar": 1.4, "gun": 1.0},
    "open":  {"lightning": 1.0, "laser": 3.1, "mortar": 2.1, "gun": 1.0},
}
COUNTERS = {"laser": "mortar", "lightning": "laser", "mortar": "lightning"}  # key beats value
ENGAGE_RANGE = {"gun": 140.0, "laser": 230.0, "mortar": 150.0, "lightning": 42.0}
LOUD = ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion", "npc_gun")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def norm(vx, vy):
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n > 1e-9 else (0.0, 0.0)


def pref_dist(my, en):
    """Distance we want to fight at, from what we hold (my) and what the enemy holds (en)."""
    L, Z, M = "laser", "lightning", "mortar"
    if Z in my and L in en:
        return 22.0          # lightning shorts a laser: get inside arc range
    if L in my and Z in en:
        return 85.0          # stay out of their 45 m arc
    if M in my and Z in en:
        return 80.0
    if Z in en and Z not in my:
        return 80.0
    if M in en and L not in my:
        return 30.0          # inside a mortar's comfortable range
    if Z in my:
        return 24.0
    if L in my:
        return 120.0
    if M in my:
        return 95.0
    return 85.0


class Bot:
    name = "elite_hunter_v3"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.reach = cfg["loot"]["pickup_radius"]
        self.step_speed = cfg["sound"]["step_speed"]
        self.rng = random.Random(hello.get("seed", 0))
        self.m = None
        try:
            d = hello["map"]["dir"]
            files = hello["map"].get("files", [])
            wp = next((os.path.join(d, f) for f in files if f.endswith("world.json")), os.path.join(d, "v2", "world.json"))
            tp = next((os.path.join(d, f) for f in files if f.endswith("terrain.i16")), os.path.join(d, "v2", "terrain.i16"))
            self.m = mm.MapModel(wp, tp, os.path.join(HERE, "mapmodel.bin"), log)
        except Exception as e:
            log("map not loaded, playing without it:", repr(e))
        gc.collect()
        gc.freeze()
        # two modes: "squad" (practice: the other mechs hunt us and never fight each other) and "ffa" (they fight each other)
        self.mode_name = "squad"
        self.alive_prev = None
        self.unexplained = 0
        self.set_mode("squad")
        # memory
        self.en = {}          # id -> last known state of each enemy mech
        self.npc = {}         # id -> drone
        self.pk = {}          # id -> weapon on the ground
        self.cr = {}          # id -> crate
        self.cu = {}          # id -> cube
        self.heard = deque(maxlen=48)   # (t, x, y, kind, uncertainty)
        self.bad_take = {}
        self.dead = set()
        # state
        self.hiding = False
        self.hide_since = 0.0
        self.cover = None
        self.cover_t = -99.0
        self.ambush = None
        self.ambush_t = -99.0
        self.goal = None
        self.goal_t = -99.0
        self.last_dmg_t = -99.0
        self.last_dmg_bearing = None
        self.last_dmg_by = None
        self.target_id = None
        self.strafe = 1
        self.flip_at = 0.0
        self.anchor = None          # (x, y) a spot with a wall behind it, for the current fight
        self.anchor_t = -99.0
        self.anchor_for = None
        self.job = None
        self.path = None
        self.pi = 0
        self.path_goal = None
        self.path_t = -99.0
        self.hist = deque(maxlen=24)
        self.tblock = {}
        self.side_until = 0.0
        self.side_sign = 1
        self.last_dt = 0.0
        self.last_move = (0.0, 0.0)
        self.last_aim = None
        self.t = 0.0

    # ------------------------------------------------------------------ map helpers
    def los(self, a, b, trees=False):
        return True if self.m is None else self.m.los(a, b, trees)

    def region(self, x, y):
        return "mid" if self.m is None else self.m.region(x, y)

    def steer(self, x, y, dx, dy):
        """Nearest direction to (dx, dy) that doesn't run into a wall."""
        if self.m is None or (dx == 0.0 and dy == 0.0):
            return dx, dy
        base = math.atan2(dy, dx)
        m = self.m
        for off in (0.0, 0.45, -0.45, 0.9, -0.9, 1.4, -1.4, 1.9, -1.9, 2.5, -2.5, math.pi):
            c, s = math.cos(base + off), math.sin(base + off)
            if m.free(x + c * 3.0, y + s * 3.0) and m.free(x + c * 7.0, y + s * 7.0):
                return c, s
        return 0.0, 0.0

    # ------------------------------------------------------------------ memory
    def update(self, obs):
        t = self.t
        me = obs["self"]
        x, y, z = me["pos"]
        st = me.get("stats", {})
        eye = (x, y, z + st.get("eye_height", 8.0))
        for e in obs["events"]:
            ty = e["type"]
            if ty == "damaged" and e.get("weapon") != "zone":
                self.last_dmg_t = t
                self.last_dmg_by = e.get("by")
                if e.get("bearing") is not None:
                    self.last_dmg_bearing = e["bearing"]
            elif ty == "take_failed":
                self.bad_take[e["id"]] = t + 15.0
            elif ty == "took":
                self.pk.pop(e["id"], None)
            elif ty == "kill":
                self.en.pop(e["target"], None)
                self.dead.add(e["target"])
        for m in obs["mechs"]:
            p = m["pos"]
            if m["kind"] == "mech":
                if m["id"] in self.dead:
                    continue
                e = self.en.setdefault(m["id"], {"first": t, "id": m["id"]})
                sc = 1.0 + 0.05 * (m.get("cubes") or 0)
                vis = self.los(eye, (p[0], p[1], p[2] + 4.5 * sc)) or self.los(eye, (p[0], p[1], p[2] + 9.0 * sc))
                e.update(pos=p, vel=m["vel"], aim=m["aim"], hp=m.get("hp"), maxhp=m.get("max_hp") or 1400.0,
                         cubes=m.get("cubes") or 0, weapons=list(m.get("weapons") or []),
                         charging=bool(m.get("charging")), t=t, vis=vis)
                h = e.setdefault("h", deque(maxlen=10))
                h.append((t, m["vel"][0], m["vel"][1]))
            else:
                self.npc[m["id"]] = {"pos": p, "vel": m["vel"], "hp": m.get("hp"), "t": t, "weapons": []}
        # a gone enemy stays remembered where we last saw it
        listed = {m["id"] for m in obs["mechs"]}
        for e in self.en.values():
            if e["id"] not in listed:
                e["vis"] = False
        self._forget(self.pk, {p["id"] for p in obs["pickups"]}, eye, 1.0)
        for p in obs["pickups"]:
            self.pk[p["id"]] = {"kind": p["kind"], "pos": p["pos"], "t": t}
        self._forget(self.cr, {c["id"] for c in obs["crates"]}, eye, 1.5)
        for c in obs["crates"]:
            self.cr[c["id"]] = {"pos": c["pos"], "hp": c.get("hp"), "t": t}
        self._forget(self.cu, {c["id"] for c in obs["cubes"]}, eye, 1.0)
        for c in obs["cubes"]:
            self.cu[c["id"]] = {"pos": c["pos"], "t": t}
        for k in [k for k, v in self.npc.items() if t - v["t"] > 60 or (v["t"] != t and math.hypot(v["pos"][0] - x, v["pos"][1] - y) < 30)]:
            self.npc.pop(k, None)
        for s in obs["sounds"]:
            r = 35.0 if s["dist"] == "near" else (110.0 if s["dist"] == "mid" else 220.0)
            b = s["bearing"]
            self.heard.append((t - s["age"], x + math.cos(b) * r, y + math.sin(b) * r, s["kind"], r * 0.6))

    def _forget(self, store, listed, eye, h):
        """Drop remembered items we should be able to see right now but can't (taken by someone)."""
        for k in list(store):
            if k in listed:
                continue
            p = store[k]["pos"]
            if math.hypot(p[0] - eye[0], p[1] - eye[1]) < 40.0 and self.los(eye, (p[0], p[1], p[2] + h)):
                del store[k]

    def danger_at(self, x, y, t):
        """How worried to be about a spot: recently seen enemies and gunfire near it."""
        d = 0.0
        for e in self.en.values():
            age = t - e["t"]
            if age <= 30.0:
                p = e["pos"]
                if math.hypot(p[0] - x, p[1] - y) < 60.0:
                    d += 1.0 - age / 30.0
        for (ht, hx, hy, kind, unc) in self.heard:
            if t - ht < 6.0 and kind in LOUD and math.hypot(hx - x, hy - y) < 50.0 + unc:
                d += 0.4
        return d

    # ------------------------------------------------------------------ weapons
    def weapon_value(self, kind, x, y, zc, t):
        """How good a weapon is for where we are and where the last fight will be, and for what we've seen."""
        loc = WEAPON_VALUE[self.region(x, y)][kind]
        fut = WEAPON_VALUE[self.region(zc[0], zc[1])][kind]
        v = 0.5 * loc + 0.5 * fut
        for e in self.en.values():
            if t - e["t"] < 40.0:
                for k in e.get("weapons", ()):
                    if COUNTERS.get(kind) == k:
                        v += 0.3
                    elif COUNTERS.get(k) == kind:
                        v -= 0.25
        return v

    def choose_upgrade(self, offer, kinds):
        ids = [c["id"] for c in offer["choices"]]
        order = ["plating", "laser_damage", "gun_damage", "laser_charge", "armor", "gun_rate", "nanites"]
        if "lightning" in kinds:
            order += ["lightning_efficiency", "lightning_chain"]
        if "laser" in kinds:
            order += ["laser_damage", "laser_charge"]
        order += ["gun_damage", "gun_rate", "servos", "gyros", "gun_mag", "reactor", "capacitor", "mortar_reload",
                  "lightning_range", "jets", "sensors", "gun_velocity", "mortar_splash", "mortar_velocity", "laser_pierce"]
        for want in order:
            if want in ids:
                return ids.index(want)
        return 0

    # ------------------------------------------------------------------ navigation
    def nav_dir(self, tx, ty, costfn=None, reach=3.0):
        """Direction (unit vector) to walk toward (tx, ty) around walls, or None when we are there."""
        x, y, t = self.x, self.y, self.t
        if math.hypot(tx - x, ty - y) < reach:
            return None
        if self.m is None:
            return norm(tx - x, ty - y)
        if self.path_goal is None or math.hypot(self.path_goal[0] - tx, self.path_goal[1] - ty) > 6.0 or t - self.path_t > 8.0:
            self.job = mm.PathJob(self.m, (x, y), (tx, ty), costfn)
            self.path_goal = (tx, ty)
            self.path_t = t
            self.path = None
            self.pi = 0
        if self.job is not None and not self.job.done:
            self.job.step(max(self.deadline, time.perf_counter() + 0.004))
        if self.job is not None and self.job.done:
            self.path = self.job.path
            self.pi = 0
            self.job = None
        if self.path:
            self.pi, tg = mm.follow(self.m, self.path, self.pi, x, y, reach=min(reach, 3.0))
            if tg is None:
                return None
            return norm(tg[0] - x, tg[1] - y)
        return self.steer(x, y, *norm(tx - x, ty - y))

    def cost_fn(self, zc, zr, t):
        """Path costs: stay inside the next safe circle, keep away from where enemies were seen."""
        m = self.m
        tb = self.tblock
        dang = [(e["pos"][0], e["pos"][1], 1.0 - (t - e["t"]) / 30.0) for e in self.en.values() if t - e["t"] <= 30.0]
        cache = {}

        def f(idx):
            v = cache.get(idx)
            if v is not None:
                return v if v > 0 else None
            if tb.get(idx, 0) > t:
                cache[idx] = -1.0
                return None
            cx, cy = m.center(idx)
            mult = 1.0
            if math.hypot(cx - zc[0], cy - zc[1]) > zr:
                mult += 3.0
            for ex, ey, w in dang:
                dd = math.hypot(cx - ex, cy - ey)
                if dd < 45.0:
                    mult += w * (1.0 - dd / 45.0) * 1.5
            cache[idx] = mult
            return mult
        return f

    def find_cover(self, threats, zc, zr):
        """Nearest reachable spot inside the safe circle that none of `threats` (list of (x, y, z)) can see."""
        m = self.m
        if m is None or not threats:
            return None
        x, y = self.x, self.y
        best, bs = None, 1e9
        for r in (6.0, 12.0, 20.0, 30.0, 44.0):
            for k in range(16):
                if time.perf_counter() > self.deadline + 0.006:
                    return best
                a = k * math.pi / 8 + r * 0.37
                cx, cy = x + r * math.cos(a), y + r * math.sin(a)
                if math.hypot(cx - zc[0], cy - zc[1]) > zr * 0.92 or not m.free(cx, cy) or not m.line_free(x, y, cx, cy):
                    continue
                gz = m.ground(cx, cy)
                ok = True
                for (px, py, pz) in threats:
                    ep = (px, py, pz + 8.0)
                    if m.los(ep, (cx, cy, gz + 4.5), True) or m.los(ep, (cx, cy, gz + 9.0), True):
                        ok = False
                        break
                if ok:
                    s = r - (8.0 if m.region(cx, cy) == "tight" else 0.0) + 4.0 * self.danger_at(cx, cy, self.t)
                    if s < bs:
                        best, bs = (cx, cy), s
            if best is not None:
                return best
        return best

    def find_ambush(self, zc, zr):
        """A tight spot near us inside the safe circle, away from where enemies were seen."""
        m = self.m
        x, y, t = self.x, self.y, self.t
        if m is None:
            return None
        best, bs = None, 1e9
        for _ in range(16):
            a = self.rng.uniform(0, 2 * math.pi)
            r = self.rng.uniform(8.0, 55.0)
            cx, cy = x + r * math.cos(a), y + r * math.sin(a)
            if math.hypot(cx - zc[0], cy - zc[1]) > zr * 0.8 or not m.free(cx, cy):
                continue
            s = m.openness(cx, cy) + 0.4 * r + 60.0 * self.danger_at(cx, cy, t)
            if s < bs:
                best, bs = (cx, cy), s
        return best

    # ------------------------------------------------------------------ main decision
    def act(self, obs: mf.Observation) -> mf.Action:
        t0 = time.perf_counter()
        try:
            a = self._act(obs, t0)
        except Exception as e:
            log("act error:", repr(e))
            a = {"tick": obs["tick"], "move": [self.last_move[0], self.last_move[1]]}
        self.last_dt = time.perf_counter() - t0
        return a

    def set_mode(self, name):
        """Per-mode settings. squad: stay in the fight and keep pressure. ffa: let the others thin each other out."""
        self.mode_name = name
        if name == "ffa":
            self.hide_below, self.heal_to = 0.62, 0.95      # heal earlier and longer
            self.zone_pad, self.zone_lead = 20.0, 22.0       # keep further inside the circle, rotate sooner
            self.range_scale = 0.6                            # only start fights that are close, or that find us
        else:
            self.hide_below, self.heal_to = HIDE_BELOW, HEAL_TO
            self.zone_pad, self.zone_lead = 14.0, 16.0
            self.range_scale = 1.0

    def watch_mode(self, obs, t):
        """Practice: alive only drops when we kill or the zone does. A death before the zone shrinks that we did not cause means others fight."""
        alive = obs.get("alive")
        if not isinstance(alive, (int, float)):
            return
        if self.alive_prev is not None and alive < self.alive_prev and self.mode_name == "squad" \
                and t < obs["zone"]["shrink_start"] - 1.0:
            ours = sum(1 for e in obs["events"] if e["type"] == "kill")
            self.unexplained += max(0, int(self.alive_prev - alive) - ours)
            if self.unexplained >= 2:
                self.set_mode("ffa")
        self.alive_prev = alive

    def _act(self, obs, t0):
        self.t = t = obs["t"]
        me = self.me = obs["self"]
        x, y, z = me["pos"]
        self.x, self.y, self.z = x, y, z
        st = me.get("stats", {})
        walk = max(st.get("walk_speed", 11.7), 1.0)
        muz = z + st.get("muzzle_height", 7.0)
        self.deadline = t0 + (SOFT_BUDGET if self.last_dt < 0.03 else 0.002)
        hpf = me["hp"] / me["max_hp"] if me["max_hp"] else 1.0
        slots = me["weapons"]
        kinds = [w["kind"] for w in slots if w]
        act = {"tick": obs["tick"]}
        self.update(obs)
        self.obs_zone = obs["zone"]
        self.watch_mode(obs, t)
        if obs["offer"]:
            act["pick"] = self.choose_upgrade(obs["offer"], kinds)

        # ---- the picture
        vis = sorted((e for e in self.en.values() if e["t"] == t and e["vis"]), key=lambda e: math.hypot(e["pos"][0] - x, e["pos"][1] - y))
        recent = [e for e in self.en.values() if t - e["t"] <= 6.0]
        near_recent = [e for e in self.en.values() if t - e["t"] <= 12.0 and math.hypot(e["pos"][0] - x, e["pos"][1] - y) < 90.0]
        zn = obs["zone"]
        if t >= zn["shrink_start"] - self.zone_lead:
            zc, zr = zn["next_center"], zn["next_radius"]
        else:
            zc, zr = zn["center"], zn["radius"]
        live_margin = zn["radius"] - math.hypot(x - zn["center"][0], y - zn["center"][1])
        zone_urgent = (not me["in_zone"]) or live_margin < self.zone_pad or (math.hypot(x - zc[0], y - zc[1]) > zr - 3.0 and t >= zn["shrink_start"] - 8.0)
        # a spot inside the next circle on our side of it (not the middle, where everyone converges)
        ux, uy = norm(x - zc[0], y - zc[1])
        zr_in = max(zr * 0.55, 0.0)
        zgoal = (zc[0] + ux * min(zr_in, math.hypot(x - zc[0], y - zc[1])), zc[1] + uy * min(zr_in, math.hypot(x - zc[0], y - zc[1])))
        costfn = self.cost_fn(zc, max(zr, 5.0), t) if self.m is not None else None

        # ---- stuck detector
        self.hist.append((t, x, y))
        if t < self.side_until:
            pass

        mv = (0.0, 0.0)
        speed = 1.0
        boost = False
        aim = None
        fire = [False, False]
        mode = "hold"
        quiet = False
        jump = False

        # ---- health: when to break contact and heal, when to come back
        multi = len(recent) >= 2
        finish = DUEL and hpf > 0.2 and not multi and self.duel_ok(vis, me, x, y)
        if not self.hiding and not finish and (hpf < self.hide_below or (multi and hpf < 0.7)):
            self.hiding = True
            self.hide_since = t
            self.cover = None
        if self.hiding and hpf >= self.heal_to and not vis:
            self.hiding = False
        if self.hiding and t - self.hide_since > 45.0:
            self.hiding = False

        tgt = self.pick_target(vis, x, y)
        engage = False
        if tgt is not None:
            d = math.hypot(tgt["pos"][0] - x, tgt["pos"][1] - y)
            rng = self.range_scale * max([ENGAGE_RANGE[k] for k in kinds] or [ENGAGE_RANGE["gun"]])
            engage = d <= rng or (t - self.last_dmg_t < 4.0)
        incoming = self.mortar_dodge(obs, x, y, z)

        if engage:
            r = self.engage(tgt, st, muz, slots, kinds, farm=False)
            aim, fire, mv, boost = r["aim"], r["fire"], r["move"], r["boost"]
            mode = r["mode"]
            if self.hiding and math.hypot(tgt["pos"][0] - x, tgt["pos"][1] - y) > 30.0:
                mv, boost, mode = self.go_cover(tgt, vis, zc, zr, costfn, mv, boost, hpf)
            if zone_urgent:
                zd = norm(zc[0] - x, zc[1] - y)
                mv = norm(mv[0] + zd[0] * 1.6, mv[1] + zd[1] * 1.6)
                boost = boost or (not me["in_zone"] and me["energy"] > 15)
            mv = self.steer(x, y, *mv)
        elif zone_urgent:
            mode = "zone"
            d2 = self.nav_dir(zgoal[0], zgoal[1], costfn, 4.0) or norm(zc[0] - x, zc[1] - y)
            mv = d2
            boost = (not me["in_zone"] or live_margin < 4.0) and me["energy"] > 15
        elif self.hiding:
            mv, boost, mode = self.go_cover(None, vis, zc, zr, costfn, (0.0, 0.0), False, hpf)
            quiet = mode == "hide"
        else:
            mode, mv, speed, boost, aim, fire, quiet, jump = self.roam(obs, hpf, kinds, slots, zc, zr, zgoal, costfn, near_recent, st, muz)

        dodged = False
        if DODGE_BULLETS:
            pl = self.hazard_plan(obs, x, y, z, (mv[0], mv[1], speed), me["energy"] > 25.0)
            if pl is not None:
                mv, speed, bo_, mg_ = pl
                boost = bo_ or (boost and mg_ > 0.0)
                mode, dodged = "dodge", True
                if mv[0] or mv[1]:
                    mv = mv
        incoming = (0.0, 0.0) if dodged else None

        # ---- quiet: steps are heard 40 m, boost 90 m, reloading 60 m; only matters when someone may be close
        if not engage and not zone_urgent and incoming is None:
            quiet = quiet or bool(near_recent) or mode in ("hide", "stage")
        qs = min(1.0, 0.9 * self.step_speed / walk)
        if quiet and not boost:
            speed = min(speed, qs)

        # ---- stuck: no progress while trying to move
        if (mv[0] or mv[1]) and speed > 0.3:
            old = next((h for h in self.hist if t - h[0] <= 1.6 and t - h[0] >= 1.2), None)
            if old is not None and math.hypot(x - old[1], y - old[2]) < 1.0 and t >= self.side_until:
                self.side_until = t + 1.2
                self.side_sign = self.rng.choice((-1, 1))
                self.path_goal = None
                if self.m is not None:
                    I, J = self.m.cell(x + mv[0] * 5.0, y + mv[1] * 5.0)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            self.tblock[(J + dj) * self.m.nx + I + di] = t + 8.0
        if t < self.side_until and (mv[0] or mv[1]):
            ang = math.atan2(mv[1], mv[0]) + self.side_sign * 1.1
            mv = (math.cos(ang), math.sin(ang))
            jump = True
        if len(self.tblock) > 200:
            self.tblock = {k: v for k, v in self.tblock.items() if v > t}

        # ---- weapons on the ground we are standing next to
        self.try_take(act, obs, slots, kinds, zc, t, engage)

        act["move"] = [mv[0] * speed, mv[1] * speed]
        self.last_move = (act["move"][0], act["move"][1])
        act["boost"] = bool(boost)
        if jump:
            act["jump"] = True
        if aim is None:
            if mv[0] or mv[1]:
                aim = [math.atan2(mv[1], mv[0]), 0.0]
            else:
                aim = self.scan_aim(t)
        act["aim"] = [aim[0], aim[1]]
        if fire[0] or fire[1]:
            act["fire"] = fire
        self.draw(obs, mode, hpf, vis, tgt)
        return act

    # ------------------------------------------------------------------ pieces of the decision
    def scan_aim(self, t):
        """Nothing to look at: face where trouble last came from, else sweep slowly (the 120 degree view wedge follows the aim)."""
        if t - self.last_dmg_t < 6.0 and self.last_dmg_bearing is not None:
            return [self.last_dmg_bearing, 0.0]
        for (ht, hx, hy, kind, unc) in reversed(self.heard):
            if t - ht < 4.0 and kind in LOUD:
                return [math.atan2(hy - self.y, hx - self.x), 0.0]
        return [(t * 0.9) % (2 * math.pi) - math.pi, 0.0]

    def back_cover(self, px, py, ex, ey):
        """How many of 7 rays fanned across the half-plane behind (px, py), away from the enemy at (ex, ey), hit a wall within 9 m."""
        m = self.m
        gz = m.ground(px, py) + 5.0
        away = math.atan2(py - ey, px - ex)
        hit = 0
        for k in range(-3, 4):
            a = away + k * 0.42
            if not m.los((px, py, gz), (px + 9.0 * math.cos(a), py + 9.0 * math.sin(a), gz), False):
                hit += 1
        return hit

    def find_backwall(self, e, zc, zr, pref):
        """A nearby spot that has a wall behind it (away from the enemy), a clear shot at the enemy, and is still inside the safe circle."""
        m = self.m
        x, y = self.x, self.y
        ex, ey, ez = e["pos"]
        cur = self.back_cover(x, y, ex, ey)
        if cur >= 4:
            return (x, y)
        best, bs = None, cur + 1.5
        for r in (5.0, 9.0, 14.0, 20.0):
            for k in range(12):
                if time.perf_counter() > self.deadline + 0.004:
                    return best
                a = k * math.pi / 6.0 + r * 0.31
                cx, cy = x + r * math.cos(a), y + r * math.sin(a)
                if math.hypot(cx - zc[0], cy - zc[1]) > zr * 0.9 or not m.free(cx, cy) or not m.line_free(x, y, cx, cy):
                    continue
                dd = math.hypot(ex - cx, ey - cy)
                if dd < 22.0 or dd > pref + 45.0:
                    continue
                cz = m.ground(cx, cy)
                if not m.los((cx, cy, cz + 7.0), (ex, ey, ez + 4.5), False):
                    continue
                bc = self.back_cover(cx, cy, ex, ey)
                sc = bc - r * 0.08
                if bc >= 4 and sc > bs:
                    best, bs = (cx, cy), sc
        return best

    # sustained damage per second (before hit rate) of each weapon kind; rough, used only to judge a fight
    DPS = {"gun": 100.0, "laser": 130.0, "mortar": 85.0, "lightning": 150.0}

    def dps_of(self, kinds, cubes, d):
        tot = 0.0
        for k in kinds:
            v = self.DPS.get(k, 0.0)
            if k == "lightning" and d > 45.0:
                v = 0.0
            elif k == "mortar" and d < 20.0:
                v *= 0.5
            tot += v
        return tot * 0.55 * (1.0 + 0.04 * min(cubes or 0, 10))

    def duel_ok(self, vis, me, x, y):
        """True when we expect to kill the single enemy in view before it kills us."""
        if len(vis) != 1:
            return False
        e = vis[0]
        hp_e = e.get("hp")
        if hp_e is None:
            return False
        d = math.hypot(e["pos"][0] - x, e["pos"][1] - y)
        mine = self.dps_of([w["kind"] for w in me["weapons"] if w], me.get("cubes", 0), d)
        theirs = self.dps_of(e.get("weapons") or ["gun"], e.get("cubes", 0), d)
        if mine <= 0.0:
            return False
        ttk = hp_e / mine
        ttd = me["hp"] / max(theirs, 1.0)
        return ttk < 0.75 * ttd

    def pick_target(self, vis, x, y):
        """Nearest visible enemy, with a bias to finishing the weak and to keeping the one we are on."""
        best, bs = None, 1e9
        for e in vis:
            d = math.hypot(e["pos"][0] - x, e["pos"][1] - y)
            hp = e.get("hp")
            s = d + (30.0 * hp / e["maxhp"] if hp is not None else 15.0)
            if e["id"] == self.target_id:
                s -= 25.0
            if e.get("hp") is not None and e["maxhp"] and e["hp"] < 0.35 * e["maxhp"]:
                s -= FINISH_W
            if FACE_ATTACKER and e["id"] == self.last_dmg_by and self.t - self.last_dmg_t < 3.0:
                s -= 70.0
            if s < bs:
                best, bs = e, s
        self.target_id = best["id"] if best else None
        return best

    def mortar_dodge(self, obs, x, y, z):
        """Step out from under an incoming mortar shell (splash radius is 7 m)."""
        for p in obs["projectiles"]:
            if p["kind"] != "mortar":
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            gz = self.m.ground(px, py) if self.m else 0.0
            disc = vz * vz + 2 * 22.0 * (pz - gz)
            if disc < 0:
                continue
            ti = (vz + math.sqrt(disc)) / 22.0
            if not 0.0 < ti < 2.6:
                continue
            ix, iy = px + vx * ti, py + vy * ti
            if math.hypot(ix - x, iy - y) < 11.0:
                return norm(x - ix, y - iy) if math.hypot(x - ix, y - iy) > 0.5 else (1.0, 0.0)
        return None

    def bullet_dodge(self, obs, x, y, z):
        """Sidestep incoming gun bullets: returns (dx, dy, weight) or None. (idea from the ohio-reaper iteration)"""
        cx, cy, cz = x, y, z + 4.5
        mvx, mvy = self.me["vel"][0], self.me["vel"][1]
        best = None
        for p in obs["projectiles"][:24]:
            if p["kind"] == "mortar":
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            rx, ry, rz = px - cx, py - cy, pz - cz
            ux, uy, uz = vx - mvx * 0.5, vy - mvy * 0.5, vz
            vv = ux * ux + uy * uy + uz * uz
            if vv < 1.0 or rx * ux + ry * uy + rz * uz >= 0.0:   # not coming at us (our own shots fly away)
                continue
            ts = -(rx * ux + ry * uy + rz * uz) / vv
            if ts > 0.6:
                continue
            qx, qy, qz = rx + ux * ts, ry + uy * ts, rz + uz * ts
            miss = math.sqrt(qx * qx + qy * qy + qz * qz)
            if miss > 4.2:
                continue
            hv = math.hypot(ux, uy)
            if hv < 1e-6:
                continue
            nx_, ny_ = -uy / hv, ux / hv
            s = qx * nx_ + qy * ny_
            sign = 1.0 if s <= 0 else -1.0
            if abs(s) < 0.3:
                sign = self.strafe
            w = 1.3 * (4.2 - miss) / 4.2 + (0.4 if ts < 0.35 else 0.0)
            if best is None or w > best[2]:
                best = (nx_ * sign, ny_ * sign, w)
        return best

    def hazard_plan(self, obs, x, y, z, want, boost_ok):
        """Dodge planner. Collects every incoming attack (gun bullets, mortar shells, laser telegraphs and
        fresh laser beams), rolls our own motion forward under the acceleration limit for ~34 candidate moves
        (stand, keep going, 16 headings at walk speed, 16 at boost speed), measures the closest approach of
        every hazard to our body along each rolled-out path, and returns (move, speed, boost, margin) for the
        move that clears everything by the widest margin. Returns None when nothing is threatening us or the
        move we already wanted is safe."""
        t = self.t
        me = self.me
        vx0, vy0 = me["vel"][0], me["vel"][1]
        walk = max(me["stats"].get("walk_speed", 11.7), 1.0)
        bspd = me["stats"].get("boost_speed", 19.5)
        acc = me["stats"].get("ground_accel", 58.5)
        R = 2.5 + 0.9                       # body radius plus a safety margin
        cz = z + 4.5
        hz = []                             # (kind, data)
        # --- bullets: keep those that could reach us within 1.3 s if we simply kept our velocity
        cand = []
        for p in obs["projectiles"]:
            if p["kind"] == "mortar":
                continue
            px, py, pz = p["pos"]
            bx, by, bz = p["vel"]
            rx, ry = px - x, py - y
            ux, uy = bx - vx0, by - vy0
            vv = ux * ux + uy * uy
            if vv < 1.0 or rx * ux + ry * uy >= 0.0:
                continue                    # moving away: our own shots, or already past
            ts = -(rx * ux + ry * uy) / vv
            if ts > 1.3:
                continue
            if abs((pz + bz * ts) - cz) > 8.0:
                continue                    # will pass well above or below the body
            qx, qy = rx + ux * ts, ry + uy * ts
            if math.hypot(qx, qy) > 16.0:
                continue
            cand.append((ts, p))
        cand.sort(key=lambda c: c[0])
        for ts, p in cand[:8]:
            hz.append(("b", p["pos"], p["vel"]))
        # --- predictive: an enemy gun that is pointed at us is a bullet that is about to exist
        if PREDODGE:
            for e in self.en.values():
                if e["t"] != t or not e.get("vis") or "gun" not in (e.get("weapons") or ["gun"]):
                    continue
                ex, ey, ez = e["pos"]
                dd = math.hypot(x - ex, y - ey)
                if dd > 230.0 or dd < 4.0:
                    continue
                err = abs(wrap(e["aim"][0] - math.atan2(y - ey, x - ex)))
                if err < 0.05 + 6.0 / dd:
                    cp = math.cos(e["aim"][1])
                    hz.append(("b", (ex + math.cos(e["aim"][0]) * 3.0, ey + math.sin(e["aim"][0]) * 3.0, ez + 7.0),
                               (math.cos(e["aim"][0]) * 252.0 * cp, math.sin(e["aim"][0]) * 252.0 * cp, math.sin(e["aim"][1]) * 252.0)))
        # --- mortar shells: where they will land and when
        for p in obs["projectiles"]:
            if p["kind"] != "mortar":
                continue
            px, py, pz = p["pos"]
            bx, by, bz = p["vel"]
            gz = self.m.ground(px, py) if self.m else 0.0
            disc = bz * bz + 2 * 22.0 * (pz - gz)
            if disc < 0:
                continue
            ti = (bz + math.sqrt(disc)) / 22.0
            if 0.0 < ti < 2.8 and math.hypot(px + bx * ti - x, py + by * ti - y) < 24.0:
                hz.append(("m", px + bx * ti, py + by * ti, ti))
        # --- lasers: a charging enemy aimed at us will fire a hitscan line; fresh beams linger
        for e in self.en.values():
            if e["t"] == t and e.get("charging"):
                ex, ey = e["pos"][0], e["pos"][1]
                dd = math.hypot(x - ex, y - ey)
                if dd < 300.0:
                    off = abs(wrap(e["aim"][0] - math.atan2(y - ey, x - ex)))
                    if off < 0.2 + 7.0 / max(dd, 20.0):
                        hz.append(("l", ex, ey, math.cos(e["aim"][0]), math.sin(e["aim"][0]), 0.5, 0.0))
        for bm in obs.get("beams", ()):
            if bm.get("kind") == "laser" and bm.get("age", 9.0) < 0.35 and len(bm.get("path", ())) >= 2:
                a, b = bm["path"][0], bm["path"][-1]
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                if L > 1.0:
                    hz.append(("l", a[0], a[1], (b[0] - a[0]) / L, (b[1] - a[1]) / L, 0.15, 0.35 - bm["age"]))
        if not hz:
            return None

        N, DT = 10, 0.1
        for h in hz:
            if h[0] == "m":
                N = max(N, min(28, int(h[3] / DT) + 2))

        def rollout(tvx, tvy):
            """Our position at s = DT*k, k = 0..N, going to velocity (tvx, tvy) at the acceleration limit."""
            px, py, vx, vy = x, y, vx0, vy0
            pts = [(px, py)]
            sub = 0.025
            for k in range(N * 4):
                ax, ay = tvx - vx, tvy - vy
                al = math.hypot(ax, ay)
                lim = acc * sub
                if al > lim:
                    ax, ay = ax / al * lim, ay / al * lim
                vx += ax
                vy += ay
                px += vx * sub
                py += vy * sub
                if (k + 1) % 4 == 0:
                    pts.append((px, py))
            return pts

        def margin_of(pts):
            worst = 99.0
            for h in hz:
                if h[0] == "b":
                    (bx0, by0, bz0), (bvx, bvy, bvz) = h[1], h[2]
                    for k in range(min(N, 13)):
                        s0 = k * DT
                        # relative position at segment ends (hazard minus us)
                        r0x = bx0 + bvx * s0 - pts[k][0]
                        r0y = by0 + bvy * s0 - pts[k][1]
                        r1x = bx0 + bvx * (s0 + DT) - pts[k + 1][0]
                        r1y = by0 + bvy * (s0 + DT) - pts[k + 1][1]
                        dx, dy = r1x - r0x, r1y - r0y
                        dd = dx * dx + dy * dy
                        f = 0.0 if dd < 1e-9 else max(0.0, min(1.0, -(r0x * dx + r0y * dy) / dd))
                        dist = math.hypot(r0x + dx * f, r0y + dy * f)
                        if abs((bz0 + bvz * (s0 + f * DT)) - cz) <= 5.2:
                            if dist - R < worst:
                                worst = dist - R
                elif h[0] == "m":
                    ti = h[3]
                    k = min(N - 1, int(ti / DT))
                    f = min(1.0, max(0.0, ti / DT - k))
                    px = pts[k][0] + (pts[k + 1][0] - pts[k][0]) * f
                    py = pts[k][1] + (pts[k + 1][1] - pts[k][1]) * f
                    worst = min(worst, math.hypot(px - h[1], py - h[2]) - 9.5)
                else:
                    _, ex, ey, cx_, cy_, tr, dur = h
                    steps = 1 if dur <= 0.0 else 4
                    for i in range(steps):
                        s = tr + (dur * i / (steps - 1) if steps > 1 else 0.0)
                        k = min(N - 1, int(s / DT))
                        f = min(1.0, max(0.0, s / DT - k))
                        px = pts[k][0] + (pts[k + 1][0] - pts[k][0]) * f
                        py = pts[k][1] + (pts[k + 1][1] - pts[k][1]) * f
                        along = (px - ex) * cx_ + (py - ey) * cy_
                        perp = abs((px - ex) * cy_ - (py - ey) * cx_)
                        if along > 0.0:
                            worst = min(worst, perp - R - 0.6)
            return worst

        wx, wy, wsp = want
        wvx, wvy = wx * walk * wsp, wy * walk * wsp
        base = margin_of(rollout(wvx, wvy))
        if base >= 0.8:
            return None
        m = self.m
        cands = [(0.0, 0.0, 0.0, False, 0.0)]
        for k in range(16):
            a = k * math.pi / 8
            c, s = math.cos(a), math.sin(a)
            if m is not None and not (m.free(x + c * 4.0, y + s * 4.0) and m.free(x + c * 8.0, y + s * 8.0)):
                continue
            cands.append((c * walk, s * walk, 1.0, False, 0.0))
            if boost_ok and (m is None or m.free(x + c * 14.0, y + s * 14.0)):
                cands.append((c * bspd, s * bspd, bspd / walk, True, -0.35))
        best, bs = None, -1e9
        wn = math.hypot(wvx, wvy)
        for tvx, tvy, sp, bo, pen in cands:
            mg = margin_of(rollout(tvx, tvy))
            like = 0.0
            if wn > 0.1 and (tvx or tvy):
                like = (wvx * tvx + wvy * tvy) / (wn * math.hypot(tvx, tvy))
            sc = min(mg, 2.5) + 0.25 * like + pen
            if sc > bs:
                bs, best = sc, (tvx, tvy, sp, bo, mg)
        if best is None or best[4] <= base + 0.15:
            return None
        tvx, tvy, sp, bo, mg = best
        n = math.hypot(tvx, tvy)
        if n < 1e-6:
            return (0.0, 0.0), 1.0, False, mg
        return (tvx / n, tvy / n), min(1.0, n / walk) if not bo else 1.0, bo, mg

    def go_cover(self, tgt, vis, zc, zr, costfn, mv, boost, hpf):
        """Get out of sight and stay there. Returns (move, boost, mode)."""
        t, x, y = self.t, self.x, self.y
        threats = [tuple(e["pos"]) for e in self.en.values() if t - e["t"] <= 8.0]
        if not threats and self.last_dmg_bearing is not None and t - self.last_dmg_t < 8.0:
            threats = [(x + math.cos(self.last_dmg_bearing) * 60.0, y + math.sin(self.last_dmg_bearing) * 60.0, self.z)]
        hid = bool(self.cover) and math.hypot(self.cover[0] - x, self.cover[1] - y) < 2.5
        if (self.cover is None or t - self.cover_t > 1.5) and not (hid and not vis):
            c = self.find_cover(threats, zc, zr)
            if c is not None:
                self.cover = c
            self.cover_t = t
        if self.cover is not None:
            d = self.nav_dir(self.cover[0], self.cover[1], costfn, 2.0)
            if d is None:
                return (0.0, 0.0), False, "hide"
            return d, self.me["energy"] > 40.0, "cover"
        # nowhere to hide: run from the nearest threat
        if threats:
            n = min(threats, key=lambda p: math.hypot(p[0] - x, p[1] - y))
            return self.steer(x, y, *norm(x - n[0], y - n[1])), self.me["energy"] > 40.0, "flee"
        return mv, boost, "hide"

    def roam(self, obs, hpf, kinds, slots, zc, zr, zgoal, costfn, near_recent, st, muz):
        """Nothing visible, nothing urgent: loot, supply drops, farming, otherwise wait quietly in a tight spot."""
        t, x, y, z = self.t, self.x, self.y, self.z
        me = self.me
        mode, mv, speed, boost, aim, fire, quiet, jump = "hold", (0.0, 0.0), 1.0, False, None, [False, False], bool(near_recent), False
        # --- keep or choose a goal
        if self.goal is not None and (t - self.goal_t > 1.2 or not self.goal_valid(obs)):
            self.goal = None
        if self.goal is None:
            self.goal = self.choose_goal(obs, hpf, kinds, slots, zc, zr)
            self.goal_t = t
        g = self.goal
        if g is not None:
            gx, gy = g["pos"][0], g["pos"][1]
            kind = g["kind"]
            if kind in ("crate", "drone"):
                tgt = {"pos": g["pos"], "vel": g.get("vel", [0, 0, 0]), "weapons": [], "kind": kind, "id": -1}
                d = math.hypot(gx - x, gy - y)
                if d <= 34.0 and self.los((x, y, muz), (gx, gy, g["pos"][2] + (1.75 if kind == "drone" else 1.5))):
                    r = self.engage(tgt, st, muz, slots, kinds, farm=True)
                    return "farm", self.steer(x, y, *r["move"]), 1.0, False, r["aim"], r["fire"], False, False
            if kind == "drop":
                return self.drop_step(obs, g, costfn)
            d = self.nav_dir(gx, gy, costfn, 2.0 if kind in ("cube", "weapon") else 4.0)
            if d is not None:
                return "loot-" + kind, d, 1.0, False, None, fire, quiet, False
            self.goal = None
        # --- nothing to do: wait in a tight spot inside the safe circle
        if self.ambush is None or t - self.ambush_t > 20.0:
            self.ambush = self.find_ambush(zc, zr)
            self.ambush_t = t
        if self.ambush is not None and math.hypot(self.ambush[0] - x, self.ambush[1] - y) > 4.0 and math.hypot(x - zc[0], y - zc[1]) > 3.0:
            d = self.nav_dir(self.ambush[0], self.ambush[1], costfn, 3.0)
            if d is not None:
                return "to-ambush", d, 1.0, False, None, fire, True, False
        w0 = slots[0]
        if w0 and w0["kind"] == "gun" and not near_recent and w0.get("ammo", 99) < w0["stats"].get("mag", 12) and not w0.get("reload"):
            pass  # reloading is audible 60 m: only when nobody was seen near
        return "hold", (0.0, 0.0), 1.0, False, None, fire, True, False

    def goal_valid(self, obs):
        g = self.goal
        k = g["kind"]
        if k == "weapon":
            return g["id"] in self.pk and self.bad_take.get(g["id"], 0) <= self.t
        if k == "cube":
            return g["id"] in self.cu
        if k == "crate":
            return g["id"] in self.cr
        if k == "drone":
            return g["id"] in self.npc
        if k == "drop":
            return any(d["id"] == g["id"] for d in obs["drops"])
        return False

    def choose_goal(self, obs, hpf, kinds, slots, zc, zr):
        t, x, y = self.t, self.x, self.y
        m = self.m
        cands = []
        cap = zr * 0.95 + 5.0

        def inside(px, py):
            return math.hypot(px - zc[0], py - zc[1]) <= cap

        def on_ground(p):
            return m is None or p[2] - m.ground(p[0], p[1]) < 3.0   # roofs need jets: skip

        held = [self.weapon_value(k, x, y, zc, t) for k in kinds if k != "gun"]
        worst = min(held) if held else 0.0
        for pid, p in self.pk.items():
            k = p["kind"]
            if k in kinds or self.bad_take.get(pid, 0) > t or not inside(p["pos"][0], p["pos"][1]) or not on_ground(p["pos"]):
                continue
            v = self.weapon_value(k, x, y, zc, t)
            empty = len(kinds) < 2
            if (empty and v >= 0.9) or (not empty and v > worst + 0.5):
                d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
                u = (2.0 + 1.5 * v) / (1.0 + d / 60.0)
                cands.append((u / (1.0 + 2.0 * self.danger_at(p["pos"][0], p["pos"][1], t)), {"kind": "weapon", "id": pid, "pos": p["pos"]}))
        for cid, c in self.cu.items():
            if not inside(c["pos"][0], c["pos"][1]) or not on_ground(c["pos"]):
                continue
            d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
            if d < 120.0:
                u = (1.6 if self.me["cubes"] < 10 else 0.4) / (1.0 + d / 40.0)
                cands.append((u / (1.0 + 2.0 * self.danger_at(c["pos"][0], c["pos"][1], t)), {"kind": "cube", "id": cid, "pos": c["pos"]}))
        calm = not any(t - e["t"] < 20.0 for e in self.en.values()) and hpf > 0.7
        if FARM_CRATES and calm and (len(kinds) < 2 or self.me["cubes"] < 4):
            for cid, c in self.cr.items():
                d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                if d < 70.0 and inside(c["pos"][0], c["pos"][1]) and on_ground(c["pos"]):
                    cands.append((1.2 / (1.0 + d / 40.0), {"kind": "crate", "id": cid, "pos": c["pos"]}))
        if FARM_DRONES and calm and t > 15.0:
            for nid, n in self.npc.items():
                d = math.hypot(n["pos"][0] - x, n["pos"][1] - y)
                if d < 60.0 and inside(n["pos"][0], n["pos"][1]):
                    cands.append((1.0 / (1.0 + d / 40.0), {"kind": "drone", "id": nid, "pos": n["pos"], "vel": n["vel"]}))
        for dr in obs["drops"]:
            dx, dy = dr["pos"][0], dr["pos"][1]
            d = math.hypot(dx - x, dy - y)
            if inside(dx, dy) and dr["lands_in"] < d / (0.85 * 11.7) + 30.0:
                cands.append((3.0 / (1.0 + d / 150.0), {"kind": "drop", "id": dr["id"], "pos": dr["pos"], "ring": dr["radius"]}))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        return cands[0][1]

    def drop_step(self, obs, g, costfn):
        """Walk to a staging spot short of the ring, wait out of the way, and only go in when it looks clear."""
        t, x, y = self.t, self.x, self.y
        dr = next((d for d in obs["drops"] if d["id"] == g["id"]), None)
        if dr is None:
            self.goal = None
            return "hold", (0.0, 0.0), 1.0, False, None, [False, False], True, False
        dx, dy = dr["pos"][0], dr["pos"][1]
        ring = dr["radius"]
        d = math.hypot(dx - x, dy - y)
        safe = self.drop_safe(dx, dy, ring, dr, t)
        if dr["lands_in"] <= DROP_EARLY and safe and (dr["holder"] in (None, self.me["id"])):
            if d < ring * 0.5:
                return "capture", (0.0, 0.0), 1.0, False, None, [False, False], True, False
            dd = self.nav_dir(dx, dy, costfn, ring * 0.4)
            return "go-drop", (dd or (0.0, 0.0)), 1.0, False, None, [False, False], False, False
        ux, uy = norm(x - dx, y - dy)
        sx, sy = dx + ux * (ring + 14.0), dy + uy * (ring + 14.0)
        if self.m is not None and not self.m.free(sx, sy):
            n = self.m.near_free(sx, sy, 8)
            if n is not None:
                sx, sy = self.m.center(n)
        dd = self.nav_dir(sx, sy, costfn, 4.0)
        if dd is None:
            return "stage", (0.0, 0.0), 1.0, False, None, [False, False], True, False
        eta = (d - ring) / (0.95 * 11.7)
        late = dr["lands_in"] < eta - 2.0 and self.me["energy"] > 45.0 and not any(
            t - e["t"] < 10.0 and math.hypot(e["pos"][0] - x, e["pos"][1] - y) < 130.0 for e in self.en.values())
        return "to-stage", dd, 1.0, bool(late and DROP_BOOST), None, [False, False], d < 90.0 and not late, False

    def drop_safe(self, dx, dy, ring, dr, t):
        for e in self.en.values():
            if t - e["t"] <= 4.0 and math.hypot(e["pos"][0] - dx, e["pos"][1] - dy) < 55.0:
                return False
        for (ht, hx, hy, kind, unc) in self.heard:
            if t - ht < 4.0 and kind in LOUD and math.hypot(hx - dx, hy - dy) < 70.0 + unc:
                return False
        return True

    def try_take(self, act, obs, slots, kinds, zc, t, engage):
        if engage:
            return
        x, y, z = self.x, self.y, self.z
        held = {s: self.weapon_value(w["kind"], x, y, zc, t) for s, w in enumerate(slots) if w and w["kind"] != "gun"}
        for p in obs["pickups"]:
            if p["kind"] in kinds or self.bad_take.get(p["id"], 0) > t:
                continue
            if math.hypot(p["pos"][0] - x, p["pos"][1] - y) > self.reach - 0.4 or abs(p["pos"][2] - z) > 3.5:
                continue
            v = self.weapon_value(p["kind"], x, y, zc, t)
            if not slots[1]:
                if v >= 0.9:
                    act["take"] = p["id"]
                    return
            elif held:
                s = min(held, key=lambda k: held[k])
                if v > held[s] + 0.5 and not slots[s].get("charge", 0):
                    act["take"] = p["id"]
                    act["drop"] = s
                    return

    # ------------------------------------------------------------------ fighting
    def engage(self, e, st, muz, slots, kinds, farm):
        """Aim, fire and positioning against one target. Returns dict(aim, fire, move, boost, mode)."""
        me = self.me
        x, y, z = self.x, self.y, self.z
        mx, my, mz = e["pos"]
        vx, vy = e["vel"][0], e["vel"][1]
        d = math.hypot(mx - x, my - y)
        esp = math.hypot(vx, vy)
        ek = set(e.get("weapons") or [])
        if farm:
            cz = mz + (1.75 if e.get("kind") == "drone" else 1.5)
        else:
            cz = mz + 4.5 * (1.0 + 0.05 * (e.get("cubes") or 0))
        seen = True
        gun = next((w for w in slots if w and w["kind"] == "gun"), None)
        lasr = next((w for w in slots if w and w["kind"] == "laser"), None)
        ltng = next((w for w in slots if w and w["kind"] == "lightning"), None)
        mort = next((w for w in slots if w and w["kind"] == "mortar"), None)
        laser_on = bool(lasr and d <= lasr["stats"].get("range", 280.0) * 0.95 and (lasr["cooldown"] <= 0.05 or lasr.get("charge", 0) > 0)
                        and not ("lightning" in ek and d < 50.0))
        ltng_on = bool(ltng and d <= ltng["stats"].get("range", 45.0) * 0.95 and me["energy"] >= 4.0)

        # motion prediction: current velocity + acceleration estimated from the last ~0.4 s of observed
        # velocity (so a strafing bot that is mid-turn is led correctly), accel fades out over 0.6 s because
        # house bots change direction at random; the predicted point is pulled back if it ends inside a wall
        ax_, ay_ = 0.0, 0.0
        hist = e.get("h")
        if hist and len(hist) >= 2 and e.get("t") == self.t and not farm:
            t1, v1x, v1y = hist[-1]
            old = None
            for rec in hist:
                if t1 - rec[0] <= 0.5 and t1 - rec[0] >= 0.15:
                    old = rec
                    break
            if old is not None:
                dtv = t1 - old[0]
                ax_, ay_ = (v1x - old[1]) / dtv, (v1y - old[2]) / dtv
                an = math.hypot(ax_, ay_)
                if an > 58.5:
                    ax_, ay_ = ax_ / an * 58.5, ay_ / an * 58.5
        vmax = 19.6

        def pos_at(tt):
            ta = min(tt, 0.6)
            # displacement with accel acting for ta then constant velocity
            dx = vx * tt + ax_ * (0.5 * ta * ta + ta * (tt - ta))
            dy = vy * tt + ay_ * (0.5 * ta * ta + ta * (tt - ta))
            fvx, fvy = vx + ax_ * ta, vy + ay_ * ta
            fs = math.hypot(fvx, fvy)
            if fs > vmax:      # can't outrun boost speed: scale the extra back
                k = vmax / fs
                dx, dy = vx * tt + (dx - vx * tt) * k, vy * tt + (dy - vy * tt) * k
            px_, py_ = mx + dx, my + dy
            if self.m is not None and not farm and not self.m.free(px_, py_):
                for f in (0.75, 0.5, 0.25, 0.0):
                    qx, qy = mx + dx * f, my + dy * f
                    if self.m.free(qx, qy):
                        return qx, qy
                return mx, my
            return px_, py_

        def lead(speed, extra=0.0):
            tt = d / max(speed, 1.0) + extra
            for _ in range(4):
                px, py = pos_at(tt)
                tt = math.hypot(px - x, py - y) / max(speed, 1.0) + extra
            return pos_at(tt)

        if laser_on:
            px, py = lead(1e9, 0.1)
            mode_aim = "laser"
        elif ltng_on:
            px, py = lead(1e9, 0.05)
            mode_aim = "lightning"
        elif gun:
            px, py = lead(gun["stats"].get("speed", 252.0))
            mode_aim = "gun"
        else:
            px, py = mx, my
            mode_aim = "gun"
        hd = math.hypot(px - x, py - y)
        yaw = math.atan2(py - y, px - x)
        pitch = math.atan2(cz - muz, max(hd, 1.0))
        if mort and 30.0 < d < 210.0 and (esp < 4.0 or "lightning" in ek) and mort["cooldown"] <= 0.05 and not laser_on and not ltng_on:
            v = mort["stats"].get("speed", 72.0)
            g = self.cfg["weapons"]["mortar"].get("gravity", 22.0)
            tx, ty = px, py
            for _ in range(3):
                hdm = math.hypot(tx - x, ty - y)
                disc = v ** 4 - g * (g * hdm * hdm + 2 * (cz - (z + 9.5)) * v * v)
                if disc <= 0:
                    break
                pit = math.atan2(v * v - math.sqrt(disc), g * hdm)
                tf = hdm / (v * math.cos(pit))
                tx, ty = pos_at(tf)
            else:
                hdm = math.hypot(tx - x, ty - y)
                disc = v ** 4 - g * (g * hdm * hdm + 2 * (cz - (z + 9.5)) * v * v)
                if disc > 0:
                    yaw = math.atan2(ty - y, tx - x)
                    pitch = math.atan2(v * v - math.sqrt(disc), g * hdm)
                    mode_aim = "mortar"
        aim = [yaw, max(min(pitch, 1.2), -0.65)]
        err = abs(wrap(yaw - me["aim"][0]))
        lateral = err * hd
        fire = [False, False]
        for s, w in enumerate(slots):
            if not w:
                continue
            k = w["kind"]
            if k == "gun":
                if mode_aim in ("gun", "laser", "lightning") and d <= w["stats"].get("range", 230.0) * 0.97 and w.get("ammo", 1) > 0 \
                        and not w.get("reload", 0) and (lateral < 6.0 or d < 40.0):
                    fire[s] = True
            elif k == "lightning":
                if ltng_on and err < math.radians(w["stats"].get("cone_deg", 30.0)) * 0.45:
                    fire[s] = True
            elif k == "laser":
                if laser_on:
                    full = w["stats"].get("charge", 0.75)
                    ch = w.get("charge", 0)
                    fire[s] = not (ch >= full * 0.9 and (lateral < 2.0 or ch >= full + 1.2))
            elif k == "mortar":
                if mode_aim == "mortar" and err < 0.05:
                    fire[s] = True
        # positioning
        my_k = set(kinds)
        pref = 30.0 if farm else pref_dist(my_k, ek)
        ux, uy = norm(mx - x, my - y)
        if farm:
            sx, sy = 0.0, 0.0
        else:
            t = self.t
            if t >= self.flip_at:
                if self.rng.random() < 0.7:
                    self.strafe = -self.strafe
                self.flip_at = t + self.rng.uniform(0.4, 1.2)
            sx, sy = -uy * self.strafe, ux * self.strafe
        rad = 1.0 if d > pref + 15.0 else (-1.0 if d < pref - 15.0 else 0.0)
        mvx, mvy = norm(sx * 0.9 + ux * rad * 0.7, sy * 0.9 + uy * rad * 0.7)
        if BACKWALL and not farm and self.m is not None and d > 28.0 and d < pref + 60.0:
            t = self.t
            if t - self.anchor_t > 0.8 or self.anchor_for != e.get("id"):
                zn = self.obs_zone
                self.anchor = self.find_backwall(e, zn["center"], zn["radius"], pref)
                self.anchor_t = t
                self.anchor_for = e.get("id")
            if self.anchor is not None:
                ax, ay = self.anchor[0] - x, self.anchor[1] - y
                ad = math.hypot(ax, ay)
                if ad > 2.0:
                    # walk to the wall spot, still side-stepping a little
                    ax, ay = ax / ad, ay / ad
                    mvx, mvy = norm(ax * 1.0 + sx * 0.35, ay * 1.0 + sy * 0.35)
                else:
                    # in place: shuffle along the wall instead of leaving it
                    mvx, mvy = norm(sx * 0.6, sy * 0.6)
        boost = False
        mode = "farm" if farm else "fight"
        if e.get("charging") and abs(wrap(e["aim"][0] - math.atan2(y - my, x - mx))) < 0.25 + 6.0 / max(d, 10.0):
            mvx, mvy = -uy * self.strafe, ux * self.strafe
            boost = me["energy"] > 30.0
            mode = "dodge"
        return {"aim": aim, "fire": fire, "move": (mvx, mvy), "boost": boost, "mode": mode}

    # ------------------------------------------------------------------ debug drawing (only your team sees it)
    def draw(self, obs, mode, hpf, vis, tgt):
        t, x, y, z = self.t, self.x, self.y, self.z
        mf.draw.label("%s hp%d e%d %s" % (mode, int(hpf * 100), len(vis), self.region(x, y)))
        if tgt is not None:
            mf.draw.line([x, y, z + 7], [tgt["pos"][0], tgt["pos"][1], tgt["pos"][2] + 4.5], "#ff5040")
        for e in self.en.values():
            if t - e["t"] <= 20.0:
                mf.draw.point([e["pos"][0], e["pos"][1], e["pos"][2] + 2], "#ff9000" if t - e["t"] > 1.0 else "#ff0000", 5)
        if self.goal is not None:
            mf.draw.line([x, y, z + 1], [self.goal["pos"][0], self.goal["pos"][1], z + 1], "#40c0ff")
        if self.cover is not None and self.hiding:
            mf.draw.point([self.cover[0], self.cover[1], z + 1], "#40ff80", 7)


mf.run(Bot)
