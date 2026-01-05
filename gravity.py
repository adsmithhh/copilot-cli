import yaml
import pygame
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import itertools
import os

# --- Load YAML config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "int3.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --- Use config values ---
WIDTH = config["display"]["width"]
HEIGHT = config["display"]["height"]
FPS = config["display"]["fps"]

COLORS = {k: tuple(v) for k, v in config["colors"].items()}
BLACK = COLORS["BLACK"]
WHITE = COLORS["WHITE"]
TEAL = COLORS["TEAL"]
ORANGE = COLORS["ORANGE"]
RED_ORANGE = COLORS["RED_ORANGE"]
GREEN = COLORS["GREEN"]
LIME = COLORS["LIME"]
GREEN_TEXT = COLORS["GREEN_TEXT"]
RED_TEXT = COLORS["RED_TEXT"]
YELLOW_TEXT = COLORS["YELLOW_TEXT"]
TRADE_COLOR = COLORS["TRADE_COLOR"]
DEV_COLOR = COLORS["DEV_COLOR"]
FLEX_COLOR = COLORS["FLEX_COLOR"]

MAX_NPCS_FOR_INDIVIDUAL_DISPLAY = 30
SHIFT_DURATION = 300  # Ticks per shift

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("System Architect - Integrated Zone Economy")
clock = pygame.time.Clock()

font_header = pygame.font.Font(None, config["fonts"]["header"])
font_large = pygame.font.Font(None, config["fonts"]["large"])
font_small = pygame.font.Font(None, config["fonts"]["small"])
font_tiny = pygame.font.Font(None, config["fonts"]["tiny"])

NPC_ID_GEN = itertools.count(1)
global_tick = 0
global_cycle = 0


# ---------- ZONES ----------
ZONES = []
for z in config["zones"]:
    color = COLORS[z["color"]]
    ZONES.append({
        "name": z["name"],
        "color": color,
        "angle": z["angle"],
        "description": z["description"],
        "npc_count": z.get("npc_count", 3) # NPCs number
    })

# ---------- EVENTS ----------
EVENTS = []
for e in config["events"]["list"]:
    EVENTS.append({
        "name": e["name"],
        "zone": e["zone"],
        "duration": e["duration"],
        "effect": e["effect"]
    })

# --- Add at the top, after config is loaded ---
ZONE_SECTORS = {
    "SCIENCE": 0,
    "TRADE": 90,
    "DEVELOPMENT": 180,
    "FLEX": 270
}

# --- CAMPUS LAYOUT DEFINITION ---
# H = Housing (Bottom Center)
# 1 = Science (Middle Left)
# 2 = Trade (Top Left)
# 3 = Development (Top Right)
# 4 = Flex (Middle Right)

ZONE_CENTERS = {
    "HOUSING": (WIDTH // 2, HEIGHT - 100),           # Bottom center
    "SCIENCE": (80, HEIGHT // 2),                    # Middle left edge
    "TRADE": (WIDTH // 2, 80),                       # Middle top edge
    "DEVELOPMENT": (WIDTH - 80, HEIGHT // 2),        # Middle right edge
    "FLEX": (WIDTH // 2, HEIGHT - 220)               # Middle bottom edge, above housing
}

def get_npc_position(zone, idx, total_in_zone, center_x, center_y):
    # Get base angle for the zone sector (degrees to radians)
    angle_base = math.radians(ZONE_SECTORS[zone])  # This converts degrees to radians
    
    # Spread NPCs within the sector
    spread = config["rendering"]["npc_angle_spread"]  # This is in radians (0.3)
    
    # Center the spread around the sector's base angle
    start_angle = angle_base - (spread * (total_in_zone - 1) / 2)
    angle = start_angle + idx * spread
    
    radius = config["rendering"]["npc_radius"]
    x = center_x + math.cos(angle) * radius
    y = center_y + math.sin(angle) * radius
    return x, y

 # DEBUG: Print what's happening
    print(f"  POSITION DEBUG: zone={zone}, idx={idx}, total={total_in_zone}")
    print(f"    angle_base={math.degrees(angle_base):.1f}°, spread={math.degrees(spread):.1f}°")
    print(f"    start_angle={math.degrees(start_angle):.1f}°, angle={math.degrees(angle):.1f}°")
    print(f"    radius={radius}, center=({center_x:.0f}, {center_y:.0f})")
    print(f"    final position=({x:.0f}, {y:.0f})")
   

# ---------- PERSONALITY ----------
@dataclass
class Personality:
    industrious: float
    greedy: float
    social: float
    ambitious: float
    risk_tolerance: float

    @staticmethod
    def random():
        p = config["personality"]
        return Personality(
            industrious=random.uniform(*p["industrious"]),
            greedy=random.uniform(*p["greedy"]),
            social=random.uniform(*p["social"]),
            ambitious=random.uniform(*p["ambitious"]),
            risk_tolerance=random.uniform(*p["risk_tolerance"])
        )

# ---------- COHERENCE FIELD ----------
@dataclass
class CoherenceField:
    cluster_id: str
    zone: str
    center: tuple
    field_strength: float
    field_radius: float
    coherence_mean: float
    variance: str

# ---------- NPC ----------
@dataclass
class NPC:
    name: str
    zone: str
    personality: Personality
    x: float
    y: float

    id: int = field(default_factory=lambda: next(NPC_ID_GEN))

    # Resources
    basic_needs: float = field(default_factory=lambda: config["npc_initial"]["basic_needs"])
    money: float = field(default_factory=lambda: config["npc_initial"]["money"])
    knowledge: float = field(default_factory=lambda: config["npc_initial"]["knowledge"])
    influence: float = field(default_factory=lambda: config["npc_initial"]["influence"])

    # State
    mood: float = field(default_factory=lambda: config["npc_initial"]["mood"])
    health: float = field(default_factory=lambda: config["npc_initial"]["health"])
    energy: float = field(default_factory=lambda: config["npc_initial"]["energy"])
    status: float = field(default_factory=lambda: config["npc_initial"]["status"])

    # Zone specific
    zone_integrity: float = field(default_factory=lambda: config["npc_initial"]["zone_integrity"])
    learning_progress: float = field(default_factory=lambda: config["npc_initial"]["learning_progress"])
    time_in_zone: int = field(default_factory=lambda: config["npc_initial"]["time_in_zone"])

    # Mechanics
    productivity: float = field(default_factory=lambda: config["npc_initial"]["productivity"])
    consumption_rate: float = field(default_factory=lambda: config["npc_initial"]["consumption_rate"])

    # Social
    relationships: Dict[int, float] = field(default_factory=dict)
    rival_id: Optional[int] = None

    # NEW: Action profile fields
    npc_action_profile: Dict[str, float] = field(default_factory=lambda: {
        "exchange_actions": 0,
        "long_term_commitments": 0,
        "risk_absorptions": 0,
        "stress_endured": 0,
    })

    # Shift System
    shift: int = 0
    station_x: float = 0
    station_y: float = 0
    rest_x: float = 0
    rest_y: float = 0
    
    # Physics
    vx: float = 0.0
    vy: float = 0.0

    # --- GHOST FEATURE ---
    is_ghost: bool = False
    ghost_timer: int = 0

    def get_color(self):
        # Use zone color from config, fallback to WHITE
        base_color = COLORS.get(self.zone.upper() + "_COLOR", WHITE)
        if self.is_ghost:
            return (*base_color, 100)  # RGBA, alpha=100 for ghost
        return base_color 

    def consume(self):
        self.basic_needs -= config["npc_consume"]["basic_needs_drain"]
        self.energy -= config["npc_consume"]["energy_drain"]

        if self.basic_needs < 0:
            self.basic_needs = 0
            self.health -= config["npc_consume"]["health_penalty"]
            self.mood -= config["npc_consume"]["mood_penalty"]

        if self.energy < 0:
            self.energy = 0
            self.productivity *= config["npc_consume"]["productivity_decay"]

        self.mood = max(0.0, min(1.0, self.mood))

    def detect_cycle_phase(world, threshold=None):
        if threshold is None:
            threshold = config["world_shortage"]["pool_threshold_multiplier"]  # Or another suitable config value

        npc_count = max(1, len(world.npcs))  # <-- Add this line
        pool_ratio = world.pool["basic_needs"] / (npc_count * threshold)
        avg_mood = sum(n.mood for n in world.npcs) / npc_count
        avg_commitment = sum(
            n.npc_action_profile["long_term_commitments"] 
            for n in world.npcs
        ) / npc_count

        return {
            "phase": "boom" if pool_ratio > 1.2 else "crash" if pool_ratio < 0.3 else "stable",
            "cycle_health": avg_commitment / (avg_mood + 0.01),
            # "period_ticks": detect_oscillation_period()
        }

    def get_field_influence(self, active_fields: List["CoherenceField"]) -> Dict[str, float]:
        """Check if NPC is within any active field and return modifiers."""
        base_influence = {"recovery_boost": 1.0, "stress_mod": 1.0}
        
        for field in active_fields:
            if field.zone == self.zone:
                dist = math.hypot(self.x - field.center[0], self.y - field.center[1])
                if dist < field.field_radius:
                    # Inside the field!
                    base_influence["recovery_boost"] = 1.0 + (field.field_strength * 0.5) # +50% max
                    base_influence["stress_mod"] = max(0.2, 1.0 - (field.field_strength * 0.8)) # -80% stress max
                    return base_influence
        return base_influence



    def work(self, world: "World"):
        # Apply the 5% ghost bonus to productivity
        productivity_factor = self.productivity * max(self.health, config["npc_work"]["min_health_factor"])
        if self.is_ghost:
            productivity_factor *= 1.05  # THE 5% GHOST EARNINGS BONUS
        income = productivity_factor * (self.energy / 100)

        zone_factor = self.zone_integrity / 100
        zone_level_bonus = 1.0 + (world.zone_building_levels.get(self.zone, 1) - 1) * 0.15

        if self.zone == "SCIENCE":
            w = config["npc_work"]["science"]
            learning = income * w["learning_multiplier"] * zone_factor * zone_level_bonus
            self.learning_progress += learning
            self.knowledge += learning
            world.pool["knowledge"] += learning * w["knowledge_pool_share"]
            self.money += income * w["money_per_base"] * zone_level_bonus
            self.status += w["status_gain"] * self.personality.ambitious * zone_level_bonus
            if self.knowledge > w["knowledge_sell_threshold"] and self.basic_needs < config["npc_work"]["mood_loss_threshold"]:
                self.knowledge -= w["knowledge_sell_amount"]
                self.basic_needs += w["food_gain"]
                self.money += w["money_gain"] * zone_level_bonus
        elif self.zone == "DEVELOPMENT":
            w = config["npc_work"]["development"]
            food = income * w["food_multiplier"] * zone_factor * zone_level_bonus
            self.basic_needs += food
            world.pool["basic_needs"] += food * w["pool_share"]
            self.money += income * w["money_per_base"] * zone_level_bonus
            self.zone_integrity = min(100, self.zone_integrity + w["integrity_gain"] * zone_level_bonus)
        elif self.zone == "TRADE":
            w = config["npc_work"]["trade"]
            self.money += income * w["money_multiplier"] * zone_level_bonus
            self.basic_needs += income * w["food_multiplier"] * zone_factor * zone_level_bonus
            world.pool["basic_needs"] += income * w["pool_share"]
            world.price_volatility *= w["volatility_decay"]
        elif self.zone == "FLEX":
            w = config["npc_work"]["flex"]
            roll = random.random() * self.personality.risk_tolerance
            if roll > w["success_threshold"]:
                self.influence += income * w["influence_gain"] * zone_level_bonus
                self.money += income * w["money_gain_success"] * zone_level_bonus
                self.status += w["status_gain"] * zone_level_bonus
                self.basic_needs += income * w["food_gain"] * zone_level_bonus
            else:
                self.mood = max(0.0, self.mood - w["mood_penalty"])
                self.money += income * w["money_gain_failure"] * zone_level_bonus
            self.zone_integrity = max(w["integrity_min"], self.zone_integrity - w["integrity_decay"])
        self.basic_needs = min(config["npc_work"]["basic_needs_cap"], self.basic_needs)
        self.time_in_zone += 1

        # Mood effects
        if self.basic_needs > config["npc_work"]["mood_gain_threshold"]:
            self.mood = min(1.0, self.mood + config["npc_work"]["mood_gain"])
        elif self.basic_needs < config["npc_work"]["mood_loss_threshold"]:
            self.mood = max(0.0, self.mood - config["npc_work"]["mood_loss"])

        self.mood = max(0.0, min(1.0, self.mood))

        # Integrity decay when not working
        if self.zone != "DEVELOPMENT":
            self.zone_integrity = max(config["npc_work"]["integrity_min_non_dev"], self.zone_integrity - config["npc_work"]["integrity_decay_non_dev"])
        # NEW: Track long-term commitment
        # if self.time_in_zone % 50 == 0:
        #    self.npc_action_profile["long_term_commitments"] += 1

    def trade(self, world: "World"):
        t = config["npc_trade"]
        price = world.prices["basic_needs"]

        if self.basic_needs < t["buy_threshold"] and self.money > price * t["buy_price_multiplier"]:
            qty = min(t["buy_quantity"], (t["buy_threshold"] + t["buy_buffer"] - self.basic_needs), self.money / (price * t["buy_price_multiplier"]))
            self.basic_needs += qty
            self.money -= qty * price
            world.transaction_volume += 1
            self.npc_action_profile["exchange_actions"] += 1

        elif self.basic_needs > t["sell_threshold"]:
            qty = min(t["sell_quantity"], self.basic_needs - t["sell_buffer"])
            self.basic_needs -= qty
            self.money += qty * price * (0.9 + t["greed_bonus"] * self.personality.greedy)
            if world.shortage and self.basic_needs > t["donation_buffer"]:
                donation = min(t["donation_quantity"], self.basic_needs - t["donation_buffer"])
                self.basic_needs -= donation
                world.pool["basic_needs"] += donation

    def socialize(self, world: "World"):
        s = config["npc_socialize"]
        others = [npc for npc in world.npcs if npc.id != self.id and npc.zone == self.zone]
        if not others:
            return

        other = random.choice(others)
        if other.id not in self.relationships:
            self.relationships[other.id] = 0.0

        affinity_change = s["affinity_change"] * self.personality.social
        self.relationships[other.id] = max(s["relationship_min"], min(s["relationship_max"],
            self.relationships[other.id] + affinity_change))

        self.mood = min(1.0, self.mood + s["mood_boost_base"] * (1 + self.personality.social))

        # Rivalry formation
        if self.personality.ambitious > s["rivalry"]["ambition_threshold"] and other.personality.ambitious > s["rivalry"]["ambition_threshold"]:
            if abs(self.status - other.status) < s["rivalry"]["status_diff_threshold"] and random.random() < s["rivalry"]["probability"]:
                self.rival_id = other.id
                other.rival_id = self.id

        # Resource sharing
        if self.relationships[other.id] > s["sharing"]["relationship_threshold"]:
            if other.basic_needs < s["sharing"]["need_threshold"] and self.basic_needs > s["sharing"]["surplus_threshold"]:
                transfer = min(s["sharing"]["transfer_amount"], self.basic_needs - s["sharing"]["transfer_buffer"])
                self.basic_needs -= transfer
                other.basic_needs += transfer

    def rest(self):
        r = config["npc_rest"]
        self.health = min(1.0, self.health + r["health_recovery"])
        self.energy = min(100, self.energy + r["energy_recovery"])
        self.mood = min(1.0, self.mood + r["mood_boost"])
        if self.basic_needs < r["emergency_threshold"]:
            self.basic_needs += r["emergency_rations"]

    def compete(self, world: "World"):
        c = config["npc_compete"]

        if self.rival_id and self.personality.ambitious > c["ambition_threshold"]:
            rival = next((n for n in world.npcs if n.id == self.rival_id), None)
            if rival and rival.zone == self.zone:
                if self.money > c["money_threshold"]:
                    self.money -= c["money_cost"]
                    self.status += c["status_gain"]
                    rival.status -= c["rival_status_loss"]
                    rival.mood = max(0.0, rival.mood - c["rival_mood_penalty"])
                    # Example: increment risk_absorptions when money is spent to compete
                    self.npc_action_profile["risk_absorptions"] += 1

    def choose_action(self, world: "World") -> str:
        a = config["npc_action_utility"]
        needs = max(0, a["targets"]["needs_comfortable"] - self.basic_needs)
        money_need = max(0, a["targets"]["money_comfortable"] - self.money)
        mood_need = max(0, a["targets"]["mood_comfortable"] - self.mood)

        utils = {
            "WORK": needs * a["work"]["needs_weight"] + money_need * a["work"]["money_weight"] + self.personality.industrious * a["work"]["industrious_weight"] + a["work"]["base_utility"],
            "TRADE": money_need * a["trade"]["money_weight"] + needs * a["trade"]["needs_weight"],
            "SOCIALIZE": mood_need * a["socialize"]["mood_weight"] + self.personality.social * a["socialize"]["social_weight"],
            "REST": (1 - self.health) * a["rest"]["health_weight"] + (100 - self.energy) * a["rest"]["energy_weight"],
            "COMPETE": self.personality.ambitious * a["compete"]["ambition_weight"] + self.status * a["compete"]["status_weight"],
        }

        # Emergency priorities
        if self.basic_needs < a["emergency"]["needs_critical"]:
            return "TRADE" if self.money > a["emergency"]["money_threshold"] else "WORK"

        if self.basic_needs > a["trade"]["surplus_threshold"]:
            utils["TRADE"] += a["trade"]["surplus_bonus"]

        for k in utils:
            utils[k] += random.uniform(*a["randomness"])

        return max(utils, key=utils.get)

    def act(self, world: "World"):
        # --- FIELD INFLUENCE ---
        influence = self.get_field_influence(world.active_fields)

        # --- 1. SPATIAL TRANSFORMATION (MIGRATING ANCHORS) ---
        
        # Calculate Breathing Radius (The "Sphere of Work")
        cycle_progress = (global_tick % 2000) / 2000.0 
        breathing_phase = (math.sin(cycle_progress * math.pi * 2) + 1) / 2 # 0.0 to 1.0 (Sinusoidal)
        
        # Determine Phase: Rest vs Work
        # We classify based on the breath. 
        # > 0.3 means "Expanding to Work". < 0.3 means "Contracting to Home".
        is_working = breathing_phase > 0.3
        
        # --- MIGRATING ANCHOR LOGIC ---
        # The Anchor Point itself moves or switches.
        
        home_anchor = ZONE_CENTERS["HOUSING"]
        work_anchor = ZONE_CENTERS.get(self.zone, home_anchor) # Default to home if zone not found
        
        # Linear Interpolation of the Anchor Position based on breathing phase?
        # Or a hard switch? The user asked for "Stringed".
        # Let's slide the anchor.
        
        # We map breathing_phase (0..1) to a "Commute Factor" (0..1)
        # 0.0 = At Home (Housing Anchor)
        # 1.0 = At Work (Zone Anchor)
        
        # Use checks to create a "Stay at work for a bit" and "Stay at home for a bit" rhythm
        # adjust phase to have plateaus
        commute_factor = 0.0
        if breathing_phase < 0.2:
            commute_factor = 0.0 # Stay Home
        elif breathing_phase > 0.8:
            commute_factor = 1.0 # Stay Safe at Work
        else:
            # Transit
            commute_factor = (breathing_phase - 0.2) / 0.6
            
        # Calculate Current Anchor Position (lerp)
        anchor_x = home_anchor[0] + (work_anchor[0] - home_anchor[0]) * commute_factor
        anchor_y = home_anchor[1] + (work_anchor[1] - home_anchor[1]) * commute_factor

        # Tether Logic
        # The string length can be uniform or variable.
        # Let's keep it relatively tight so they cluster around the moving anchor.
        current_tether_length = 40.0 + (30.0 * (1.0 - commute_factor)) # Tighter at work, looser at home? Or vice versa.
        
        dx = anchor_x - self.x
        dy = anchor_y - self.y
        dist = math.hypot(dx, dy)
        
        # Spring Force
        force_x = 0
        force_y = 0
        
        if dist > current_tether_length:
            # Stretched tight
            spring_k = 0.04 # Tension
            pull = (dist - current_tether_length) * spring_k
            force_x += (dx / dist) * pull
            force_y += (dy / dist) * pull
        else:
            # Inside the cluster - drift
            drift_k = 0.005 # Slight drift to keep them moving
            force_x += (dx / dist) * drift_k
            force_y += (dy / dist) * drift_k
            
        # Wander Force (Brownian motion)
        force_x += random.uniform(-0.3, 0.3)
        force_y += random.uniform(-0.3, 0.3)
        
        # Apply Physics
        self.vx += force_x
        self.vy += force_y
        
        # Friction/Damping
        self.vx *= 0.85 # Higher friction for less chaos
        self.vy *= 0.85
        
        # Update Position
        self.x += self.vx
        self.y += self.vy

        if is_working:
             self.consume()
             action = self.choose_action(world)
             in_stress = (self.basic_needs < 5 or self.energy < 30 or self.health < 0.3 or self.mood < 0.3)
     
             if action == "WORK":
                 if in_stress:
                     self.npc_action_profile["stress_endured"] += 0.3 * influence['stress_mod']
                 self.work(world)
             elif action == "TRADE":
                 if in_stress:
                     self.npc_action_profile["stress_endured"] += 0.2 * influence['stress_mod']
                 self.trade(world)
             elif action == "SOCIALIZE":
                 if in_stress:
                     self.npc_action_profile["stress_endured"] += 0.15 * influence['stress_mod']
                 self.socialize(world)
             elif action == "REST":
                 # Even at work they can rest if needed
                 # RECOVERY BOOST from Field
                 recovery_power = (1.0 + self.learning_progress * 0.1) * influence['recovery_boost']
                 # Apply boosted rest (simplified by just calling rest multiple times or hacking values)
                 # Instead of redefining rest, we'll boost the stats directly here for simplicity
                 self.health = min(1.0, self.health + config["npc_rest"]["health_recovery"] * (influence['recovery_boost'] - 1.0))
                 self.energy = min(100, self.energy + config["npc_rest"]["energy_recovery"] * (influence['recovery_boost'] - 1.0))
                 self.mood = min(1.0, self.mood + config["npc_rest"]["mood_boost"] * (influence['recovery_boost'] - 1.0))
                 
                 self.rest()
             elif action == "COMPETE":
                if in_stress:
                     self.npc_action_profile["stress_endured"] += 0.4 * influence['stress_mod']
                self.compete(world)

        else:
             # --- RESTING / "HOME" Phase ---
             # The Anchor Logic above handles the movement to H.
             
             # Just act like they are off-shift
             self.consume()
             self.rest()
             if random.random() < 0.2:
                 self.socialize(world)

        # --- 3. THE SAFETY NET (Fixed placement) ---
        if self.basic_needs < 2.0 and world.pool["basic_needs"] > 1:
            welfare = min(3, world.pool["basic_needs"] * 0.2)
            self.basic_needs += welfare
            world.pool["basic_needs"] -= welfare

        # Final state clamps
        self.mood = max(0.0, min(1.0, self.mood))
        self.health = max(0.0, min(1.0, self.health))
        self.energy = max(0.0, min(100.0, self.energy))
        self.basic_needs = max(0.0, min(config["npc_work"]["basic_needs_cap"], self.basic_needs))

        if self.time_in_zone % 50 == 0:
            self.npc_action_profile["long_term_commitments"] += 1 

# ---------- WORLD ----------
@dataclass
class World:
    npcs: List[NPC] = field(default_factory=list)
    pool: Dict[str, float] = field(default_factory=lambda: config["world_initial"]["pool"].copy())
    prices: Dict[str, float] = field(default_factory=lambda: config["world_initial"]["prices"].copy())
    
    tick: int = 0
    cycles: int = 0
    event: Optional[Dict] = None
    event_timer: int = 0
    price_volatility: float = 1.0
    transaction_volume: int = 0
    shortage: bool = False
    
    # NEW: Active Resonance Fields
    active_fields: List[CoherenceField] = field(default_factory=list)

    # World Position
    home_x: int = 0
    home_y: int = 0
    
    # NEW: Building levels per zone
    zone_building_levels: Dict[str, int] = field(default_factory=lambda: {
        "SCIENCE": 1,
        "TRADE": 1,
        "DEVELOPMENT": 1,
        "FLEX": 1
    })

    def trigger_event(self):
        if random.random() < config["events"]["probability"] and self.event is None:
            self.event = random.choice(EVENTS).copy()
            self.event_timer = self.event["duration"]

    def update_prices(self):
        wp = config["world_prices"]
        supply = sum(max(0, n.basic_needs - wp["supply_buffer"]) for n in self.npcs) + self.pool["basic_needs"]
        demand = sum(max(0, wp["demand_buffer"] - n.basic_needs) for n in self.npcs)
        
        supply = max(supply, 1)
        demand = max(demand, 1)
        
        ratio = demand / supply
        ratio = max(wp["ratio_min"], min(ratio, wp["ratio_max"]))
        
        if self.event and self.event["zone"] == "TRADE":
            ratio *= self.event["effect"].get("price", 1.0)
        
        target = 1.0 * ratio * self.price_volatility
        self.prices["basic_needs"] = (
            wp["smooth_factor"] * self.prices["basic_needs"] + 
            (1 - wp["smooth_factor"]) * target
        )
        self.prices["basic_needs"] = max(wp["price_min"], min(wp["price_max"], self.prices["basic_needs"]))
        
        self.shortage = self.pool["basic_needs"] < len(self.npcs) * config["world_shortage"]["pool_threshold_multiplier"]


    def step(self):
        self.split_and_cluster() # Check for resonance fields
        self.tick += 1
        # Passive generation
        base_gen = len(self.npcs) * config["world_generation"]["base_per_npc"]
        if self.event and self.event["zone"] == "DEVELOPMENT":
            base_gen *= self.event["effect"].get("production", 1.0)
        self.pool["basic_needs"] += base_gen
        # Event management
        if self.event:
            self.event_timer -= 1
            if self.event_timer <= 0:
                self.event = None
        else:
            self.trigger_event()
        # NPC actions
        self.transaction_volume = 0
        for npc in self.npcs:
            npc.act(self)
            # Progressive taxation
            wt = config["world_taxation"]
            if npc.money > wt["wealth_threshold"]:
                tax = (npc.money - wt["wealth_threshold"]) * wt["tax_rate"]
                npc.money -= tax
                self.pool["basic_needs"] += tax * wt["food_conversion"]
            # Event effects
            if self.event and npc.zone == self.event["zone"]:
                if "mood" in self.event["effect"]:
                    npc.mood = max(0, min(1, npc.mood + self.event["effect"]["mood"]))
                if "integrity" in self.event["effect"]:
                    npc.zone_integrity = max(60, min(100, npc.zone_integrity + self.event["effect"]["integrity"]))
        self.update_prices()
        # NEW: Check zone advancement conditions
        self._update_zone_levels()
        # UBI
        ubi = config["world_ubi"]
        for npc in self.npcs:
            if npc.money < ubi["money_threshold"] and npc.basic_needs < ubi["needs_threshold"]:
                npc.money += ubi["money_grant"]
                npc.basic_needs += ubi["needs_grant"]
        # Emergency redistribution
        wr = config["world_redistribution"]
        if self.pool["basic_needs"] < wr["pool_critical"]:
            wealthy = [n for n in self.npcs if n.basic_needs > wr["wealthy_threshold"]]
            if wealthy:
                for rich in wealthy[:wr["wealthy_count"]]:
                    transfer = min(wr["transfer_amount"], rich.basic_needs - wr["transfer_buffer"])
                    rich.basic_needs -= transfer
                    self.pool["basic_needs"] += transfer
                    pool_ratio = world.pool["basic_needs"] / (len(world.npcs) * threshold)
        # Pool management
        wpm = config["world_pool_management"]
        if self.pool["basic_needs"] > wpm["overflow_threshold"]:
            self.pool["basic_needs"] *= wpm["decay_factor"]
        # Volatility decay
        self.price_volatility = 1.0 + (self.price_volatility - 1.0) * config["world_volatility"]["decay_factor"]
        if self.tick % config["world_cycles"]["ticks_per_cycle"] == 0:
            self.cycles += 1

    def _update_zone_levels(self):
        """Gate zone leveling by NPC action profiles."""
        # SCIENCE: needs stability (commitments) over risk
        if self.zone_building_levels["SCIENCE"] == 1:
            sci = self.get_zone_readiness_profile("SCIENCE")
            if sci["npc_count"] > 0:
                commitment_ratio = sci["commitment_depth"] / (sci["risk_appetite"] + 1)
                resilience_per_npc = sci["stress_resilience"] / sci["npc_count"]
                # SCIENCE L2 requires: settlers > entrepreneurs, AND proven resilience
                if commitment_ratio > 0.6 and resilience_per_npc > 30 and sci["npc_count"] >= 2:
                    self.zone_building_levels["SCIENCE"] = 2
        # DEVELOPMENT: needs commitment + builders
        if self.zone_building_levels["DEVELOPMENT"] == 1:
            dev = self.get_zone_readiness_profile("DEVELOPMENT")
            if dev["npc_count"] > 0:
                commitment_per_npc = dev["commitment_depth"] / dev["npc_count"]
                # DEVELOPMENT L2 requires: high commitment AND low risk (stable builders)
                if commitment_per_npc > 40 and dev["risk_appetite"] < dev["commitment_depth"] * 0.5:
                    self.zone_building_levels["DEVELOPMENT"] = 2
        if self.zone_building_levels["TRADE"] == 1:
            tra = self.get_zone_readiness_profile("TRADE")
            if tra["npc_count"] > 0:
                exchange_per_npc = tra["exchange_volume"] / tra["npc_count"]
                # TRADE L2 requires: high transaction frequency
                if exchange_per_npc > 50 and tra["exchange_volume"] > 100:
                    self.zone_building_levels["TRADE"] = 2
        # FLEX: needs risk-takers but also stress resilience (survival)
        if self.zone_building_levels["FLEX"] == 1:
            flx = self.get_zone_readiness_profile("FLEX")
            if flx["npc_count"] > 0:
                risk_per_npc = flx["risk_appetite"] / flx["npc_count"]
                resilience_per_npc = flx["stress_resilience"] / flx["npc_count"]
                # FLEX L2 requires: risk-takers who also survive (brave AND tough)
                if risk_per_npc > 30 and resilience_per_npc > 20:
                    self.zone_building_levels["FLEX"] = 2

    def split_and_cluster(self):
        
        self.active_fields = []
        zone_names = ["SCIENCE", "TRADE", "DEVELOPMENT", "FLEX"]
        
        for zone in zone_names:

            npcs = [n for n in self.npcs if n.zone == zone]
            if len(npcs) < 5: continue
            
            # 1. Calculate centroid
            pos_x = [n.x for n in npcs]
            pos_y = [n.y for n in npcs]
            center_x = sum(pos_x) / len(npcs)
            center_y = sum(pos_y) / len(npcs)
            
            # 2. Calculate variance (average squared distance from center)
            dist_sq_sum = sum((n.x - center_x)**2 + (n.y - center_y)**2 for n in npcs)
            variance = dist_sq_sum / len(npcs)
            
            # 3. Coherence Mean: Lower variance = Higher Resonance
            # Normalizing variance to a 0-1 scale is tricky without bounds, but let's use the formula from 2zmiany.txt
            coherence = 1.0 / (1.0 + (variance * 0.005))
            
            # Threshold for field creation
            if coherence > 0.60: # Lowered slightly for easier testing (was 0.75)
                field = CoherenceField(
                    cluster_id=f"{zone[:3]}_C2",
                    zone=zone,
                    center=(center_x, center_y),
                    field_strength=round(coherence, 2),
                    field_radius=96.0, # From report
                    coherence_mean=coherence,
                    variance="low"
                )
                self.active_fields.append(field)



    def get_zone_readiness_profile(self, zone_name: str) -> Dict[str, float]:
        """Aggregate action profiles for all NPCs in a zone."""
        zone_npcs = [n for n in self.npcs if n.zone == zone_name]
        if not zone_npcs:
            return {
                "exchange_volume": 0,
                "commitment_depth": 0,
                "risk_appetite": 0,
                "stress_resilience": 0,
                "npc_count": 0,
            }
        return {
            "exchange_volume": sum(n.npc_action_profile["exchange_actions"] for n in zone_npcs),
            "commitment_depth": sum(n.npc_action_profile["long_term_commitments"] for n in zone_npcs),
            "risk_appetite": sum(n.npc_action_profile["risk_absorptions"] for n in zone_npcs),
            "stress_resilience": sum(n.npc_action_profile["stress_endured"] for n in zone_npcs),
            "npc_count": len(zone_npcs),
        }

# ---------- RENDERING ----------

zone_angles = {
    "SCIENCE": 0,
    "TRADE": math.pi/2,
    "DEVELOPMENT": math.pi,
    "FLEX": 3*math.pi/2
}

def draw_hexagon(surface, x, y, size, color):
    points = [(x + math.cos(math.pi/3 * i) * size,
               y + math.sin(math.pi/3 * i) * size) for i in range(6)]
    pygame.draw.polygon(surface, color, points)

def draw_roof(surface, x, y, size, color):
    points = [(x - size, y), (x, y - size), (x + size, y)]
    pygame.draw.polygon(surface, color, points)

# --- Add these helper functions in the rendering section ---
def draw_small_hexagon(x, y, size, color):
    points = [(x + math.cos(math.pi/3 * i) * size,
               y + math.sin(math.pi/3 * i) * size) for i in range(6)]
    pygame.draw.polygon(screen, color, points)

def draw_small_diamond(x, y, size, color):
    points = [
        (x, y - size),
        (x + size, y),
        (x, y + size),
        (x - size, y)
    ]
    pygame.draw.polygon(screen, color, points)

def draw_zone_marker(zone_name, center_x, center_y):
    if zone_name == "SCIENCE":
        draw_small_hexagon(center_x, center_y, config["rendering"].get("zone_symbol_size", 15), TEAL)
    elif zone_name == "TRADE":
        draw_small_diamond(center_x, center_y, config["rendering"].get("zone_symbol_size", 15), TRADE_COLOR)
    elif zone_name == "DEVELOPMENT":
        pygame.draw.rect(screen, DEV_COLOR, (center_x-7, center_y-7, 14, 14))
    elif zone_name == "FLEX":
        pygame.draw.circle(screen, FLEX_COLOR, (int(center_x), int(center_y)), config["rendering"].get("zone_symbol_size", 15)//2)
    # Add more as needed


# --- For NPC positioning ---
zone_angles = {
    "SCIENCE": 0,
    "TRADE": math.pi/2,
    "DEVELOPMENT": math.pi,
    "FLEX": 3*math.pi/2
}

# Replace or modify the calculate_npc_position function
# Define calculate_npc_position BEFORE you use it
def calculate_npc_position(zone, idx, total, center_x, center_y):
    # Circular layout - each zone gets a quadrant
    zone_angles = {
        "SCIENCE": 0,           # Right (East)
        "TRADE": math.pi/2,     # Bottom (South) 
        "DEVELOPMENT": math.pi, # Left (West)
        "FLEX": 3*math.pi/2     # Top (North)
    }
    
    base_angle = zone_angles[zone]
    spread = math.pi/4  # 45-degree spread per zone
    radius = 400  # Distance from center
    
    # Spread NPCs within the zone's arc
    if total > 1:
        angle = base_angle - spread/2 + (idx/(total-1)) * spread
    else:
        angle = base_angle
    
    x = center_x + math.cos(angle) * radius
    y = center_y + math.sin(angle) * radius
    
    return x, y

# ---------- INITIALIZATION ----------
worlds = []
territories = config["map"]["territories"]

print(f"Creating worlds from {len(territories)} territories...")

# In the initialization loop where you create NPCs
for terr in territories:
    world = World()
    home_x = WIDTH * terr["center"][0]
    home_y = HEIGHT * terr["center"][1]
    world.home_x = int(home_x)
    world.home_y = int(home_y)
    
    print(f"  Creating world at ({home_x}, {home_y})")
    
    # In your initialization code, use get_npc_position instead:
    # Then later in your initialization code...
    for zone_info in ZONES:
        count = zone_info["npc_count"]
        
        for j in range(count):
            # Debug: print what function you're using
            x, y = get_npc_position(
                zone_info["name"], 
                j, 
                count, 
                home_x, 
                home_y
            )
            
            print(f"  NPC {zone_info['name'][:3]}{j}: position ({x:.1f}, {y:.1f})")
            
            npc = NPC(
                name=f"{zone_info['name'][:3]}{j}",
                zone=zone_info["name"],
                personality=Personality.random(),
                x=x, y=y
            )
            npc.station_x = x # Store their assigned station
            npc.station_y = y
            
            # Assign a random rest spot near home (scattered village)
            angle_rest = random.uniform(0, 2*math.pi)
            dist_rest = random.uniform(30, 120) # 30-120px from center
            npc.rest_x = home_x + math.cos(angle_rest) * dist_rest
            npc.rest_y = home_y + math.sin(angle_rest) * dist_rest
            
            npc.shift = j % 2 # ALTERNATING SHIFTS
            npc.is_ghost = True  # <--- Make all NPCs ghosts at start
            world.npcs.append(npc)
        
    worlds.append((world, home_x, home_y))
    print(f"    Total NPCs in world: {len(world.npcs)}")

print(f"Total worlds created: {len(worlds)}")
print("=== CONFIG CHECK ===")
print(f"Territories in config: {len(territories)}")
for i, t in enumerate(territories):
    print(f"  {i}: {t['name']} at {t['center']}")

print(f"\nWorlds created: {len(worlds)}")
print(f"Zones defined: {len(ZONES)}")

# Also check each world:
for i, (world, x, y) in enumerate(worlds):
    print(f"\nWorld {i}:")
    print(f"  Position: ({x:.0f}, {y:.0f})")
    print(f"  NPC count: {len(world.npcs)}")
    # Count by zone
    zone_counts = {}
    for npc in world.npcs:
        zone_counts[npc.zone] = zone_counts.get(npc.zone, 0) + 1
    print(f"  Zone distribution: {zone_counts}")

current_world_idx = 0
time_counter = 0.0
paused = False

show_conflict_area = False

shared_zone_active = False

# Add at the top, after other control flags
show_npc_inspector = False
inspector_zone_idx = None  # None = all zones, or 0–3 for zone filter

print(f"DEBUG: Territories count: {len(territories)}")
print(f"DEBUG: Worlds count: {len(worlds)}")
print(f"DEBUG: Territory names: {[t['name'] for t in territories]}")

print(f"DEBUG: TAB pressed. Current idx: {current_world_idx}")
print(f"DEBUG: Worlds length: {len(worlds)}")
print(f"DEBUG: Territories length: {len(territories)}")
print(f"DEBUG: New idx: {current_world_idx}")

# NPC Inspector header explanations
NPC_INSPECTOR_HEADER_EXPLANATIONS = {
    "ID": "NPC's unique identifier (npc.id). Distinguishes individuals.",
    "Zone": "Zone the NPC currently belongs to (npc.zone). Functional area.",
    "Ex": "Exchange actions (npc_action_profile['exchange_actions']). Number of trades performed.",
    "Lt": "Long-term commitments (npc_action_profile['long_term_commitments']). Persistence in zone.",
    "Ri": "Risk absorptions (npc_action_profile['risk_absorptions']). Times NPC absorbed risk.",
    "St": "Stress endured (npc_action_profile['stress_endured']). Actions performed under stress.",
    "Needs": "Current basic needs value (npc.basic_needs). Well-being and survival status.",
    "$": "Current money (npc.money). Economic resources.",
    "Mood": "Current mood (npc.mood). Emotional state, affects decision-making.",
    "Health": "Current health (npc.health). Physical condition, impacts productivity.",
    "Energy": "Current energy (npc.energy). Stamina, influences ability to act."
}

def get_zone_action_summary(npcs):
    summary = {
        "exchange_actions": 0,
        "long_term_commitments": 0,
        "risk_absorptions": 0,
        "stress_endured": 0,
    }
    for npc in npcs:
        for k in summary:
            summary[k] += npc.npc_action_profile[k]
    return summary

def get_zone_position(zone_name):
    # Place the aggregate at the same position as the zone marker
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    angle = zone_angles[zone_name]
    radius = 250  # Match your marker radius
    x = center_x + math.cos(angle) * radius
    y = center_y + math.sin(angle) * radius
    return int(x), int(y)

def draw_mini_status_bar(x, y, health, mood):
    # Draw two small bars for health and mood
    bar_w, bar_h = 40, 6
    # Health bar (green to red)
    health_color = (int(255 * (1 - health)), int(255 * health), 0)
    pygame.draw.rect(screen, (50, 50, 50), (x - bar_w//2, y, bar_w, bar_h))
    pygame.draw.rect(screen, health_color, (x - bar_w//2, y, int(bar_w * health), bar_h))
    # Mood bar (yellow to blue)
    mood_color = (255, 255, int(255 * (1 - mood)))
    pygame.draw.rect(screen, (50, 50, 50), (x - bar_w//2, y + bar_h + 2, bar_w, bar_h))
    pygame.draw.rect(screen, mood_color, (x - bar_w//2, y + bar_h + 2, int(bar_w * mood), bar_h))

def draw_zone_aggregate(zone_name, npcs):
    count = len(npcs)
    avg_health = sum(n.health for n in npcs) / count
    avg_mood = sum(n.mood for n in npcs) / count

    x, y = get_zone_position(zone_name)
    radius = min(100, 20 + count * 0.5)

    # Color based on average health
    if avg_health > 0.7:
        color = COLORS[zone_name] if zone_name in COLORS else GREEN
    elif avg_health > 0.4:
        color = YELLOW_TEXT
    else:
        color = RED_TEXT

    pygame.draw.circle(screen, color, (x, y), int(radius))

    # Draw count label
    count_text = font_small.render(f"{zone_name}: {count}", True, WHITE)
    screen.blit(count_text, (x - count_text.get_width()//2, y - 10))

    # Draw mini health/mood bars
    draw_mini_status_bar(x, y + 15, avg_health, avg_mood)

def draw_zone_with_hover(zone_name, npc_count, position):
    draw_zone_label(zone_name, npc_count, position)
    if mouse_over_zone(position):
        show_zone_tooltip(zone_name, get_zone_stats(zone_name))

# ---------- MAIN LOOP ----------
dream_mode = False # Start with HUD visible
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.KEYDOWN:
            # Space - Pause
            if event.key == pygame.K_SPACE:
                paused = not paused
                print(f"DEBUG: Paused = {paused}")
            
            # Tab - Switch territories
            elif event.key == pygame.K_TAB:
                if not shared_zone_active:
                    current_world_idx = (current_world_idx + 1) % 3
                    print(f"DEBUG: Switched to territory {current_world_idx}")
            
            # B - NPC Inspector
            elif event.key == pygame.K_b:
                show_npc_inspector = not show_npc_inspector
                print(f"DEBUG: NPC Inspector = {show_npc_inspector}")
            
            # C - Shared Zone (disabled for now)
            elif event.key == pygame.K_c:
                show_conflict_area = not show_conflict_area
                print(f"DEBUG: Conflict Area = {show_conflict_area}")
            
            # Number keys for filtering inspector
            elif show_npc_inspector:
                if event.key == pygame.K_1:
                    inspector_zone_idx = 0
                    print(f"DEBUG: Filtering to zone 0")
                elif event.key == pygame.K_2:
                    inspector_zone_idx = 1
                elif event.key == pygame.K_3:
                    inspector_zone_idx = 2
                elif event.key == pygame.K_4:
                    inspector_zone_idx = 3
                elif event.key == pygame.K_0:
                    inspector_zone_idx = None
                    print(f"DEBUG: Showing all zones")
            
            # Q - Dream Mode (Toggle HUD)
            elif event.key == pygame.K_q:
                dream_mode = not dream_mode

    # === INDENT EVERYTHING FROM HERE ===
    if not paused:
        time_counter += 0.016
        for w, _, _ in worlds:
            w.step()
        global_tick += 1
        if global_tick % config["world_cycles"]["ticks_per_cycle"] == 0:
            global_cycle += 1

    screen.fill(BLACK)

    world, home_x, home_y = worlds[current_world_idx]
    
    # --- NPC Inspector Overlay ---
    if show_npc_inspector:
        # Panel setup
        panel_w, panel_h = 900, 850
        panel_x, panel_y = WIDTH // 2 - panel_w // 2, HEIGHT // 2 - panel_h // 2
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((30, 30, 30, 220))

        # Title
        title = font_header.render("NPC Inspector", True, YELLOW_TEXT)
        panel_surf.blit(title, (panel_w//2 - title.get_width()//2, 10))
        
        # Column headers
        headers = [
            "ID", "Zone", "Ex", "Lt", "Ri", "St",
            "Needs", "$", "Mood", "Health", "Energy"
        ]
        # 30% wider columns
        header_text = font_small.render(
            "{:<8} {:<13} {:<11} {:<11} {:<11} {:<11} {:<15} {:<15} {:<13} {:<13} {:<13}".format(*headers),
            True, RED_TEXT
        )
        panel_surf.blit(header_text, (20, 50))

        npcs = world.npcs
        if inspector_zone_idx is not None:
            zone_name = [z["name"] for z in ZONES][inspector_zone_idx]
            npcs = [n for n in npcs if n.zone == zone_name]

        max_lines = 15
        for i, npc in enumerate(npcs[:max_lines]):
            line = (
               "{:<8} {:<13} {:<11} {:<11} {:<11} {:<11} {:<15} {:<15} {:<13} {:<13} {:<13}".format(
                    npc.id,
                    npc.zone[:3],
                    int(npc.npc_action_profile['exchange_actions']),
                    int(npc.npc_action_profile['long_term_commitments']),
                    int(npc.npc_action_profile['risk_absorptions']),
                    int(npc.npc_action_profile['stress_endured']),
                    f"{npc.basic_needs:5.1f}",
                    f"{npc.money:5.1f}",
                    f"{npc.mood:4.2f}",
                    f"{npc.health:4.2f}",
                    f"{npc.energy:5.1f}"
                )
            )
            line_render = font_small.render(line, True, WHITE)
            panel_surf.blit(line_render, (20, 80 + i * 30))

        # Instructions
        instr = "B: Toggle | 1–4: Filter zone | 0: All zones"
        instr_render = font_tiny.render(instr, True, YELLOW_TEXT)
        panel_surf.blit(instr_render, (20, panel_h - 30))

        # --- Explanations ---
        lines_rendered = min(len(npcs), max_lines)
        explanation_y = 80 + lines_rendered * 30 + 30
        
        for key, explanation in NPC_INSPECTOR_HEADER_EXPLANATIONS.items():
            header_render = font_small.render(f"{key}:", True, RED_TEXT)
            expl_render = font_small.render(f" {explanation}", True, GREEN_TEXT)
            panel_surf.blit(header_render, (20, explanation_y))
            panel_surf.blit(expl_render, (20 + header_render.get_width(), explanation_y))
            explanation_y += 30

        screen.blit(panel_surf, (panel_x, panel_y))
        pygame.display.flip()
        clock.tick(FPS)
        continue  # Skip normal rendering

    # Define center_x here, after the inspector check
    center_x, center_y = WIDTH // 2, HEIGHT // 2 
    offset_x = center_x - home_x
    offset_y = center_y - home_y

    # Draw connections
    if not config["rendering"].get("use_minimal_connections", False):
        # Draw all connections as before
        for npc in world.npcs:
            color = GREEN if npc.health > 0.7 else (200, 150, 50) if npc.health > 0.4 else RED_TEXT
            pygame.draw.line(screen, color, (center_x, center_y), (npc.x + offset_x, npc.y + offset_y), 1)
            nx = npc.x + offset_x
            ny = npc.y + offset_y
    else:
        # Only draw connections for a subset, e.g., one per zone
        for zone in ZONE_SECTORS:
            npcs_in_zone = [n for n in world.npcs if n.zone == zone]
            if npcs_in_zone:
                npc = npcs_in_zone[0]  # or random.choice(npcs_in_zone)
                color = GREEN if npc.health > 0.7 else (200, 150, 50) if npc.health > 0.4 else RED_TEXT
                pygame.draw.line(screen, color, (center_x, center_y), (npc.x + offset_x, npc.y + offset_y), 1)

    # Draw particles
    for npc in world.npcs:
        color = npc.get_color()
        if len(color) == 4: # GHOST DETECTED
            ghost_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(ghost_surf, color, (6, 6), 5)
            screen.blit(ghost_surf, (int(npc.x)-6, int(npc.y)-6))
        else:
            pygame.draw.circle(screen, color, (int(npc.x), int(npc.y)), 5)

        for i in range(config["rendering"]["particles"]["count"]):
            t = (time_counter * config["rendering"]["particles"]["speed"] + i * 0.1 + npc.id * 0.1) % 1.0
            px = center_x + (npc.x + offset_x - center_x) * t
            py = center_y + (npc.y + offset_y - center_y) * t
            alpha = int((math.sin(time_counter * 2 + i) * 0.5 + 0.5) * config["rendering"]["particles"]["alpha_base"])
            size = max(1, int(math.sin(time_counter * 3 + i) * config["rendering"]["particles"]["size_variation"] + config["rendering"]["particles"]["size_base"]))
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            color = LIME if npc.mood > 0.6 else (255, 200, 100, alpha) if npc.mood > 0.4 else (255, 100, 100, alpha)
            pygame.draw.rect(surf, (*color[:3], alpha), surf.get_rect())
            screen.blit(surf, (px - size//2, py - size//2))

    # Draw HOME
    draw_hexagon(screen, center_x, center_y, config["rendering"]["home_size"], ORANGE)
    draw_roof(screen, center_x, center_y, config["rendering"]["home_size"], RED_ORANGE)
    home_label = f"HOME {current_world_idx + 1}"
    home_text = font_small.render(home_label, True, WHITE)
    screen.blit(home_text, (center_x - home_text.get_width()//2, center_y - 6))

    # --- DRAW ACTIVE FIELDS (RESONANCE) ---
    for field in world.active_fields:
        # Draw a delicate circle around the cluster
        fx, fy = field.center[0] + offset_x, field.center[1] + offset_y
        
        # Pulsating effect
        pulse = (math.sin(time_counter * 3) + 1) * 0.5 # 0 to 1
        alpha = int(50 + 50 * pulse)
        
        field_surf = pygame.Surface((int(field.field_radius * 2), int(field.field_radius * 2)), pygame.SRCALPHA)
        color = COLORS.get(field.zone + "_COLOR", WHITE)
        pygame.draw.circle(field_surf, (*color, alpha), (int(field.field_radius), int(field.field_radius)), int(field.field_radius), 2)
        # Fill slightly
        pygame.draw.circle(field_surf, (*color, 10), (int(field.field_radius), int(field.field_radius)), int(field.field_radius))
        
        screen.blit(field_surf, (int(fx - field.field_radius), int(fy - field.field_radius)))
        
        # Label
        label = font_tiny.render(f"RESONANCE {field.field_strength}", True, color)
        screen.blit(label, (fx - label.get_width()//2, fy - field.field_radius - 10))

    # Draw NPCs
    for zone_name in zone_angles:
        npcs_in_zone = [n for n in world.npcs if n.zone == zone_name]
        if len(npcs_in_zone) > MAX_NPCS_FOR_INDIVIDUAL_DISPLAY:
            draw_zone_aggregate(zone_name, npcs_in_zone)
        else:
            for npc in npcs_in_zone:
                color = npc.get_color()
                nx, ny = npc.x + offset_x, npc.y + offset_y
                if len(color) == 4:  # Ghost (has alpha)
                    ghost_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
                    pygame.draw.circle(ghost_surf, color, (6, 6), 5)
                    screen.blit(ghost_surf, (int(nx) - 6, int(ny) - 6))
                else:
                    pygame.draw.circle(screen, color, (int(nx), int(ny)), 5)
                # ... (draw integrity bar, stats, rival, etc. as before)
                
                # Integrity bar
                bar_w, bar_h = config["rendering"]["integrity_bar"]["width"], config["rendering"]["integrity_bar"]["height"]
                bar_x = nx - bar_w//2
                bar_y = ny - 28
                pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
                integrity_w = int((npc.zone_integrity / 100) * bar_w)
                good = config["rendering"]["integrity_bar"]["good_threshold"]
                warn = config["rendering"]["integrity_bar"]["warn_threshold"]
                bar_color = (100, 200, 100) if npc.zone_integrity > good else (200, 150, 50) if npc.zone_integrity > warn else (200, 50, 50)
                pygame.draw.rect(screen, bar_color, (bar_x, bar_y, integrity_w, bar_h))
                
                # NPC stats
                zone_text = font_small.render(npc.zone[:3], True, WHITE)
                stats = f"N:{int(npc.basic_needs)} ${int(npc.money)}"
                stats_text = font_tiny.render(stats, True, WHITE)
                screen.blit(zone_text, (nx - zone_text.get_width()//2, ny - 18))
                screen.blit(stats_text, (nx - stats_text.get_width()//2, ny + 5))
                
                if npc.rival_id:
                    # Draw small red dot for rival
                    pygame.draw.circle(screen, RED_TEXT, (int(nx + 8), int(ny - 8)), 4)

    # Draw zone building levels (grouped by zone type)
    zone_positions = {}
    for npc in world.npcs:
        if npc.zone not in zone_positions:
            zone_positions[npc.zone] = []
            # Draw line from zone center to NPC
        # Get zone center position
        zone_center_x = home_x + offset_x
        zone_center_y = home_y + offset_y

        zone_positions[npc.zone].append((npc.x + offset_x, npc.y + offset_y))
    for zone_name, positions in zone_positions.items():
        level = world.zone_building_levels[zone_name]
        stars = "⭐" * level
        label = font_tiny.render(f"{zone_name} {stars} (L{level})", True, YELLOW_TEXT)
        margin = 140
        # "Flattened square" layout
        if zone_name == "SCIENCE":
            label_x = margin
            label_y = HEIGHT // 2 - label.get_height() // 2
        elif zone_name == "DEVELOPMENT":
            label_x = WIDTH - label.get_width() - margin
            label_y = HEIGHT // 2 - label.get_height() // 2
        elif zone_name == "TRADE":
            label_x = WIDTH // 2 - label.get_width() // 2
            label_y = HEIGHT - label.get_height() - margin
        elif zone_name == "FLEX":
            label_x = WIDTH // 2 - label.get_width() // 2
            label_y = margin - 30  # Move FLEX label higher to avoid collis
        else:
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            label_x = avg_x - label.get_width() // 2
            label_y = avg_y - label.get_height() // 2
        screen.blit(label, (label_x, label_y))
        # After zone level display, show readiness
        profile = world.get_zone_readiness_profile(zone_name)
        if level < 5:
            if zone_name == "SCIENCE":
                commitment_ratio = profile["commitment_depth"] / (profile["risk_appetite"] + 1)
                resilience = profile["stress_resilience"] / (profile["npc_count"] + 0.01)
                status = f"Commit:{commitment_ratio:.1f} Resilience:{resilience:.0f}"
                color = GREEN_TEXT if commitment_ratio > 0.6 and resilience > 30 else YELLOW_TEXT
            elif zone_name == "DEVELOPMENT":
                commit_per_npc = profile["commitment_depth"] / (profile["npc_count"] + 0.01)
                status = f"Commitment/NPC: {commit_per_npc:.0f}"
                color = GREEN_TEXT if commit_per_npc > 40 else YELLOW_TEXT
            elif zone_name == "TRADE":
                exchange_per_npc = profile["exchange_volume"] / (profile["npc_count"] + 0.01)
                status = f"Exchange/NPC: {exchange_per_npc:.0f}"
                color = GREEN_TEXT if exchange_per_npc > 50 else YELLOW_TEXT
            elif zone_name == "FLEX":
                risk_per_npc = profile["risk_appetite"] / (profile["npc_count"] + 0.01)
                resilience = profile["stress_resilience"] / (profile["npc_count"] + 0.01)
                status = f"Risk:{risk_per_npc:.0f} Tough:{resilience:.0f}"
                color = GREEN_TEXT if risk_per_npc > 30 and resilience > 20 else YELLOW_TEXT
            readiness_text = font_tiny.render(status, True, color)
            screen.blit(readiness_text, (label_x, label_y + 20))

                # --- ZONE MARKERS AT MIDDLE EDGES ---
        for zone_name, (zx, zy) in ZONE_CENTERS.items():
            if zone_name == "HOUSING":
                continue  # Skip home, only show real zones
            sx = zx + offset_x
            sy = zy + offset_y
            draw_zone_marker(zone_name, sx, sy)
            npcs_in_zone = [n for n in world.npcs if n.zone == zone_name]
            if len(npcs_in_zone) > 0:
                draw_zone_aggregate(zone_name, npcs_in_zone)

        # Territory name header (centered at top)
        if len(territories) > 0:
            terr_text = font_header.render(f"ZONE {current_world_idx + 1}", True, (50, 50, 50))
            screen.blit(terr_text, (WIDTH//2 - terr_text.get_width()//2, 10))

        # Central Header (Global Stats)
        header_y = HEIGHT - 30
        global_stats = f"CYCLE {global_cycle} | TICK {global_tick} | Pool {int(world.pool['basic_needs'])}"
        header = font_small.render(global_stats, True, (100, 100, 100))
        screen.blit(header, (WIDTH//2 - header.get_width()//2, header_y))

        # Event display (centered)
        if world.event:
            event_text = font_large.render(
                f"⚡ {world.event['name']} in {world.event['zone']}",
                True, WHITE
            )
            screen.blit(event_text, (WIDTH//2 - event_text.get_width()//2, HEIGHT//2 - 100))
            
            
            # 3. Action Stats (The Profile)
            ap = get_zone_action_summary(npcs_in_zone)
            
          
            
        # Central Header (Global Stats)
        header_y = HEIGHT - 30
        global_stats = f"CYCLE {global_cycle} | TICK {global_tick} | Pool {int(world.pool['basic_needs'])}"
        header = font_small.render(global_stats, True, (100, 100, 100))
        screen.blit(header, (WIDTH//2 - header.get_width()//2, header_y))

        # Event display
        if world.event:
            event_text = font_large.render(
                f"⚡ {world.event['name']} in {world.event['zone']}",
                True, WHITE
            )
            # Center of screen
            screen.blit(event_text, (WIDTH//2 - event_text.get_width()//2, HEIGHT//2 - 100))


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
