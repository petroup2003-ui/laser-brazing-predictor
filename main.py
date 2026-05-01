import pandas as pd
import json
import os
import sys

# 1. Path Implementation


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

# 2. Structure Definitions 

class Metal:
    def __init__(self, name, composition, elements_db, hardcoded_mp=None):
        self.name = name
        self.composition = composition
        self.hardcoded_mp = hardcoded_mp
        
        # Property assignment and calculation bounds:
        self.melting_point = self.hardcoded_mp if self.hardcoded_mp is not None else self._calculate_property(elements_db, "mp", default_val=1000)
        self.cte = self._calculate_property(elements_db, "cte", default_val=15e-6)

    def _calculate_property(self, db, key, default_val):
        """
        Implementation of the Rule of Mixtures:
        """
        total_val = 0
        total_wt = sum(self.composition.values())
        
        if total_wt == 0: return default_val

        for el, wt in self.composition.items():
            if el in db:
                pure_prop = db[el].get(key, default_val)
                total_val += (wt / total_wt) * pure_prop
                
        return total_val

class Filler(Metal):
 
    pass

# 3. PHYSICS ENGINE

def calculate_imc_risk(m1, m2):
 
    # Get elemental concentrations (default to 0 if missing)
    al_1, fe_1 = m1.composition.get("Al", 0), m1.composition.get("Fe", 0)
    al_2, fe_2 = m2.composition.get("Al", 0), m2.composition.get("Fe", 0)

    # (Al on side A * Fe on side B) + (Al on side B * Fe on side A)

    potential = (al_1 * fe_2) + (al_2 * fe_1)
    
    # Normalize potential to a 0-10 scale 
    return (potential / 10000) * 10

def assess_risk(m1, m2):
    """
    Generates the 0-12 Physics Risk Score based on 3 pillars:
    1. Chemical Incompatibility (IMCs)
    2. Thermal Expansion Mismatch (Stress)
    3. Melting Point Gap (Wettability)
    """
    score = 0
    reasons = []

    # A. Chemical Conflict (Intermetallic Risk)
    imc_score = calculate_imc_risk(m1, m2)
    if imc_score > 1.0:
        penalty = round(imc_score * 0.8) # Weighting factor
        score += penalty
        reasons.append(f"Chemical Incompatibility: High Fe-Al Reaction Potential (Risk Score: {penalty}/8)")

    # B. Thermal Expansion Mismatch (CTE)
    delta_cte = abs(m1.cte - m2.cte) * 1e6 # Convert to ppm
    if delta_cte > 10:
        score += 3
        reasons.append(f"Severe Thermal Stress: ΔCTE = {delta_cte:.1f} ppm/K (High risk of solidification cracking)")
    elif delta_cte > 5:
        score += 1
        reasons.append(f"Moderate Thermal Stress: ΔCTE = {delta_cte:.1f} ppm/K")

    # C. Melting Point Gap
    delta_tm = abs(m1.melting_point - m2.melting_point)
    if delta_tm > 400:
        score += 1
        reasons.append(f"Large Melting Point Gap: {delta_tm:.0f}°C (Risk of poor wetting/evaporation)")

    return min(12, score), reasons

# Physics Gatekeeper Module
def physics_gatekeeper(m1, m2, all_fillers, brazing_temp):
    """
    Eliminates fillers that violate fundamental thermal physics 
    and critical kinetic rules (like requiring Silicon for Al-Fe joints).
    """
    survivors = []
    lowest_base_mp = min(m1.melting_point, m2.melting_point)
    brazing_temp_c = brazing_temp - 273.15
    
    # Check if the joint creates a high risk of Fe-Al Intermetallics
    imc_score = calculate_imc_risk(m1, m2)
    needs_silicon = imc_score > 2.0
    
    # Check if it is a pure Steel-to-Steel joint
    is_steel_joint = (m1.composition.get("Fe", 0) > 50) and (m2.composition.get("Fe", 0) > 50)
    
    for f in all_fillers:
        # Rule 1: The laser temperature must be hot enough to melt the filler
        if brazing_temp_c < f.melting_point:
            continue
        
        # Rule 2: The filler must melt BEFORE the base metals melt (Brazing principle)
        # A 100 degree tolerance buffer added here to account for mathematical 
        # inaccuracies in the Rule of Mixtures for complex Stainless Steel fillers.
        if f.melting_point > (lowest_base_mp + 100):
            continue
            
        # Rule 3: If it's an Al-Fe joint, ONLY allow Aluminum fillers IF they have >= 5% Silicon.
        # (Skip this rule for alternative filler types)
        if needs_silicon:
            f_al = f.composition.get("Al", 0)
            f_si = f.composition.get("Si", 0)
            # If the filler is mostly Aluminum but lacks Silicon, it will cause brittle failure
            if f_al > 50 and f_si < 5.0:
                continue
                
        # Rule 4: If it is a Steel-to-Steel joint, absolutely ban Aluminum fillers (to prevent Fe-Al IMCs)
        if is_steel_joint:
            f_al = f.composition.get("Al", 0)
            if f_al > 50:
                continue
            
        survivors.append(f)
        
    return survivors

# Data Formatting Module
def build_single_interface_df(base_metal, surviving_fillers, temp_k, feature_names):
    """
    Formats the surviving fillers into the exact Pandas DataFrame 
    required by the loaded .pkl Machine Learning model, isolating ONE interface.
    """
    rows = []
    for f in surviving_fillers:
        row_data = {
            'Temp_K': temp_k,
            'Base_MP': base_metal.melting_point + 273.15,
            'Base_CTE': base_metal.cte,
            'Filler_MP': f.melting_point + 273.15
        }
        
        # Dynamically fill all the elemental weights (e.g., Base_Al_wt, Filler_Si_wt)
        for feature in feature_names:
            if feature.endswith('_wt'):
                parts = feature.split('_')
                prefix, element = parts[0], parts[1]
                
                if prefix == 'Base':
                    row_data[feature] = base_metal.composition.get(element, 0.0)
                elif prefix == 'Filler':
                    row_data[feature] = f.composition.get(element, 0.0)
                    
        rows.append(row_data)
        
    # Return a DataFrame with columns in the exact order the ML model expects
    return pd.DataFrame(rows, columns=feature_names)

#  4. DATA LOADING 

def load_databases():
    """
    Loads chemical composition databases and returns instantiated objects.
    """
    try:
        # Load Pure Elements 
        with open(os.path.join(DATA_PATH, "elements.json"), "r") as f:
            elements_db = json.load(f)

        # Load Metal 
        with open(os.path.join(DATA_PATH, "metals.json"), "r") as f:
            metals_data = json.load(f)
        
        # Load Filler 
        with open(os.path.join(DATA_PATH, "fillers.json"), "r") as f:
            fillers_data = json.load(f)

        # Turn data into objects
        metals = [Metal(m["name"], m["composition"], elements_db, m.get("mp")) for m in metals_data]
        fillers = [Filler(f["name"], f["composition"], elements_db, f.get("mp")) for f in fillers_data]
        
        return elements_db, metals, fillers
        
    except FileNotFoundError as e:
        return None, None, None