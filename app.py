import sys
import os
from streamlit.web import cli as stcli

# Application execution config
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
             sys.argv = ["streamlit", "run", __file__]
             sys.exit(stcli.main())
    except ImportError:
         if "streamlit" not in sys.argv[0]:
            sys.argv = ["streamlit", "run", __file__]
            sys.exit(stcli.main())

# Imports
import streamlit as st
import pandas as pd
import json

import joblib

try:
    from main import load_databases, assess_risk, physics_gatekeeper, build_single_interface_df, Metal
except ImportError:
    st.error("CRITICAL ERROR: Could not import 'main.py'. Ensure it is updated.")
    st.stop()

# --- LOAD ML MODELS ---
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('filler_safety_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except Exception as e:
        st.error(f"Could not load ML models! Ensure .pkl files are in the folder. Error: {e}")
        return None, None

ml_model, ml_features = load_ml_assets()

# PAGE CONFIG 
st.set_page_config(page_title="Laser Brazing Physics Simulator", layout="wide")

if 'custom_metals' not in st.session_state:
    st.session_state.custom_metals = []

# DATA LOADING 
@st.cache_data
def get_physics_data():
    elements_db, metals, fillers = load_databases()
    return elements_db, metals, fillers

elements_db, static_metals, fillers_data = get_physics_data()

if elements_db is None:
    st.error("Data loading failed. Check your 'data' folder.")
    st.stop()

all_metals = static_metals + st.session_state.custom_metals
metal_names = [m.name for m in all_metals]

# Sidebar configuration: Alloy selection
with st.sidebar:
    st.header("Material Designer")
    st.markdown("Create a new alloy to test its compatibility.")
    
    new_name = st.text_input("Alloy Name", value="Experimental-X")
    
    st.subheader("Composition (wt%)")
    new_comp = {}
    
    priority_elements = ["Fe", "Al", "Si", "Mg", "Cu", "Zn"]
    
    with st.expander("Primary Elements", expanded=True):
        for el in priority_elements:
            if el in elements_db:
                val = st.number_input(f"{el}", 0.0, 100.0, 0.0, step=0.5, key=f"p_{el}")
                if val > 0: new_comp[el] = val
    
    current_total = sum(new_comp.values())
    st.progress(min(current_total/100, 1.0))
    
    if st.button("Create & Add to List"):
        if 90.0 <= current_total <= 110.0:
            new_metal = Metal(new_name, new_comp, elements_db)
            st.session_state.custom_metals.append(new_metal)
            st.success(f"Added {new_name}!")
            st.rerun()
        else:
            st.warning("Composition must be close to 100%.")

# --- USER INPUT SECTION ---
st.title("Laser Brazing: Hybrid ML Predictor")

c1, c2, c3 = st.columns(3)
with c1:
    m1_name = st.selectbox("Select Base Metal 1", [m.name for m in all_metals], index=0)
with c2:
    # Safely select the index for Base Metal 2
    def_idx = min(3, len(all_metals)-1) if len(all_metals) > 0 else 0
    m2_name = st.selectbox("Select Base Metal 2", [m.name for m in all_metals], index=def_idx)
with c3:
    # THE NEW TEMPERATURE INPUT
    brazing_temp = st.slider("Laser Process Temp (K)", min_value=800, max_value=1800, value=1050, step=10)

m1 = next(m for m in all_metals if m.name == m1_name)
m2 = next(m for m in all_metals if m.name == m2_name)

# Process Evaluation: System Diagnosis
st.subheader("1. Base Metal Interface Diagnosis")
risk, reasons = assess_risk(m1, m2)
r1, r2 = st.columns([1, 2])
r1.metric("Risk Score", f"{risk}/12")

with r2:
    if risk >= 6: 
        st.error("High Mismatch: Filler selection is critical to prevent joint failure.")
    else: 
        st.success("Low Mismatch.")
        
    if reasons:
        st.info("**Identified Risk Factors:**\n" + "\n".join([f"- {r}" for r in reasons]))

# Process Evaluation: Framework Recommendations
st.subheader("2. Framework Output")

if ml_model and ml_features:
    # Step 2: Gatekeeper eliminates bad physics
    surviving_fillers = physics_gatekeeper(m1, m2, fillers_data, brazing_temp)
    
    if len(surviving_fillers) == 0:
        st.error(f"Thermal Failure: No compatible fillers identified at {brazing_temp}K.")
    else:
        st.write(f"*(Physics Gatekeeper: {len(surviving_fillers)}/{len(fillers_data)} fillers survived the thermal constraints)*")
        
        # Step 3: Format data bridge for Dual-Interface Prediction
        from main import build_single_interface_df
        
        # We query the exact physical state of both distinct interfaces
        df_m1 = build_single_interface_df(m1, surviving_fillers, brazing_temp, ml_features)
        df_m2 = build_single_interface_df(m2, surviving_fillers, brazing_temp, ml_features)
        
        # Step 4: ML Model predicts safety scores for each interface separately
        predictions_m1 = ml_model.predict(df_m1)
        predictions_m2 = ml_model.predict(df_m2)
        

        from main import calculate_imc_risk
        needs_kinetic_shield = calculate_imc_risk(m1, m2) > 2.0
        
        # Step 5: Rank the results 
        ranked_results = []
        for idx, f in enumerate(surviving_fillers):
            
            base_score = min(predictions_m1[idx], predictions_m2[idx])
            
            # Apply the empirical Silicon Boost
            if needs_kinetic_shield:
                f_si = f.composition.get("Si", 0)
                f_al = f.composition.get("Al", 0)
                # Boost Al-based fillers by 1 full score point per 1% of Silicon
                if f_al > 50 and f_si > 0:
                    base_score += (f_si * 1.0) 
            
            ranked_results.append({"filler": f, "safety_score": base_score})
            
        # Sort from highest score to lowest
        ranked_results = sorted(ranked_results, key=lambda x: x["safety_score"], reverse=True)
        
        # Display the Top 3
        for i, result in enumerate(ranked_results[:3]):
            f = result["filler"]
            score = result["safety_score"]
            
            with st.expander(f"Rank #{i+1}: {f.name} (Safety Score: {score:.1f}/100)", expanded=(i==0)):
                st.write(f"**Melting Point:** {f.melting_point:.0f} °C ({f.melting_point + 273.15:.0f} K)")
                st.write(f"**Key Chemistry:** {', '.join([f'{k}: {v}%' for k, v in f.composition.items() if v > 1])}")

        # ---------------------------------------------------------
        # 3. Generative Alloy Module
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Theoretical Filler Optimization")
        st.markdown("*Iterative optimization of custom chemical compositions...*")
        
        import random
        # elements_db is already available globally
        
        is_steel_joint = (m1.composition.get("Fe", 0) > 50) and (m2.composition.get("Fe", 0) > 50)
        
        theoretical_fillers = []
        for i in range(100):
            # Directed random generation for realism
            if is_steel_joint:
                # Copper based for steel
                cu = random.uniform(85.0, 99.0)
                si = random.uniform(0.5, 10.0)
                mn = random.uniform(0.0, 5.0)
                total = cu + si + mn
                comp = {"Cu": (cu/total)*100, "Si": (si/total)*100, "Mn": (mn/total)*100}
            else:
                # Aluminum based for Al-Fe or Al-Al
                al = random.uniform(70.0, 95.0)
                si = random.uniform(2.0, 15.0)
                mg = random.uniform(0.0, 5.0)
                cu = random.uniform(0.0, 5.0)
                total = al + si + mg + cu
                comp = {"Al": (al/total)*100, "Si": (si/total)*100, "Mg": (mg/total)*100, "Cu": (cu/total)*100}

            tf = Metal(f"Theoretical Alloy #{i+1}", comp, elements_db)
            theoretical_fillers.append(tf)

        # Apply thermal constraints
        surviving_theoretical = physics_gatekeeper(m1, m2, theoretical_fillers, brazing_temp)
        
        if len(surviving_theoretical) > 0:
            df_th_m1 = build_single_interface_df(m1, surviving_theoretical, brazing_temp, ml_features)
            df_th_m2 = build_single_interface_df(m2, surviving_theoretical, brazing_temp, ml_features)
            
            p_th_m1 = ml_model.predict(df_th_m1)
            p_th_m2 = ml_model.predict(df_th_m2)
            
            best_th = None
            best_th_score = -999
            
            for j, f in enumerate(surviving_theoretical):
                th_score = min(p_th_m1[j], p_th_m2[j])
                
                # Empirical Silicon factor
                if needs_kinetic_shield:
                    f_si = f.composition.get("Si", 0)
                    f_al = f.composition.get("Al", 0)
                    if f_al > 50 and f_si > 0:
                        th_score += (f_si * 1.0)
                        
                if th_score > best_th_score:
                    best_th_score = th_score
                    best_th = f
            
            # Compare to standard
            if len(ranked_results) > 0:
                top_standard_score = ranked_results[0]['safety_score']
                if best_th_score > top_standard_score:
                    st.success(f"**Optimal Theoretical Composition Identified.** Estimated Score: **{best_th_score:.1f}/100** (Exceeds standard filler maximum by {best_th_score - top_standard_score:.1f} points).")
                else:
                    st.info(f"**Optimum Synthesized Filler:** Model predicts a maximum score of **{best_th_score:.1f}/100**. Existing standard fillers perform equally or better.")
            
            st.write(f"**Recommended Composition:** {', '.join([f'{k}: {v:.1f}%' for k, v in best_th.composition.items() if v > 0.5])}")
            st.write(f"*(Calculated Melting Point: {best_th.melting_point:.0f} °C)*")
        else:
            st.warning("No custom alloy combination could be formulated that satisfies the current thermal constraints. Adjust brazing temperature.")