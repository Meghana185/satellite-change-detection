import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config
import utils
import predict
import change_analysis

# --- Caching the Model ---
@st.cache_resource
def load_model():
    """
    Loads the PyTorch model once and caches it to prevent reloading on every interaction.
    """
    if not os.path.exists(config.MODEL_SAVE_PATH):
        return None
    return predict.load_trained_model(config.MODEL_SAVE_PATH)

# --- Main App ---
def main():
    st.set_page_config(page_title="SatChange DL MVP", layout="wide")
    
    st.title("🛰️ Satellite Image Change Detection & Environmental Risk")
    st.markdown("Deep Learning MVP to detect land-cover changes and assess environmental risk.")
    
    # 1. Check for Model
    model = load_model()
    if model is None:
        st.error(f"⚠️ Model weights not found at `{config.MODEL_SAVE_PATH}`.")
        st.warning("Please run `python train_model.py` first to train and save the model.")
        return

    # 2. Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Allow tile size tweaking
    tile_size = st.sidebar.slider("Tile Size (px)", min_value=32, max_value=128, value=config.TILE_SIZE, step=16)
    
    st.sidebar.subheader("Upload Images")
    img1_file = st.sidebar.file_uploader("Image 1 (Older)", type=["png", "jpg", "jpeg"])
    year1 = st.sidebar.text_input("Year 1 Label", "2015")
    
    img2_file = st.sidebar.file_uploader("Image 2 (Newer)", type=["png", "jpg", "jpeg"])
    year2 = st.sidebar.text_input("Year 2 Label", "2025")
    
    # Demo fallback logic: if no file, we look for default ones if they exist, or just prompt.
    if not img1_file or not img2_file:
        st.info("Please upload both 'Before' and 'After' images in the sidebar to proceed.")
        # Optional: check if demo images exist in sample_images
        demo1_path = os.path.join("sample_images", "before.jpg")
        demo2_path = os.path.join("sample_images", "after.jpg")
        if os.path.exists(demo1_path) and os.path.exists(demo2_path):
            st.markdown("Or click below to use the demo images.")
            if st.button("Use Demo Images"):
                img1_file = demo1_path
                img2_file = demo2_path
        else:
            return

    # Ensure files are loaded before showing "Analyze"
    if img1_file and img2_file:
        if st.sidebar.button("Analyze Changes", type="primary"):
            run_analysis(img1_file, img2_file, year1, year2, model, tile_size)

def run_analysis(img1_file, img2_file, year1, year2, model, tile_size):
    """
    Executes the analysis pipeline and renders the dashboard.
    """
    with st.spinner("Loading and resizing images..."):
        # Load Image 1
        img1_array = utils.load_and_resize_image(img1_file)
        
        # Crop Image 1 to be an exact multiple of tile_size to avoid remainder pixels
        h, w = img1_array.shape[:2]
        h_crop = (h // tile_size) * tile_size
        w_crop = (w // tile_size) * tile_size
        img1_array = img1_array[:h_crop, :w_crop]
        
        # Load Image 2, explicitly resizing to match Image 1's cropped dimensions
        # target_size expects (width, height)
        target_size = (img1_array.shape[1], img1_array.shape[0]) 
        img2_array = utils.load_and_resize_image(img2_file, target_size=target_size)
        
    st.success("Images loaded successfully!")
    
    # --- Section 1: Original Images ---
    st.header("1. Original Satellite Imagery")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1_array, caption=f"{year1} (Before)", use_container_width=True)
    with col2:
        st.image(img2_array, caption=f"{year2} (After)", use_container_width=True)
        
    # --- Inference ---
    with st.spinner("Running Deep Learning Classification..."):
        map1 = predict.classify_tiles(img1_array, model, tile_size=tile_size)
        map2 = predict.classify_tiles(img2_array, model, tile_size=tile_size)
        
        color_map1 = predict.generate_color_map_image(map1, tile_size=tile_size)
        color_map2 = predict.generate_color_map_image(map2, tile_size=tile_size)
        
    # --- Section 2: Land-Cover Maps ---
    st.header("2. AI-Generated Land-Cover Maps")
    # Show legend
    legend_cols = st.columns(len(config.CATEGORY_COLORS))
    for i, (cat, color) in enumerate(config.CATEGORY_COLORS.items()):
        with legend_cols[i]:
            st.markdown(
                f"<div style='background-color:rgb{color}; padding:10px; text-align:center; color:white; font-weight:bold; border-radius:5px;'>{cat}</div>", 
                unsafe_allow_html=True
            )
    st.write("") # Spacer
            
    col1, col2 = st.columns(2)
    with col1:
        st.image(color_map1, caption=f"Land Cover {year1}", use_container_width=True)
    with col2:
        st.image(color_map2, caption=f"Land Cover {year2}", use_container_width=True)
        
    # --- Analysis ---
    with st.spinner("Computing change statistics and risk score..."):
        analysis_results = change_analysis.analyze_changes(map1, map2)
        changed_mask_2d = analysis_results['changed_tiles_mask']
        change_pp = analysis_results['change_in_pp']
        perc_before = analysis_results['percentages_before']
        perc_after = analysis_results['percentages_after']
        
        # Expand mask to pixel resolution
        expanded_mask = change_analysis.expand_mask_to_image_size(changed_mask_2d, tile_size=tile_size)
        
        # Ensure mask fits image just in case of rounding errors during tiling
        h, w = img2_array.shape[:2]
        expanded_mask = expanded_mask[:h, :w]
        
        overlay_img = utils.overlay_change_mask(img2_array, expanded_mask, alpha=0.5, color=(255, 0, 0))
        
        risk_score, risk_label, risk_explanation = change_analysis.calculate_environmental_risk(change_pp)

    # --- Section 3: Change Mask ---
    st.header("3. Highlighted Change Map")
    st.markdown("Areas highlighted in red show where the AI detected a change in the broad land-cover category between the two years.")
    st.image(overlay_img, caption="Changed Tiles Overlay (on newer image)", use_container_width=True)

    # --- Section 4: Data & Charts ---
    st.header("4. Area Distribution & Changes")
    
    # Prepare Dataframe for table and charts
    df = pd.DataFrame({
        "Category": list(config.CATEGORY_COLORS.keys()),
        f"{year1} Area (%)": [perc_before[k] for k in config.CATEGORY_COLORS.keys()],
        f"{year2} Area (%)": [perc_after[k] for k in config.CATEGORY_COLORS.keys()],
        "Change (pp)": [change_pp[k] for k in config.CATEGORY_COLORS.keys()]
    })
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df.style.format({f"{year1} Area (%)": "{:.1f}", f"{year2} Area (%)": "{:.1f}", "Change (pp)": "{:+.1f}"}), use_container_width=True)
        
    with col2:
        # Grouped Bar Chart (Before vs After) using Plotly
        # Melt dataframe for plotly express
        df_melted = df.melt(id_vars=["Category"], value_vars=[f"{year1} Area (%)", f"{year2} Area (%)"], 
                            var_name="Year", value_name="Percentage Area")
        
        fig = px.bar(df_melted, x="Category", y="Percentage Area", color="Year", barmode="group",
                     title=f"Land Cover Comparison: {year1} vs {year2}")
        st.plotly_chart(fig, use_container_width=True)

    # --- Section 5: Environmental Risk Gauge ---
    st.header("5. Environmental Risk Assessment")
    
    gauge_col, exp_col = st.columns([1, 1])
    
    with gauge_col:
        # Determine gauge color
        if risk_label == "Low":
            bar_color = "green"
        elif risk_label == "Moderate":
            bar_color = "orange"
        else:
            bar_color = "red"
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Level: {risk_label}", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.1)'},
                    {'range': [30, 60], 'color': 'rgba(255, 165, 0, 0.1)'},
                    {'range': [60, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with exp_col:
        st.subheader("Explanation")
        st.info(risk_explanation)
        st.markdown(
            """
            **How is this calculated?**
            - The score is rule-based and heavily penalizes detrimental changes.
            - *Vegetation or Water decreases* increase the risk score.
            - *Built-up or Barren increases* increase the risk score.
            - Scores above 60 indicate significant negative environmental impact.
            """
        )

if __name__ == "__main__":
    main()
