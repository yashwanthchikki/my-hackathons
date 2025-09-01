import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

 
# Load data
df = pd.read_excel('data.xlsx')

image = Image.open("logo.png")
resized_image = image.resize((15000, 3000))  # (width, height)
st.image(resized_image)
 

st.title('Webpage Prioritization Dashboard')

 
# --- 1. FILTERS ---
        
        # Weight sliders
st.sidebar.header('Adjust Feature Weights (Total must be 100%)')
pg_visitors = st.sidebar.slider('visitor count (%)', 0, 100, 0)
sub_friction1_weight = st.sidebar.slider('PG Friction# Calls within 7 day Weight (%)', 0, 100, 0)
sub_friction2_weight = st.sidebar.slider('Avg. AHT per call Weight (%)', 0, 100, 0)
Desktop_Switch_Rate = st.sidebar.slider('Desktop Switch Rate (%)', 0, 100, 0)
pg_visits = st.sidebar.slider('visit count (%)', 0, 100, 0)
PG_Visits_per_Visitor = st.sidebar.slider('PG Visits per Visitor(%)', 0, 100, 0)
sub_friction3_weight = st.sidebar.slider('PG Friction - Switch to Desktop within 7 days(%)', 0, 100, 0)
Ease_of_Use = st.sidebar.slider('Ease_of_Use(%)', 0, 100, 0)
CEI_Top2Box = st.sidebar.slider('CEI - Top2Box(%)', 0, 100, 0)

total_weight = pg_visitors + sub_friction1_weight + sub_friction2_weight+Desktop_Switch_Rate+pg_visits+PG_Visits_per_Visitor+sub_friction3_weight+Ease_of_Use +CEI_Top2Box
if total_weight == 0:
    st.sidebar.error("Total weight cannot be zero.")
    st.stop()

# Apply normalizationDesktop Switch Rate
pg_visitors = pg_visitors / total_weight * 100
sub_friction1_weight = sub_friction1_weight / total_weight * 100
sub_friction2_weight = sub_friction2_weight / total_weight * 100
Desktop_Switch_Rate = Desktop_Switch_Rate / total_weight * 100
pg_visits = pg_visits / total_weight * 100
PG_Visits_per_Visitor = PG_Visits_per_Visitor / total_weight * 100
sub_friction3_weight=sub_friction3_weight / total_weight * 100
Ease_of_Use=Ease_of_Use / total_weight * 100 
CEI_Top2Box= CEI_Top2Box / total_weight * 100


        # Engagement checkboxes in dropdown
with st.sidebar.expander("Filter Engagement Classes", expanded=False):
    engagement_classes = df['MULTI_PDT_DEEPER_ENGAGEMENT'].dropna().unique().tolist()
    selected_engagement = [
        cls for cls in engagement_classes
        if st.checkbox(cls, value=True, key=f'eng_{cls}')
    ]
        
# Filter by engagement
filtered_df = df[df['MULTI_PDT_DEEPER_ENGAGEMENT'].isin(selected_engagement)]
        
# PRODUCT filter if 'Non-DC' selected
if 'Non-DC' in selected_engagement:
    non_dc_rows = filtered_df[filtered_df['MULTI_PDT_DEEPER_ENGAGEMENT'] == 'Non-DC']
    if not non_dc_rows.empty:
        with st.sidebar.expander("Filter PRODUCT for Non-DC", expanded=False):
            product_classes = non_dc_rows['PRODUCT'].dropna().unique().tolist()
            selected_products = [
                p for p in product_classes
                if st.checkbox(p, value=True, key=f'prod_{p}')
                    ]
        # Apply PRODUCT filter only to Non-DC rows
        filtered_df = filtered_df[~(
            (filtered_df['MULTI_PDT_DEEPER_ENGAGEMENT'] == 'Non-DC') &
            (~filtered_df['PRODUCT'].isin(selected_products))
                )]
        
# Toggle switch for column 'PG transaction'
if 'PG Transaction' in df.columns:
    zz_toggle = st.sidebar.toggle("PG Transaction ", value=True)
    zz_value = 0 if zz_toggle else 1
    filtered_df = filtered_df[filtered_df['PG Transaction'] == zz_value]

# Toggle switch for column 'Satisfaction - CEI #Responses
if 'Satisfaction - CEI #Responses' in df.columns:
    zz_toggle = st.sidebar.toggle("Satisfaction count  >30 ", value=True)
    filtered_df = filtered_df[filtered_df['Satisfaction - CEI #Responses'] >=30]

#normalisation of big fetures
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())
filtered_df['Normalized PG Visits'] = normalize(filtered_df['PG Visits'])
filtered_df['Normalized PG Visitors'] = normalize(filtered_df['PG Visitors'])
filtered_df['Normalized PG Friction - # Calls within 7 days'] = normalize(filtered_df['PG Friction - # Calls within 7 days'])
filtered_df['Normalized Switch to Desktop within 7 days'] = normalize(filtered_df['PG Friction - Switch to Desktop within 7 days'])

        
        # --- 2. PRIORITIZATION LOGIC ---
        
# Calculate weighted raw score
filtered_df['RawPriority'] =  (
    filtered_df['Normalized PG Visitors'] * (pg_visitors/ 100) +
    filtered_df['Normalized PG Friction - # Calls within 7 days'] * (sub_friction1_weight / 100) +
    filtered_df['Avg. AHT per call'] * (sub_friction2_weight / 100)+
    filtered_df['Desktop Switch Rate'] * (Desktop_Switch_Rate  / 100)+
    filtered_df['Normalized PG Visits'] * (pg_visits / 100) +
    filtered_df['PG Visits per Visitor'] * (PG_Visits_per_Visitor / 100)+
    filtered_df['Normalized Switch to Desktop within 7 days'] * (sub_friction3_weight/ 100)+
    filtered_df['Ease of Use - Top2Box'] * ( Ease_of_Use / 100)+ 
    filtered_df['CEI - Top2Box'] * ( CEI_Top2Box/ 100)
           
        )
        
# Normalize to percentage
max_priority = filtered_df['RawPriority'].max()
if max_priority == 0:
    filtered_df['Priority (%)'] = 0
else:
    filtered_df['Priority (%)'] = (filtered_df['RawPriority'] / max_priority * 100).round(2)
        
# Sort by priority
filtered_df = filtered_df.sort_values(by='Priority (%)', ascending=False)
        
# --- 3. DISPLAY RESULTS ---
        
st.subheader('Prioritized Pages')
st.write(filtered_df[['PAGE_GROUP', 'Priority (%)']])

    
    
    

# Define your fixed features here
fixed_features = ['PG Visitors', 'Call Rate', 'CEI - Top2Box', 'Desktop Switch Rate']  

# Input for pagename
pagename = st.text_input("🔍 Enter the pagename to visualize")

# Generate radar plot if pagename is provided
if pagename:
    row = df[df['PAGE_GROUP'] == pagename]

    if not row.empty:
        values = row[fixed_features].values.flatten()
        max_values = df[fixed_features].max().values
        normalized_values = values / max_values

        # Prepare radar plot
        labels = fixed_features
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        normalized_values = normalized_values.tolist()
        normalized_values += normalized_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, normalized_values, color='blue', linewidth=2)
        ax.fill(angles, normalized_values, color='blue', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f"Radar Plot for '{pagename}'", size=16)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Pagename not found in the dataset.")
# my-hackathons
