import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import Delaunay
import numpy as np

# Page configuration
st.set_page_config(page_title="FairMobility Digital Twin", layout="wide", page_icon="🚦")

# Load data with cache to make the app fast
@st.cache_data
def load_data(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer)
    
    # We aggregate the dataset by Intersection AND Hour.
    # 100 intersections * 24 hours = 2,400 rows. This makes the dashboard lightning fast!
    hourly_agg = df.groupby(['latitude', 'longitude', 'hour']).agg({
        'emission_estimate': 'mean',
        'heavy_vehicle_count': 'mean',
        'nearby_school': 'max',
        'nearby_hospital': 'max',
        'city_zone': 'first',
        'congestion_level': 'first',
        'average_wait_time': 'mean',
        'vehicle_count': 'mean',
        'public_transport_count': 'mean',
        'intersection_id': 'first',
        'road_type': 'first',
        'lanes': 'first',
        'speed_limit': 'first',
        'queue_length': 'mean',
        'emergency_vehicle_detected': 'max',
        'accident_reported': 'max'
    }).reset_index()
    
    return df, hourly_agg

# --- SIDEBAR (The Control Room) ---
st.sidebar.title("🎛️ Fairness Control Room")

st.sidebar.subheader("📂 Custom Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your own CSV to test the Digital Twin:", type="csv")

if uploaded_file is not None:
    df, hourly_agg = load_data(uploaded_file)
else:
    df, hourly_agg = load_data("smart_city_traffic_mobility.csv")

st.sidebar.markdown("---")

st.sidebar.subheader("⏳ Time Evolution")
st.sidebar.markdown("Watch how the city breathes throughout the day.")
selected_hour = st.sidebar.slider("Select Hour of Day", 0, 23, 8, format="%02d:00")

st.sidebar.markdown("---")
st.sidebar.markdown("Adjust the algorithm weights to balance traffic fairly.")

st.sidebar.subheader("1. Environmental Justice")
env_weight = st.sidebar.slider("🌿 Protect Schools/Hospitals", 0, 100, 100, step=10)

st.sidebar.subheader("2. Modal Equality")
modal_weight = st.sidebar.slider("🚌 Prioritize Public Transport", 0, 100, 100, step=10)

st.sidebar.subheader("3. Spatial Equity")
spatial_weight = st.sidebar.slider("🗺️ Balance Neighborhood Delay", 0, 100, 100, step=10)

st.sidebar.divider()
st.sidebar.subheader("🚦 Simulation Engine Mode")
sim_mode = st.sidebar.radio(
    "Choose how the city manages traffic:",
    ["Hybrid (Routing + Webster Signal Control)", "Dynamic Routing Only", "Webster Signal Control Only", "None (Original City State)"]
)

st.sidebar.divider()
st.sidebar.subheader("🌍 Global Constraints")
_temp_df = hourly_agg[hourly_agg['hour'] == selected_hour]
min_cap_possible = int(_temp_df['emission_estimate'].mean())
max_cap_possible = int(_temp_df['emission_estimate'].max())
global_emission_cap = st.sidebar.slider(
    "Global Emission Cap (kg)", 
    min_value=min_cap_possible, 
    max_value=max_cap_possible + 100, 
    value=max_cap_possible, 
    step=10,
    help=f"The lowest possible cap is {min_cap_possible}kg. Below this, physics prevents further reduction because traffic has to go *somewhere*!"
)
traditional_efficiency = st.sidebar.slider("Traditional Optimization (Max Throughput)", 0, 100, 0, step=10)

st.sidebar.divider()
st.sidebar.subheader("🗺️ Map Visualization")
map_metric = st.sidebar.radio("Color intersections by:", ["Emissions (kg)", "Congestion (Wait Time)"])


# --- MAIN DASHBOARD ---
st.title("🚦 FairMobility: Smart City Digital Twin")
st.markdown("This dashboard demonstrates how combining **Environmental**, **Modal**, and **Spatial** equity can reshape urban mobility optimization.")

st.divider()

# --- SECTION 1: MAP ---
st.header(f"1. Digital Twin City Map (Time: {selected_hour:02d}:00)")
st.markdown("The map below updates dynamically to show the traffic situation at the hour you selected. Move the Time Slider to see the morning/evening rush hours evolve!")

# Filter data for the specific hour selected
agg_df = hourly_agg[hourly_agg['hour'] == selected_hour].copy()
agg_df.reset_index(drop=True, inplace=True)

# --- SIMULATION ENGINE: Dynamic Routing & Load Balancing ---
coords = agg_df[['latitude', 'longitude']].values
tri = Delaunay(coords)

# 1. Build Adjacency List & Filter Long Edges
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sqrt(np.sum(diff**2, axis=-1))

adj_list = {i: set() for i in range(len(coords))}
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i+1, 3):
            u, v = simplex[i], simplex[j]
            if dist[u, v] < 0.04:
                adj_list[u].add(v)
                adj_list[v].add(u)
                edges.add(tuple(sorted([u, v])))

# 2. Dynamic Capacity Balancing (Water-filling / Spatial Diffusion)
agg_df['simulated_vehicle_count'] = agg_df['vehicle_count'].astype(float)
agg_df['simulated_heavy_vehicle_count'] = agg_df['heavy_vehicle_count'].astype(float)

iterations = 3 # Number of diffusion passes to reach equilibrium

if "Routing" in sim_mode:
    for step in range(iterations):
        vehicle_delta = np.zeros(len(agg_df))
        heavy_delta = np.zeros(len(agg_df))
        
        for i in range(len(agg_df)):
            neighbors = list(adj_list[i])
            if len(neighbors) == 0:
                continue
                
            current_heavy = agg_df.at[i, 'simulated_heavy_vehicle_count']
            current_veh = agg_df.at[i, 'simulated_vehicle_count']
            is_vuln = (agg_df.at[i, 'nearby_school'] == 1) or (agg_df.at[i, 'nearby_hospital'] == 1)
            
            neighbor_heavy_avg = np.mean([agg_df.at[n, 'simulated_heavy_vehicle_count'] for n in neighbors])
            neighbor_veh_avg = np.mean([agg_df.at[n, 'simulated_vehicle_count'] for n in neighbors])
            
            # Calculate total physical capacity of neighbors
            total_neighbor_lanes = sum([agg_df.at[n, 'lanes'] for n in neighbors])
            
            # 1. Global Emission Cap: Forcibly push traffic out if emissions are too high (applies everywhere)
            if agg_df.at[i, 'emission_estimate'] > global_emission_cap and current_veh > 0:
                # Panic routing: dump up to 50% of traffic to avoid exceeding the cap
                to_dump = current_veh * 0.5
                vehicle_delta[i] -= to_dump
                for n in neighbors:
                    vehicle_delta[n] += to_dump * (agg_df.at[n, 'lanes'] / total_neighbor_lanes)
                    
            # 2. Environmental Diffusion (Goal 1): Actively push trucks away from vulnerable areas
            if is_vuln and current_heavy > 0:
                push_factor = (env_weight / 100.0) * 0.8
                to_move = current_heavy * push_factor
                heavy_delta[i] -= to_move
                for n in neighbors:
                    heavy_delta[n] += to_move * (agg_df.at[n, 'lanes'] / total_neighbor_lanes)
                    
            # 3. Spatial Homogenization (Goal 3): Let traffic flow to less congested neighbors (Based on Capacity)
            my_vc = current_veh / (agg_df.at[i, 'lanes'] * 1800.0)
            neighbor_vc_avg = np.mean([agg_df.at[n, 'simulated_vehicle_count'] / (agg_df.at[n, 'lanes'] * 1800.0) for n in neighbors])
            
            diff_vc = my_vc - neighbor_vc_avg
            if diff_vc > 0:
                excess_cars = diff_vc * (agg_df.at[i, 'lanes'] * 1800.0)
                flow_factor = (spatial_weight / 100.0) * 0.5 
                to_flow = excess_cars * flow_factor
                vehicle_delta[i] -= to_flow
                for n in neighbors:
                    vehicle_delta[n] += to_flow * (agg_df.at[n, 'lanes'] / total_neighbor_lanes)
                    
            # 4. Traditional Optimization: Pure throughput maximization (Overrides fairness)
            # Pushes traffic from low-lane roads to high-lane roads aggressively
            if traditional_efficiency > 0 and current_veh > 0:
                # If my density (cars per lane) is higher than the neighborhood average, dump it
                my_density = current_veh / agg_df.at[i, 'lanes']
                neighbor_density_avg = np.mean([agg_df.at[n, 'simulated_vehicle_count'] / agg_df.at[n, 'lanes'] for n in neighbors])
                
                if my_density > neighbor_density_avg:
                    efficiency_flow = (current_veh * 0.3) * (traditional_efficiency / 100.0)
                    vehicle_delta[i] -= efficiency_flow
                    for n in neighbors:
                        vehicle_delta[n] += efficiency_flow * (agg_df.at[n, 'lanes'] / total_neighbor_lanes)
                    
        agg_df['simulated_vehicle_count'] += vehicle_delta
        agg_df['simulated_heavy_vehicle_count'] += heavy_delta
        
        # Floor to zero
        agg_df['simulated_vehicle_count'] = np.maximum(agg_df['simulated_vehicle_count'], 0)
        agg_df['simulated_heavy_vehicle_count'] = np.maximum(agg_df['simulated_heavy_vehicle_count'], 0)

# 3. Recalculate Wait Times & Emissions based on the new balanced fluid volumes
ratio_veh = np.where(agg_df['vehicle_count'] > 0, agg_df['simulated_vehicle_count'] / agg_df['vehicle_count'], 1.0)

# Modal shift: prioritize buses, reducing wait time where bus density is high
if "None" in sim_mode:
    modal_shift = 1.0
else:
    modal_shift = 1.0 - (agg_df['public_transport_count'] / (agg_df['public_transport_count'].max() + 1)) * (modal_weight / 100.0)

# Webster's Formula Calculation (Total Cycle)
saturation_flow = agg_df['lanes'] * 1800
# Cap Y_total at 0.95 to simulate maximum saturation before mathematical gridlock
Y_total = np.clip(agg_df['simulated_vehicle_count'] / saturation_flow, 0.01, 0.95)
L = 10.0 # 10 seconds total lost time
agg_df['webster_cycle'] = (1.5 * L + 5) / (1 - Y_total)

if "Webster" in sim_mode:
    # --- Advanced HCM Delay Calculation (from optimizer.py) ---
    def hcm_delay(q_flow, lanes):
        sat_flow = lanes * 1800
        # Use Webster for baseline Cycle
        Y_base = np.clip(q_flow / sat_flow, 0.01, 0.95)
        L = 10.0
        C = (1.5 * L + 5) / (1 - Y_base)
        g = np.maximum(15.0, (C - L) / 2.0)
        lam = np.clip(g / C, 0.05, 0.95)
        
        q_flow = np.maximum(1.0, q_flow)
        capacity = lam * sat_flow
        x = q_flow / np.maximum(capacity, 1.0)
        
        denom1 = 2 * (1.0 - np.minimum(lam * x, 0.99))
        d_uniform = (C * ((1.0 - lam) ** 2)) / np.maximum(denom1, 0.01)
        
        q_sec = q_flow / 3600.0
        d_random = np.where(
            x < 0.95,
            (x ** 2) / np.maximum(2 * q_sec * (1.0 - x), 0.001),
            15.0 + 120.0 * (x - 0.95)
        )
        return np.clip(d_uniform + d_random, 2.0, 900.0)

    # 1. Calculate the theoretical delay for the ORIGINAL traffic state
    base_hcm_delay = hcm_delay(agg_df['vehicle_count'], agg_df['lanes'])
    
    # 2. Calculate the theoretical delay for the NEW simulated traffic state
    sim_hcm_delay = hcm_delay(agg_df['simulated_vehicle_count'], agg_df['lanes'])
    
    # 3. Calculate the Gain/Loss percentage
    delay_ratio = np.where(base_hcm_delay > 0, sim_hcm_delay / base_hcm_delay, 1.0)
    
    # 4. Apply this realistic percentage to the dataset's actual sensor wait times
    agg_df['simulated_wait'] = agg_df['average_wait_time'] * delay_ratio * modal_shift
    agg_df['webster_cycle'] = (1.5 * 10.0 + 5) / (1 - np.clip(agg_df['simulated_vehicle_count'] / (agg_df['lanes'] * 1800), 0.01, 0.95))

else:
    # Standard macroscopic ratio adjustment
    agg_df['simulated_wait'] = agg_df['average_wait_time'] * ratio_veh * modal_shift
    agg_df['webster_cycle'] = 0.0

agg_df['simulated_emission'] = agg_df['emission_estimate'] * ratio_veh * ratio_veh # Emissions scale non-linearly with volume

# Determine Node Status based on load changes
agg_df['node_status'] = "🟢 STABLE (Flow Balanced)"
agg_df.loc[ratio_veh < 0.90, 'node_status'] = "🔵 TRAFFIC DIVERTED (Protected)"
agg_df.loc[ratio_veh > 1.10, 'node_status'] = "🟠 ABSORBING LOAD (Homogenizing)"

# --- END SIMULATION ENGINE ---

# Select which metric to display on the map
if map_metric == "Emissions (kg)":
    node_color = agg_df['simulated_emission']
    color_title = "Emissions (kg)"
    map_title = f"City Road Network - Emissions at {selected_hour:02d}:00"
else:
    node_color = agg_df['simulated_wait']
    color_title = "Wait Time (min)"
    map_title = f"City Road Network - Congestion at {selected_hour:02d}:00"

# Create rich hover text
agg_df['hover_text'] = (
    "<b>Intersection:</b> " + agg_df['intersection_id'] + "<br>" +
    "<b>Status:</b> " + agg_df['node_status'] + "<br>" +
    "<b>Zone:</b> " + agg_df['city_zone'] + " (" + agg_df['road_type'] + ")<br>" +
    "<b>Webster Cycle:</b> " + agg_df['webster_cycle'].round(1).astype(str) + " s<br>" +
    "<b>Avg Wait Time:</b> " + agg_df['simulated_wait'].round(1).astype(str) + " min<br>" +
    "<b>Emissions:</b> " + agg_df['simulated_emission'].round(1).astype(str) + " kg<br>" +
    "<b>Cars / Buses:</b> " + agg_df['simulated_vehicle_count'].round(0).astype(str) + " / " + agg_df['public_transport_count'].round(0).astype(str) + "<br>" +
    "<b>Near School:</b> " + agg_df['nearby_school'].map({1: 'Yes', 0: 'No'}) + " | " +
    "<b>Near Hospital:</b> " + agg_df['nearby_hospital'].map({1: 'Yes', 0: 'No'})
)

edge_x = []
edge_y = []
# Calculate distances to drop artificially long edges on the outer border
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist = np.sqrt(np.sum(diff**2, axis=-1))

for (i, j) in edges:
    if dist[i, j] < 0.04:
        edge_x.extend([coords[i][1], coords[j][1], None])
        edge_y.extend([coords[i][0], coords[j][0], None])
        
# 3. Plot using Plotly Graph Objects
fig_map = go.Figure()

# Draw roads (light gray lines)
fig_map.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=1, color='#E0E0E0'),
    hoverinfo='none',
    showlegend=False
))

# Define marker shapes based on traffic routing status
agg_df['marker_symbol'] = 'circle'
agg_df.loc[agg_df['node_status'].str.contains('DIVERTED'), 'marker_symbol'] = 'triangle-down'
agg_df.loc[agg_df['node_status'].str.contains('ABSORBING'), 'marker_symbol'] = 'triangle-up'

# Define marker sizes (make active routing nodes bigger so they pop out)
agg_df['marker_size'] = agg_df['heavy_vehicle_count'] * 0.3
agg_df.loc[agg_df['node_status'].str.contains('DIVERTED'), 'marker_size'] *= 1.5
agg_df.loc[agg_df['node_status'].str.contains('ABSORBING'), 'marker_size'] *= 1.5

# Draw intersections (red hot spots)
fig_map.add_trace(go.Scatter(
    x=agg_df['longitude'], y=agg_df['latitude'],
    mode='markers',
    marker=dict(
        size=agg_df['marker_size'],
        symbol=agg_df['marker_symbol'],
        color=node_color,
        colorscale=['#E0E0E0', '#FF9999', '#FF0000', '#8B0000'],
        showscale=True,
        colorbar=dict(title=color_title),
        line=dict(width=1, color='black') # Add a border so triangles are easy to see
    ),
    text=agg_df['hover_text'],
    hoverinfo='text',
    showlegend=False
))

# Overlay Icons for Schools
schools_df = agg_df[agg_df['nearby_school'] == 1]
fig_map.add_trace(go.Scatter(
    x=schools_df['longitude'], y=schools_df['latitude'],
    mode='text',
    text=['🏫'] * len(schools_df),
    textposition='top center',
    textfont=dict(size=18),
    hoverinfo='none',
    showlegend=False
))

# Overlay Icons for Hospitals
hospitals_df = agg_df[agg_df['nearby_hospital'] == 1]
fig_map.add_trace(go.Scatter(
    x=hospitals_df['longitude'], y=hospitals_df['latitude'],
    mode='text',
    text=['🏥'] * len(hospitals_df),
    textposition='bottom center',
    textfont=dict(size=18),
    hoverinfo='none',
    showlegend=False
))

fig_map.update_layout(
    title=map_title,
    template="plotly_white",
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(visible=False, showgrid=False),
    yaxis=dict(visible=False, showgrid=False, scaleanchor="x", scaleratio=1.3),
    height=800  # Make the map much taller
)

# Display the map full-width
st.plotly_chart(fig_map, use_container_width=True)

# --- METRICS & CHARTS ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Impact Simulator (Vulnerable Zones)")
    # We calculate original metrics for the selected hour only
    vuln_mask = (agg_df['nearby_school'] == 1) | (agg_df['nearby_hospital'] == 1)
    orig_emission = agg_df[vuln_mask]['emission_estimate'].mean()
    sim_emission = orig_emission * (1 - (env_weight / 100.0) * 0.4)
    st.metric("Avg Emissions in Vulnerable Zones", f"{sim_emission:.1f} kg", f"{sim_emission - orig_emission:.1f} kg (Simulated)")

    orig_trucks = agg_df[vuln_mask]['heavy_vehicle_count'].mean()
    sim_trucks = orig_trucks * (1 - (env_weight / 100.0) * 0.6)
    st.metric("Avg Heavy Vehicles (Trucks)", f"{sim_trucks:.1f}", f"{sim_trucks - orig_trucks:.1f} (Diverted)")

with col2:
    st.subheader("Neighborhood Delay Balance")
    zone_waits = agg_df.groupby('city_zone')['simulated_wait'].mean().reset_index()
    fig_bar = px.bar(
        zone_waits, x='city_zone', y='simulated_wait',
        title=f"Wait Times by City Zone at {selected_hour:02d}:00",
        labels={'simulated_wait': 'Wait Time (mins)', 'city_zone': 'City Zone'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
