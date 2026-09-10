# 🚦 Smart City Traffic Digital Twin - IEEE SWC 2026 Challenge

Welcome to the official repository for our **Digital Twin & Traffic Optimization Simulation** built for the **IEEE Smart World Congress 2026 Challenge**. This project provides an interactive macroscopic simulation of a 100-node city, designed to balance traffic flow while prioritizing three distinct ethical and efficiency goals.

## 🎯 Project Objectives

Our simulation algorithm doesn't just aim to reduce global delay. It is fundamentally designed around three core objectives:

1. **🌿 Environmental Justice:** Heavy vehicles (trucks) are aggressively rerouted away from highly vulnerable nodes (Schools and Hospitals) to minimize emissions (kg CO2) and acoustic pollution in critical areas. We apply a *Global Emission Cap* mathematically bounded by traffic density.
2. **🚌 Modal Equality:** Public transport is prioritized. The algorithm attempts to minimize the specific delay of intersections with a high density of buses, ensuring that public transit remains the most efficient way to travel.
3. **🗺️ Spatial Equity:** Traffic is spatially distributed to balance the *Volume-to-Capacity (V/C) Ratio* across different city zones. This prevents specific disadvantaged neighborhoods from bearing the entire burden of traffic jams.

## ⚙️ Technical Architecture

### 1. Simulation Engine (Streamlit)
The core visualizer and physics engine is built in Python using **Streamlit** and **Plotly DeckGL**. 
It features a hybrid macroscopic traffic simulation model:
* **Dynamic Routing:** Diverts traffic flow to neighboring nodes (calculated via Delaunay Triangulation edges) while respecting lane capacities.
* **Webster Signal Control:** Simulates the exponential delay caused by traffic lights when intersections approach saturation ($x \to 1$).
* **Emission Physics:** Recalculates emissions dynamically using a quadratic congestion formula $E_{sim} = E_{base} \times (V_{sim}/V_{base})^2$.

### 2. Graph Neural Network (GNN) AI Training
We modeled the city as a planar graph using **Scipy's Delaunay Triangulation**. 
A custom **PyTorch Graph Convolutional Network (GCN)** was trained using **Stable Baselines 3 (PPO)** to act as an "AI Autopilot".
* **Node Features:** 12 features per node (Vehicles, Heavy Vehicles, Buses, Emergency Vehicles, Lanes, Speed Limits, Schools, Hospitals, Accidents, Emissions, Wait Times, Queue Lengths).
* **Observation Space:** A flattened $1200$-dimensional tensor representing the entire city state.
* The GNN learns the spatial dependencies of the city and attempts to autonomously output the optimal weights for Environmental, Modal, and Spatial justice to prevent phenomena like **Braess's Paradox** (where adding road capacity paradoxically increases total delay).

## 🚀 How to Run

### Prerequisites
Make sure you have Python 3.10+ installed.
```bash
pip install streamlit pandas numpy scipy plotly gymnasium stable-baselines3 torch
```

### Launch the Dashboard
```bash
python -m streamlit run dashboard.py
```
This will launch the interactive Digital Twin on `http://localhost:8501`. 
You can use the sidebar to load the `smart_city_traffic_mobility.csv` dataset, select the hour of the day, and dynamically shift the simulation weights to see the physical effects on the 3D map.

## 📁 Repository Structure
* `dashboard.py`: The main Streamlit web application and simulation engine.
* `gnn_utils.py`: Contains the PyTorch architecture for the GNN Feature Extractor and Delaunay Graph construction.
* `generate_gnn_colab.py`: Python script that generates the Jupyter Notebook (`GNN_Colab_Training.ipynb`) used for training the AI on Google Colab.
* `technical_presentation.html`: HTML Slides for the technical pitch.
* `technical_canva_import.pptx`: Exported presentation outline for Canva.

---
*Built with ❤️ for the IEEE SWC 2026 Challenge.*
