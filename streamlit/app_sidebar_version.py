import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
# Load required libraries for modeling
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime

# Set page configuration
def add_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "autorxn_logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    else:
        st.sidebar.warning("Logo image not found.")

st.set_page_config(
    page_title="AutoRxn Optimiser Platform",
    page_icon="🧪",
    layout="wide"
)
add_logo()

# Sidebar with user inputs
st.sidebar.header("Dashboard")

from streamlit_option_menu import option_menu

with st.sidebar:
    # Logo (already handled by add_logo above)

    # Divider
    st.markdown("---")

    # Navigation menu
    selected = option_menu(
    menu_title=None,  # No "Navigation" title
    options=["Home", "Configuration", "Reaction Data", "Modelling"],
    icons=["house", "gear", "table", "graph-up"],
    menu_icon=None,
    default_index=0,
    styles={
        "container": {
            "padding": "0",
            "background-color": "#f9f9f9",
            "border-radius": "0"
        },
        "icon": {"color": "#6c757d", "font-size": "18px"},
        "nav-link": {
            "font-size": "15px",
            "padding": "8px 12px",
            "margin": "4px 0",
            "border-radius": "6px",
            "--hover-color": "#e9ecef"
        },
        "nav-link-selected": {
            "background-color": "#e1efff",
            "font-weight": "600",
            "color": "#0056b3"
        }
    }
)

        # ── Save Project ─────────────────────────────────────────────
    st.markdown("### 📁 Project")
    data = {
        "project_name":   st.session_state.get("project_name", "MyProject"),
        "parameters":     st.session_state.get("parameters", []),
        "reaction_data":  st.session_state.get("reaction_data", []),
        "model_settings": st.session_state.get("model_settings", {}),
        "suggestions":    st.session_state.get("suggestions", []),
        "project_notes":  st.session_state.get("project_notes", ""),
        "change_log":     st.session_state.get("change_log", []),
    }
    # Save Project button returns True when clicked
    save_clicked = st.download_button(
        label="💾 Save Project",
        data=json.dumps(data, default=str),
        file_name=f"{st.session_state.get('project_name','MyProject')}.json",
        mime="application/json"
    )
    if save_clicked:
        log = st.session_state.get("change_log", [])
        log.append(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} — Project saved")
        st.session_state.change_log = log

    st.markdown("---")
    st.caption("AutoRxn © 2025")




user_name = st.sidebar.text_input("Chemist Name", "Iron Doctor")


# Add a greeting
st.header(f"Welcome, {user_name}! 👋")

# Create tabs for different sections

if selected == "Home":
    st.title("AutoRxn Dashboard")
    st.markdown("Let's get started! Create a new project or load an existing one to get started.")

    # ── Two columns: left for create/load, right for notes/log ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Project Setup")
        # Create New Project
        if st.button("➕ Create New Project"):
            for k in [
                "project_name","parameters","reaction_data",
                "model_settings","suggestions","project_notes",
                "change_log","project_loaded"
            ]:
                st.session_state.pop(k, None)
            st.success("Starting a new project…")
            st.rerun()

        st.markdown("---")

        # Load Existing Project
        uploaded = st.file_uploader("📂 Load Project File (.json)", type="json")
        if uploaded is not None and not st.session_state.get("project_loaded", False):
            data = json.load(uploaded)
            st.session_state.project_name   = data.get("project_name", "")
            st.session_state.parameters     = data.get("parameters", [])
            st.session_state.reaction_data  = data.get("reaction_data", [])
            st.session_state.model_settings = data.get("model_settings", {})
            st.session_state.suggestions    = data.get("suggestions", [])
            st.session_state.project_notes  = data.get("project_notes", "")
            st.session_state.change_log     = data.get("change_log", [])
            st.session_state.project_loaded = True
            st.success(f"Loaded project: {st.session_state.project_name}")
            st.rerun()

    with col2:
        st.markdown("### Project Details")
        if st.session_state.get("project_loaded", False):
            st.markdown(f"**Current Project:** {st.session_state.project_name}")

            # Project Notes
            st.markdown("#### 📝 Notes")
            notes = st.text_area(
                "Enter your project notes here",
                value=st.session_state.get("project_notes", ""),
                height=120
            )
            st.session_state.project_notes = notes

            st.markdown("---")

            # Change Log
            st.markdown("#### 🕓 Change Log")
            log = st.session_state.get("change_log", [])
            if log:
                for entry in log:
                    st.markdown(f"- {entry}")
            else:
                st.info("No changes logged yet.")
        else:
            st.info("Load or create a project to see notes & change log.")




elif selected == "Configuration":
    # ── Initialize session-state containers ────────────────────────
    if "parameters" not in st.session_state:
        st.session_state.parameters = []
    if "categorical_choices" not in st.session_state:
        st.session_state.categorical_choices = ["", "", ""]
    if "change_log" not in st.session_state:
        st.session_state.change_log = []
    

    st.title("Configuration")

    # ── Two-column layout ─────────────────────────────────────────
    left, right = st.columns([1, 1])

    # ── Left: Define Parameters Form ─────────────────────────────
    with left:
        st.markdown("### 📝 Define Parameters")
    
        # Parameter name & type
        name = st.text_input("Parameter Name")
        param_type = st.selectbox("Type", ["Continuous", "Categorical"])
    
        # Continuous inputs
        if param_type == "Continuous":
            mn, mx = st.columns(2)
            with mn:
                lower = st.number_input("Min", value=0.0, key="min_val")
            with mx:
                upper = st.number_input("Max", value=1.0, key="max_val")
            choices = []

        # Categorical inputs (outside any form!)
        else:
            choices = []
            for i, val in enumerate(st.session_state.categorical_choices):
                c = st.text_input(
                    f"Choice {i+1}", 
                    value=val, 
                    key=f"cat_choice_{i}"
                )
                choices.append(c)

            # Add one more blank row on click
            if st.button("➕ Add Choice Row"):
                st.session_state.categorical_choices.append("")
                st.rerun()


        # Handle the click immediately after the form 
        if st.session_state.get("param_form_submitted", False) or st.session_state.get("param_form_submitted", None) is None:
            # But rather than fancy callbacks, simply use a normal button:
            if st.button("➕ Add Parameter"):
                entry = {
                    "Name": name,
                    "Type": param_type,
                    "Range": f"{lower}–{upper}" if param_type=="Continuous" else "",
                    "Choices": [c for c in choices if c.strip()]
                }
                st.session_state.parameters.append(entry)
                st.session_state.categorical_choices = ["", "", ""]
                st.success(f"Added parameter '{name}' ({param_type})")
                st.session_state.change_log.append(
                    f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} — Added parameter '{name}' ({param_type})"
                )

    # ── Full-width Parameter Table ─────────────────────────────────
    st.markdown("### 🔧 Current Parameters")
    if st.session_state.parameters:
        df = pd.DataFrame(st.session_state.parameters)
        edited = st.data_editor(df, num_rows="dynamic", key="param_editor")
        st.session_state.parameters = edited.to_dict("records")
    else:
        st.info("No parameters defined yet.")

    # ── Right: Configuration Settings ─────────────────────────────
    with right:
        st.markdown("### ⚙️ Configuration Settings")
        project_name = st.text_input("Project Name", value=st.session_state.get("project_name", ""))
        optimiser = st.selectbox("Optimiser", ["Bayesian", "Grid Search", "Random"])
        objective = st.selectbox("Objective", ["Maximise Yield", "Minimise Cost"])
        iterations = st.number_input("Max Iterations", value=st.session_state.get("model_settings", {}).get("iterations", 20), min_value=1)

        if st.button("💾 Save Configuration"):
            st.session_state.project_name = project_name
            st.session_state.model_settings = {
                "optimiser": optimiser,
                "objective": objective,
                "iterations": iterations
            }
            st.success("Configuration saved!")


       
elif selected == "Reaction Data":
    project = selected.split(":")[0]

    # ── Ensure session-state keys exist ───────────────────────────
    if "parameters" not in st.session_state:
        st.session_state.parameters = []
    if "reaction_data" not in st.session_state:
        st.session_state.reaction_data = []

    st.title(f"{project} — Reaction Explorer")

    tab_data, tab_suggest = st.tabs(["Reaction Data", "Suggest New Experiments"])

    with tab_data:
        st.subheader("Experimental Data")

        if st.session_state.get("suggestions"):
            st.markdown("### 🔮 Suggested Experiments")
            df_sugg = pd.DataFrame(st.session_state.suggestions)
            st.dataframe(df_sugg, use_container_width=True)

        if st.button("➕ Import Suggestions into Experimental Data"):
            # Append to reaction_data
            st.session_state.reaction_data.extend(st.session_state.suggestions)
            # Clear out suggestions so they don’t re-import
            st.session_state.suggestions = []
            st.success("Suggestions imported!")
            st.rerun()
            st.session_state.change_log.append(
                f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} — Imported {count} suggested experiments"
            )

        # 1) Build dynamic column names from your defined parameters
        param_names = [p["Name"] for p in st.session_state.parameters]
        columns = param_names + ["Reaction_Yield_%"]

        # 2) Download blank template for bulk upload
        df_template = pd.DataFrame(columns=columns)
        st.download_button(
            label="📥 Download Template CSV",
            data=df_template.to_csv(index=False),
            file_name=f"{project}_template.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # 3) Upload a filled CSV to populate the table
        uploaded = st.file_uploader("📤 Upload Filled CSV", type="csv")
        if uploaded is not None:
            df_uploaded = pd.read_csv(uploaded)
            # (Optional) you could validate df_uploaded.columns here
            st.session_state.reaction_data = df_uploaded.to_dict("records")
            st.success("Reactions loaded from CSV!")
            st.session_state.change_log.append(
                f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} — Uploaded reaction data CSV ({len(df_uploaded)} rows)"
            )

        st.markdown("---")

        # 4) Render editable grid **always** (empty or pre-loaded)
        df = (
            pd.DataFrame(st.session_state.reaction_data)
            if st.session_state.reaction_data
            else pd.DataFrame(columns=columns)
        )
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="reaction_data_editor"
        )
        # 5) Persist edits back to session_state
        st.session_state.reaction_data = edited.to_dict("records")

        st.markdown("---")

        # 6) Download the current data as CSV
        st.download_button(
            label="📥 Download Reaction Data",
            data=edited.to_csv(index=False),
            file_name=f"{project}_reactions.csv",
            mime="text/csv"
        )

    with tab_suggest:
        st.subheader("Suggest Next Experiments")

        focus = st.radio(
            "Experiment Focus",
            ["Optimize Yield", "Explore Parameter Space", "Test New Catalysts"]
        )

        num_suggestions = st.slider("Number of suggestions", 1, 10, 3)

        if st.button("Generate Experiment Suggestions"):
            # Pull in your parameter definitions
            params = st.session_state.get("parameters", [])

            suggestions = []
            for _ in range(num_suggestions):
                entry = {}
                for p in params:
                    name = p["Name"]
                    if p["Type"] == "Continuous":
                        # parse the saved range "lower–upper"
                        low, high = p["Range"].split("–")
                        low, high = float(low), float(high)
                        entry[name] = round(np.random.uniform(low, high), 2)
                    else:
                        # categorical: p["Choices"] is a list
                        choices = p.get("Choices", [])
                        entry[name] = np.random.choice(choices) if choices else ""

                # leave yield blank for later measurement
                entry["Reaction_Yield_%"] = None
                suggestions.append(entry)

            df_suggestions = pd.DataFrame(suggestions)
            st.markdown("#### Suggested Experiments")
            st.dataframe(df_suggestions, use_container_width=True)

            # store for import into Reaction Data tab
            st.session_state.suggestions = suggestions


elif selected == "Modelling":
    st.title("Predictive Modelling")

    model_tab, plot_tab, stats_tab, explain_tab = st.tabs([
        "Train Model", "Plots", "Summary Statistics", "Model Explanation / Version"
    ])

    with model_tab:
        st.subheader("Train a Regression Model")
        target = st.selectbox("Select Target Variable", ["Reaction_Yield_%"])
        model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost", "Linear Regression"])

        if st.button("Train Model"):
            st.success(f"Training {model_type} to predict {target}... (placeholder)")
            # Model training logic will go here

    with plot_tab:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted vs Actual")

            # Generate dummy prediction data
            n = 50
            actual = np.random.normal(70, 10, n)
            predicted = actual + np.random.normal(0, 5, n)

            # Create Plotly figure
            fig = px.scatter(
                x=actual,
                y=predicted,
                labels={"x": "Actual Yield (%)", "y": "Predicted Yield (%)"},
                title="Predicted vs Actual Yield"
            )
            fig.add_shape(
                type="line", line=dict(dash="dash"),
                x0=min(actual), y0=min(actual), x1=max(actual), y1=max(actual)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.download_button("📤 Export Plot as HTML", data=fig.to_html(), file_name="pred_vs_actual.html")

            st.subheader("Residual Plot")

            residuals = actual - predicted

            fig_resid = px.scatter(
                x=actual,
                y=residuals,
                labels={"x": "Actual Yield (%)", "y": "Residuals (Actual - Predicted)"},
                title="Residuals vs Actual Yield"
            )

            fig_resid.add_shape(
                type="line",
                line=dict(color="red", dash="dash"),
                x0=min(actual), y0=0,
                x1=max(actual), y1=0
            )

            st.plotly_chart(fig_resid, use_container_width=True)
            st.download_button(
                "📤 Export Residual Plot",
                data=fig_resid.to_html(),
                file_name="residual_plot.html"
            )


            st.subheader("Prediction Surface (3D Plot)")

            all_params = ["Pressure (bar)", "Reagent1 Conc (M)", "Reagent2 Conc (M)"]
            x_param = st.selectbox("X-axis Parameter", all_params, index=0, key="x_axis_3d")
            y_param = st.selectbox("Y-axis Parameter", all_params, index=1, key="y_axis_3d")

            # Create a parameter grid
            grid_x = np.linspace(1, 10, 40)
            grid_y = np.linspace(0.1, 2.0, 40)
            X, Y = np.meshgrid(grid_x, grid_y)

            # Simulate predicted surface
            Z = 75 + 5*np.sin(X) - 3*np.cos(Y) + np.random.normal(0, 1, X.shape)

            surface_fig = go.Figure(
                data=[go.Surface(z=Z, x=grid_x, y=grid_y, colorscale='Viridis')],
                layout=go.Layout(
                    title="Predicted Yield Surface",
                    scene=dict(
                        xaxis_title=x_param,
                        yaxis_title=y_param,
                        zaxis_title="Predicted Yield (%)"
                    )
                )
            )

            st.plotly_chart(surface_fig, use_container_width=True)
            st.download_button("📤 Export Surface Plot", data=surface_fig.to_html(), file_name="surface_plot.html")

        with col2:
            st.subheader("Parameter Importance")

            importance_data = {
                "Parameter": ["Pressure (bar)", "Reagent1 Conc (M)", "Reagent2 Conc (M)"],
                "Importance": np.random.uniform(0.1, 1.0, 3).round(2)
            }
            df_importance = pd.DataFrame(importance_data).sort_values("Importance", ascending=False)

            bar_fig = px.bar(
                df_importance,
                x="Importance",
                y="Parameter",
                orientation="h",
                title="Estimated Parameter Importance",
                labels={"Importance": "Feature Weight"},
                color="Importance",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(bar_fig, use_container_width=True)
            st.download_button("📤 Export Importance Plot", data=bar_fig.to_html(), file_name="param_importance.html")

            st.subheader("Acquisition Function (Simulated)")

            grid_x = np.linspace(1, 10, 40)
            grid_y = np.linspace(0.1, 2.0, 40)
            X, Y = np.meshgrid(grid_x, grid_y)

            # Simulate acquisition (e.g., Expected Improvement)
            acquisition = np.exp(-((X-6)**2 + (Y-1.2)**2)) + np.random.normal(0, 0.05, X.shape)

            acq_fig = px.imshow(
                acquisition,
                x=grid_x,
                y=grid_y,
                labels={"x": x_param, "y": y_param, "color": "EI"},
                color_continuous_scale="Turbo",
                title="Expected Improvement (Simulated)"
            )
            acq_fig.update_yaxes(autorange="reversed")

            st.plotly_chart(acq_fig, use_container_width=True)
            st.download_button("📤 Export Acquisition Plot", data=acq_fig.to_html(), file_name="acquisition.html")

            st.subheader("Prediction Uncertainty")

            uncertainty = np.random.uniform(0.5, 5.0, X.shape)

            unc_fig = px.imshow(
                uncertainty,
                x=grid_x,
                y=grid_y,
                labels={"x": x_param, "y": y_param, "color": "Std Dev"},
                color_continuous_scale="Oranges",
                title="Model Uncertainty (Std Dev)"
            )
            unc_fig.update_yaxes(autorange="reversed")

            st.plotly_chart(unc_fig, use_container_width=True)
            st.download_button("📤 Export Uncertainty Plot", data=unc_fig.to_html(), file_name="uncertainty.html")

            
    with stats_tab:
        st.subheader("Summary Statistics")

        # Ensure actual and predicted are available
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

        st.metric("R² Score", f"{r2:.3f}")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

    with explain_tab:
        st.subheader("Model Explanation / Versioning")
        st.markdown("""
        - **Model Type**: Not trained yet  
        - **Training Data Size**: —  
        - **Timestamp**: —  
        - **Notes**: Explain the rationale or upload training log here.
        """)
