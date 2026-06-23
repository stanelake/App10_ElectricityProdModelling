import streamlit as st

# 1. Global Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="Kaikohe Hybrid Asset Cockpit",
    page_icon="🏢",
    layout="wide"
)

# 2. Define the Page Routing Grid
# We assign the landing page code to a clean, isolated separate function rather than self-referencing the file name.

project_1_page = st.Page(page="Pages/01_SolarProd/mainSolar.py",
                         title = "Solar Calibration Dashboard",
                         icon = "☀️",)
project_2_page = st.Page(page="Pages/02_PriceMod/priceStreamlit.py", # This is a bike purchase project
                         title = "Electricity Price Modelling",
                         icon = "⚡",)
project_3_page = st.Page(page="Pages/03_SolarBatteryDemand/solarFarmDynamics.py",
                         title = "Solar Farm Dynamics",
                         icon = "🔋",)
project_4_page = st.Page(page="Pages/04_MonteCarloPricing/proj_Valuation.py",
                         title = "Project Valuation Room",
                         icon = "🎲",)

# 3. Handle Local Landing Page State Mechanics
# We create a placeholder 'page' that doesn't target an external file.
project_0_page = st.Page(
    page=lambda: render_landing_page(), # Directly runs our function when selected!
    title="Main Landing Page",
    icon="🏢",
    default=True
)

# 4. Initialize Navigation Container
pg = st.navigation({
                    "Executive Overview": [project_0_page,],
                    "Projects": [project_1_page,
                                 project_2_page,
                                 project_3_page,
                                 project_4_page]
                    }
                )

# 5. Define the Local Landing Page UI Layout Function
def render_landing_page():
    st.title("🏢 Kaikohe Hybrid Solar + Storage Investment Cockpit")
    st.markdown(
        "Welcome to the multi-page financial underwriting and engineering console for the Kaikohe Hybrid Energy Infrastructure Asset. "
        "This system links your underlying quantitative code bases to evaluate asset cash flows under stochastic risk."
    )

    # ── SIDEBAR CONTROLS & MODEL METADATA OVERRIDES ──────────────────────────────
    st.sidebar.header("🛠️ Plant System Parameters")

    st.markdown("### 🗺️ System Navigation Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.info("### ☀️ A. Solar Generation Module\n"
                "Evaluates atmospheric metrics, engineers clear-sky profiles, and calibrates "
                "the physics-informed Performance Ratio (PR) engine via LightGBM models.\n\n"
                "👉 *Navigate using the sidebar to: 1_☀️_Solar_Generation*")

        st.success("### 🔋 C. Asset Interaction Hub\n"
                "Simulates co-located operation profiles. Models how merchant solar production "
                "and regional electricity demand interact with the physical state-of-charge limits of a battery system.\n\n"
                "👉 *Navigate using the sidebar to: 3_🔋_Asset_Interaction*")

    with col2:
        st.warning("### ⚡ B. Electricity Price Engine\n"
                "Decomposes regional merchant prices into deterministic diurnal curves using 67 robust parameters "
                "and infers MRSVCJ stochastic volatility trace parameters using Gibbs MCMC chains.\n\n"
                "👉 *Navigate using the sidebar to: 2_⚡_Electricity_Pricing*")

        st.error("### 🎲 D. Project Valuation Room\n"
                "Orchestrates full-scale high-dimensional Monte Carlo simulation paths over thousands of runs. "
                "Transforms correlated random paths into discounted net cash flows and maps asset NPV risks.\n\n"
                "👉 *Navigate using the sidebar to: 4_🎲_Project_Valuation*")

    st.markdown("---")
    st.subheader("📊 Strategic Project Configuration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Installed Solar Capacity", "65.0 MW", "Nameplate DC")
    c2.metric("Target Battery Storage Block", "30.0 MW / 60.0 MWh", "2-Hour Duration")
    c3.metric("Project Evaluation Lifecycle", "20 Years", "Financial Anchor")

pg.run()