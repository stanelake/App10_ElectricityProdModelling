import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("🔋 Solar, Demand & Battery Storage Interaction Hub")
st.markdown(
    "This environment isolates and maps operational interactions over a single representative path window. "
    "It demonstrates how the physical asset dispatch logic handles battery degradation thresholds and price spreads."
)

# Operational Sidebars
st.sidebar.header("🔋 Battery Configuration Matrix")
bat_power = st.sidebar.slider("Battery Max Power Limit (MW)", 5.0, 100.0, 30.0)
bat_energy = st.sidebar.slider("Battery Energy Storage Volume (MWh)", 10.0, 300.0, 60.0)
roundtrip_eff = st.sidebar.slider("Round-Trip Efficiency Factor (η)", 0.75, 0.98, 0.88)

# Generate a mock 48-bar (24 hour) interactive lookahead array
st.subheader("🔄 Single-Day Operations Timeline View")

times = pd.date_range("2026-01-15 00:00", periods=48, freq="30min")
solar_gen = np.clip(35.0 * np.sin(np.pi * np.arange(48) / 48) ** 2 - 2, 0, None)
solar_gen[0:12] = 0; solar_gen[36:48] = 0 # force night

base_demand = 45.0 + 15.0 * np.sin(2 * np.pi * np.arange(48)/48 - 1.5)
prices = 75.0 + 40.0 * np.sin(2 * np.pi * np.arange(48)/48 - 1.0)
# inject a severe evening peak spike
prices[32:38] += 250.0

# Simple operational dispatch script loop
soc = np.zeros(48)
soc[0] = bat_energy * 0.2  # start at 20%
battery_action = np.zeros(48)  # positive = charging, negative = discharging

for t in range(1, 48):
    # Charge battery during cheap solar generation solar blocks
    if solar_gen[t] > 15.0 and prices[t] < 90.0 and soc[t-1] < bat_energy:
        charge_amt = min(solar_gen[t] * 0.5, bat_power * 0.5, bat_energy - soc[t-1])
        battery_action[t] = charge_amt
        soc[t] = soc[t-1] + charge_amt * roundtrip_eff
    # Discharge battery during evening price spike periods
    elif prices[t] > 200.0 and soc[t-1] > 0:
        discharge_amt = min(bat_power * 0.5, soc[t-1])
        battery_action[t] = -discharge_amt
        soc[t] = soc[t-1] - discharge_amt
    else:
        soc[t] = soc[t-1]

net_export = solar_gen - battery_action

# Structural multi-axis chart outputs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

ax1.plot(times, solar_gen, color="gold", label="Solar Production Output (MW)", linewidth=2)
ax1.plot(times, base_demand, color="brown", label="Regional Micro-Demand Profile (MW)", linestyle="--")
ax1.bar(times, battery_action, color="teal", label="Battery Actions (Positive=Charge)", width=0.015)
ax1.set_ylabel("Power Matrices (MW)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.2)

ax1_twin = ax1.twinx()
ax1_twin.plot(times, prices, color="crimson", alpha=0.6, label="Spot Price Target ($/MWh)", linestyle=":")
ax1_twin.set_ylabel("Spot Prices ($/MWh)")
ax1_twin.legend(loc="upper right")

ax2.fill_between(times, 0, (soc / bat_energy) * 100, color="steelblue", alpha=0.3, label="State of Charge (%)")
ax2.plot(times, (soc / bat_energy) * 100, color="green", linewidth=1.5)
ax2.set_ylabel("Storage Volume State (%)")
ax2.set_ylim(-5, 105)
ax2.grid(True, alpha=0.2)

st.pyplot(fig)
plt.close()