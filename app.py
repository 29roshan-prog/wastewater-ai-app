import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ✅ Load trained model
with open("wastewater_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Wastewater AI", layout="wide")
st.title("💧 AI-Powered Wastewater Treatment Predictor")

# Tabs for input
tab1, tab2 = st.tabs(["📋 Manual Input", "📁 Upload CSV"])

# ========================= TAB 1 ========================= #
with tab1:
    st.subheader("🧾 Enter Raw Wastewater Parameters")

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            bod = st.number_input("🧪 BOD (mg/l)", min_value=0.0, value=225.0)
            tss = st.number_input("📉 TSS (mg/l)", min_value=0.0, value=200.0)
            ph = st.number_input("⚗️ pH", min_value=0.0, value=7.5)
        with col2:
            cod = st.number_input("🧪 COD (mg/l)", min_value=0.0, value=500.0)
            oil = st.number_input("🧴 Oil & Grease (mg/l)", min_value=0.0, value=55.0)
        with col3:
            ammonical = st.number_input("💨 Ammonical Nitrogen (mg/l)", min_value=0.0, value=12.5)
            total_n = st.number_input("🌡️ Total Nitrogen (mg/l)", min_value=0.0, value=30.0)

    flow_rate = st.number_input("💧 Flow Rate (L/min)", min_value=0.0, value=150.0)

    st.markdown("---")

    if st.button("🚀 Predict Treated Output"):
        input_data = np.array([[bod, cod, tss, oil, ph, ammonical, total_n]])
        prediction = model.predict(input_data)[0]

        st.subheader("🎯 Treated Effluent Prediction")

        output_labels = [
            'BOD_treated', 'COD_treated', 'TSS_treated',
            'Oil_and_Grease_treated', 'pH_treated',
            'Ammonical_N_treated', 'Total_N_treated'
        ]

        raw_values = [bod, cod, tss, oil, ph, ammonical, total_n]

        for label, val in zip(output_labels, prediction):
            st.markdown(f"✅ **{label.replace('_', ' ')}**: `{val:.2f} mg/l`")

        st.subheader("📊 Raw vs Treated Parameter Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        x_labels = ['BOD', 'COD', 'TSS', 'Oil & Grease', 'pH', 'Ammonical N', 'Total N']
        ax.bar(x_labels, raw_values, width=0.4, label='Raw', align='center', color='tomato')
        ax.bar(x_labels, prediction, width=0.4, label='Treated', align='edge', color='skyblue')
        ax.set_ylabel('mg/L (except pH)')
        ax.set_title('Raw vs Treated Parameter Comparison')
        ax.legend()
        st.pyplot(fig)

        st.subheader("⚙️ Smart Motor Suggestion Based on Flow Rate")
        if flow_rate < 100:
            st.info("💡 Suggestion: Run blower/motor for **10 minutes/hour** to save electricity.")
        elif flow_rate < 250:
            st.info("⚡ Suggestion: Run blower/motor for **30 minutes/hour** (moderate flow).")
        else:
            st.success("🚀 Suggestion: Run blower/motor for **60 minutes/hour** (high flow detected).")

        st.subheader("🧪 Suggested Chlorine Dose for Disinfection")
        predicted_BOD = prediction[0]
        predicted_TSS = prediction[2]
        chlorine_dose = 0.1 * (predicted_BOD + predicted_TSS) + 0.5
        st.info(f"🧼 Based on treated BOD and TSS, suggested chlorine dose is: **{chlorine_dose:.2f} mg/L**")

# ========================= TAB 2 ========================= #
with tab2:
    st.subheader("📁 Upload CSV with Raw Parameters")
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.write("🧾 Raw Data Uploaded:")
            st.dataframe(df_raw)

            expected_cols = ['BOD_raw', 'COD_raw', 'TSS_raw', 'Oil_and_Grease_raw', 'pH_raw', 'Ammonical_N_raw', 'Total_N_raw']
            if all(col in df_raw.columns for col in expected_cols):

                predictions = model.predict(df_raw[expected_cols])
                output_df = pd.DataFrame(predictions, columns=[
                    'BOD_treated', 'COD_treated', 'TSS_treated',
                    'Oil_and_Grease_treated', 'pH_treated',
                    'Ammonical_N_treated', 'Total_N_treated'
                ])

                final_df = pd.concat([df_raw.reset_index(drop=True), output_df], axis=1)
                st.success("✅ Prediction Complete")
                st.dataframe(final_df)

                st.subheader("📊 BOD Reduction Graph")
                fig, ax = plt.subplots()
                ax.bar(final_df.index, final_df['BOD_raw'], label="Raw BOD", color="tomato")
                ax.bar(final_df.index, final_df['BOD_treated'], label="Treated BOD", color="skyblue")
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("BOD (mg/l)")
                ax.legend()
                st.pyplot(fig)

                csv_data = final_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Predicted Output as CSV",
                    data=csv_data,
                    file_name="predicted_output.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Your CSV must include these columns:\n" + ", ".join(expected_cols))
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ========================= SIDEBAR ========================= #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/727/727790.png", width=80)
    st.title("💧 Wastewater AI")
    st.markdown("🚀 Built with Random Forest")
    st.markdown("🧪 Predict treated effluent quality")
    st.markdown("📊 Visualize BOD, COD, and TSS reduction")
    st.markdown("⚙️ Includes smart flow-based motor logic")
    st.markdown("🧼 Suggests chlorine dose for disinfection")
    st.markdown("👨‍💻 Created by: **Roshan Prabhu**")
