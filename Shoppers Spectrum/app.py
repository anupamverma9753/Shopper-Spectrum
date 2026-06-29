import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Set page configuration at the very top
st.set_page_config(page_title="Shoppers Spectrum Dashboard", page_icon="🛍️", layout="wide")


# SIDEBAR NAVIGATION

st.sidebar.title("Navigation")
st.sidebar.write("Analytical dashboards:")
page = st.sidebar.radio(
    "Go To Module:",
    ["🛒 Product Recommendations", "📊 Customer Segmentation"]
)


# MODULE 1: PRODUCT RECOMMENDATION MODULE

if page == "🛒 Product Recommendations":
     st.title("🛒 Product Recommendation Module")
     st.write("Find matching inventory options based on historical customer checkouts.")
     @st.cache_resource
     def load_recommendation_data():
         with open('Shoppers Spectrum/Models/item_similarity.pkl','rb') as f:
           return pickle.load(f)


     try:
       similarity_df=load_recommendation_data()
       product_list=similarity_df.index.tolist()

       if "product_select" not in st.session_state:
          st.session_state.product_select = ""
       if "selectbox_key" not in st.session_state:
          st.session_state.selectbox_key = 0
       options_list = [""] + product_list
       current_index = options_list.index(st.session_state.product_select) if st.session_state.product_select in options_list else 0
        # Search and Select Input with autocomplete matching unique description
       selected_product=st.selectbox("Search or Select a Product Name:",
                                  options=options_list,
                                  index=current_index,
                                  key=f"prod_search_{st.session_state.selectbox_key}", 
                                  format_func=lambda x: "Start typing product name..." if x=="" else x
                                 )


    # Buttons
       col1, col2 = st.columns([1, 3])

       with col1:
        # Trigger Button
          get_rec=st.button("Get Recommendations", type="primary")
       with col2:
        # Clear Selction Button
          if st.button("Clear Selection", type="secondary"):
              st.session_state.product_select = "" # Reset to empty string
              st.session_state.selectbox_key += 1
              st.rerun()
       if get_rec:
           if selected_product=="":
            st.warning("Please select a product from the dropdown first.")
           else:
               recommendations= similarity_df[selected_product].sort_values(ascending=False).iloc[1:6]

               st.markdown(f"####  Top 5 complementary items for **{selected_product}**:")

               # Render Styled Card using HTML Container
               for rank, (prod_name, score) in enumerate(recommendations.items(),1):
                  st.html(f"""
                    <div style="
                        background-color: #f8f9fa; 
                        padding: 12px 20px; 
                        border-radius: 6px; 
                        margin-bottom: 8px; 
                        border-left: 5px solid #ff4b4b;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        cursor: pointer;
                        box-shadow: 0px 1px 2px rgba(0,0,0,0.05);
                    ">
                        <span style="font-weight: 600; color: #212529;">#{rank} &nbsp; {prod_name}</span>
                        <span style="color: #6c757d; font-size: 0.85rem;">Match: {score*100:.1f}%</span>
                    </div>
                    """)
     except FileNotFoundError:
         st.error("⚠️ `item_similarity.pkl` not found. Run the recommendation script inside your notebook first.")








# MODULE 2: CUSTOMER SEGMENTATION MODULE

elif page == "📊 Customer Segmentation":
    st.title("📊 Customer Segmentation Module")
    st.write("Input a customer's purchasing parameters to predict their loyalty cluster.")


    @st.cache_resource
    def load_segmentation_module():
        with open('Shoppers Spectrum/Models/scaler.pkl','rb') as sf , open('Shoppers Spectrum/Models/kmeans_model.pkl', 'rb') as mf:
            return pickle.load(sf), pickle.load(mf)
    try:
       scaler, kmeans= load_segmentation_module()

       with st.form("segmentaion_form"):
           col_r, col_f, col_m=st.columns(3)
           with col_r:
               recency=st.number_input("Recency (Days since last purchase):", min_value=1,value=30, step=1)
           with col_f:
               frequency=st.number_input("Frequency (Total distinct orders):", min_value=1, value=3,step=1)
           with col_m:
               monetary= st.number_input("Monetary (Total Spend value in £):", min_value=0.01, value=500.00, step=10.00)
        
            # Formn Submit Button
           predict_btn=st.form_submit_button("Predict Cluster", type="primary")


       # handling prediction logic upon form submission
       if predict_btn:
          raw_inputs= np.array([[recency, frequency, monetary]])
          log_inputs=np.log1p(raw_inputs)


          scaled_inputs=scaler.transform(log_inputs)

          cluster_id=kmeans.predict(scaled_inputs)[0]

          cluster_labels= {
              0: ('High-Value Customer', '#DCFCE7', '#15803D', '💰'),  # Deep Emerald Green
              1: ('Customer At-Risk', '#FEE2E2', '#B91C1C', '⚠️'),     # Deep Coral Red
              2: ('Occasional Customer', '#FEF3C7', '#B45309', '⏱️'),  # Rich Amber Orange
              3: ('Regular Customer', '#DBEAFE', '#1D4ED8', '🔹')       # Classic Navy Blue
         }

          label, bg_color,text_color, icon =cluster_labels.get(cluster_id, ("Unknown Segment", "#F3F4F6", "#374151", "❓"))


        # Prediction Result Card
          st.markdown(f"### Predicted Cluster Result:")
          st.html(f"""
           <div style="
             background-color: {bg_color}; 
             padding: 20px; 
             border-radius: 8px; 
             border-left: 6px solid {text_color};
             margin-top: 10px;
             box-shadow: 0px 2px 4px rgba(0,0,0,0.08);
          ">
             <h3 style="margin: 0; color: {text_color}; font-family: sans-serif; font-weight: 800; font-size: 1.5rem; display: block;">
                {icon} {label}
            </h3>
             <p style="margin: 8px 0 0 0; color: #1F2937 !important; font-family: sans-serif; font-size: 1rem; font-weight: 500;">
                Assigned to Cluster ID index: <strong style="color: {text_color}; font-weight: 700;">{cluster_id}</strong> based on pipeline metrics.
            </p>
          </div>
          """)


    except FileNotFoundError:
      st.error("⚠️ `scaler.pkl` or `kmeans_model.pkl` files are missing! Run your notebook pipeline scripts first.")
