import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import io

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Banarasi Products Analytics", page_icon="🏺", layout="wide")

COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#639922", "#D4537E",
          "#EF9F27", "#E24B4A", "#888780", "#085041"]

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("banarasi_products_survey_2000.csv")
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# HELPER: ENCODE FEATURES FOR ML
# ─────────────────────────────────────────────────────────────
ORDINAL_MAPS = {
    "Q1_age_group": ["18-24", "25-34", "35-44", "45-54", "55+"],
    "Q6_monthly_income": ["Below ₹25,000", "₹25,000 – ₹50,000", "₹50,001 – ₹1,00,000",
                          "₹1,00,001 – ₹2,00,000", "Above ₹2,00,000"],
    "Q4_city_tier": ["Rural", "Tier-3 / Small Town", "Tier-2", "Tier-1", "Metro"],
    "Q8_cultural_affinity": [
        "Little to no interest in traditional/handcrafted products",
        "Prefer modern/western; sometimes buy traditional as gifts",
        "Open to traditional if modern/contemporary design",
        "Appreciate traditional; buy occasionally for festivals/occasions",
        "Actively seek traditional products; core part of my identity"],
    "Q9_banarasi_familiarity": ["Not familiar at all",
                                 "Slightly familiar — heard of sarees only",
                                 "Somewhat familiar — know but haven't purchased",
                                 "Very familiar — purchased before"],
    "Q10_sustainability_importance": [
        "Not Important — focus on quality and price only",
        "Somewhat Important — nice to have",
        "Important — influences alongside other factors",
        "Very Important — primary reason for handcrafted"],
    "Q11_purchase_frequency": ["Never", "Rarely", "Twice a year", "Quarterly", "Monthly", "Weekly"],
    "Q23_per_txn_spend": ["Below ₹1,000", "₹1,000 – ₹5,000", "₹5,001 – ₹15,000",
                          "₹15,001 – ₹30,000", "₹30,001 – ₹50,000", "Above ₹50,000"],
    "Q25_premium_willingness": ["Definitely not", "Probably not", "Not sure",
                                 "Probably yes", "Yes, definitely"],
}

NOMINAL_COLS = ["Q2_gender", "Q5_occupation", "Q7_family_structure"]

FEATURE_COLS_CAT = list(ORDINAL_MAPS.keys()) + NOMINAL_COLS
BINARY_FEATURE_PREFIXES = ["Q12_", "Q13_", "Q14_", "Q15_", "Q16_", "Q26_", "Q27_"]

@st.cache_data
def prepare_ml_features(dataframe):
    feat = dataframe.copy()
    for col, order in ORDINAL_MAPS.items():
        mapping = {v: i for i, v in enumerate(order)}
        feat[col] = feat[col].map(mapping).fillna(0).astype(int)
    for col in NOMINAL_COLS:
        le = LabelEncoder()
        feat[col] = le.fit_transform(feat[col].astype(str))
    binary_cols = [c for c in feat.columns
                   if any(c.startswith(p) for p in BINARY_FEATURE_PREFIXES)]
    prod_cols = [c for c in feat.columns if c.startswith("Q16_product_") and "None" not in c]
    feat["product_interest_count"] = feat[prod_cols].sum(axis=1)
    barrier_cols = [c for c in feat.columns if c.startswith("Q26_barrier_")]
    feat["barrier_count"] = feat[barrier_cols].sum(axis=1)
    feature_names = list(ORDINAL_MAPS.keys()) + NOMINAL_COLS + binary_cols + \
                    ["product_interest_count", "barrier_count"]
    feature_names = [f for f in feature_names if f in feat.columns]
    return feat, feature_names

@st.cache_data
def get_binary_target(dataframe):
    return (dataframe["Q28_purchase_likelihood"].isin(["Highly Likely", "Likely"])).astype(int)

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/silk.png", width=60)
st.sidebar.title("Banarasi Products")
st.sidebar.caption("Data-Driven Decision Engine")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Executive Summary", "📈 Descriptive Analytics",
     "👥 Customer Segmentation", "🔗 Association Rules",
     "🤖 Predictive Models", "🎯 Prescriptive Actions",
     "📤 New Customer Scorer"],
)

# =================================================================
# PAGE 1: EXECUTIVE SUMMARY
# =================================================================
if page == "📊 Executive Summary":
    st.title("Executive Summary")
    st.caption("Founder's cockpit — key metrics at a glance")

    interested = df["Q28_purchase_likelihood"].isin(["Highly Likely", "Likely"]).sum()
    total = len(df)
    avg_spend_interested = df.loc[
        df["Q28_purchase_likelihood"].isin(["Highly Likely", "Likely"]),
        "Q24_annual_spending"].mean()
    total_addressable = interested * avg_spend_interested

    prod_cols = [c for c in df.columns if c.startswith("Q16_product_") and "None" not in c]
    top_product = max(prod_cols, key=lambda c: df[c].sum())
    top_product_name = top_product.replace("Q16_product_", "").replace("_", " ")

    barrier_cols = [c for c in df.columns if c.startswith("Q26_barrier_") and "No_concerns" not in c]
    top_barrier = max(barrier_cols, key=lambda c: df[c].sum())
    top_barrier_name = top_barrier.replace("Q26_barrier_", "").replace("_", " ")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interested Customers", f"{interested:,} / {total:,}",
              f"{interested/total*100:.0f}% conversion potential")
    c2.metric("Avg Annual Spend (Interested)", f"₹{avg_spend_interested:,.0f}")
    c3.metric("Total Addressable Revenue", f"₹{total_addressable/1e7:.1f} Cr")
    c4.metric("Top Barrier", top_barrier_name[:30])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Purchase likelihood distribution")
        order = ["Highly Likely", "Likely", "Neutral", "Unlikely", "Highly Unlikely"]
        vc = df["Q28_purchase_likelihood"].value_counts().reindex(order).fillna(0)
        fig = px.bar(x=vc.index, y=vc.values,
                     color=vc.index,
                     color_discrete_sequence=["#1D9E75", "#5DCAA5", "#EF9F27", "#D85A30", "#E24B4A"],
                     labels={"x": "", "y": "Count"})
        fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Product interest rates")
        prod_rates = {c.replace("Q16_product_", "").replace("_", " "): df[c].mean()*100
                      for c in prod_cols}
        prod_df = pd.DataFrame(prod_rates.items(), columns=["Product", "Interest %"])
        prod_df = prod_df.sort_values("Interest %", ascending=True)
        fig = px.bar(prod_df, x="Interest %", y="Product", orientation="h",
                     color_discrete_sequence=["#534AB7"])
        fig.update_layout(height=350, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Spending distribution (annual)")
        fig = px.histogram(df, x="Q24_annual_spending", nbins=50,
                           color_discrete_sequence=["#1D9E75"],
                           labels={"Q24_annual_spending": "Annual Spending (₹)"})
        fig.update_layout(height=300, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Top 5 states by interest")
        interested_df = df[df["Q28_purchase_likelihood"].isin(["Highly Likely", "Likely"])]
        state_vc = interested_df["Q3_state"].value_counts().head(5)
        fig = px.bar(x=state_vc.index, y=state_vc.values,
                     color_discrete_sequence=["#378ADD"],
                     labels={"x": "State", "y": "Interested Respondents"})
        fig.update_layout(height=300, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGE 2: DESCRIPTIVE ANALYTICS
# =================================================================
elif page == "📈 Descriptive Analytics":
    st.title("Descriptive Analytics")
    st.caption("Understanding the market landscape — what is happening?")

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Product Demand", "Spending Analysis", "Channels & Barriers"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Q1_age_group", color="Q2_gender",
                               barmode="group", color_discrete_sequence=COLORS,
                               category_orders={"Q1_age_group": ["18-24","25-34","35-44","45-54","55+"]},
                               labels={"Q1_age_group": "Age Group", "Q2_gender": "Gender"})
            fig.update_layout(title="Age × Gender distribution", height=400, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ct_order = ["Metro", "Tier-1", "Tier-2", "Tier-3 / Small Town", "Rural"]
            fig = px.histogram(df, x="Q4_city_tier", color="Q6_monthly_income",
                               barmode="stack", color_discrete_sequence=COLORS,
                               category_orders={"Q4_city_tier": ct_order},
                               labels={"Q4_city_tier": "City Tier", "Q6_monthly_income": "Income"})
            fig.update_layout(title="City tier × Income distribution", height=400, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(df, x="Q8_cultural_affinity",
                               color_discrete_sequence=["#534AB7"],
                               labels={"Q8_cultural_affinity": "Cultural Affinity"})
            fig.update_layout(title="Cultural affinity distribution", height=380,
                              margin=dict(t=40), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.histogram(df, x="Q9_banarasi_familiarity",
                               color_discrete_sequence=["#1D9E75"],
                               labels={"Q9_banarasi_familiarity": "Familiarity"})
            fig.update_layout(title="Banarasi product familiarity", height=380,
                              margin=dict(t=40), xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        prod_cols = [c for c in df.columns if c.startswith("Q16_product_") and "None" not in c]
        prod_by_city = []
        for ct in ["Metro", "Tier-1", "Tier-2", "Tier-3 / Small Town", "Rural"]:
            sub = df[df["Q4_city_tier"] == ct]
            for pc in prod_cols:
                pname = pc.replace("Q16_product_", "").replace("_", " ")
                prod_by_city.append({"City Tier": ct, "Product": pname,
                                     "Interest %": sub[pc].mean() * 100})
        pbc_df = pd.DataFrame(prod_by_city)
        fig = px.bar(pbc_df, x="Product", y="Interest %", color="City Tier",
                     barmode="group", color_discrete_sequence=COLORS)
        fig.update_layout(title="Product interest by city tier", height=450, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("State-wise interest heatmap (top 10 states)")
        top_states = df["Q3_state"].value_counts().head(10).index.tolist()
        state_prod = []
        for s in top_states:
            sub = df[df["Q3_state"] == s]
            for pc in prod_cols:
                pname = pc.replace("Q16_product_", "").replace("_", " ")
                state_prod.append({"State": s, "Product": pname,
                                   "Interest %": round(sub[pc].mean() * 100, 1)})
        sp_df = pd.DataFrame(state_prod)
        pivot = sp_df.pivot(index="State", columns="Product", values="Interest %")
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale="Teal",
                        aspect="auto")
        fig.update_layout(height=450, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="Q6_monthly_income", y="Q24_annual_spending",
                         color_discrete_sequence=["#534AB7"],
                         labels={"Q6_monthly_income": "Monthly Income",
                                 "Q24_annual_spending": "Annual Spending (₹)"})
            fig.update_layout(title="Annual spending by income bracket", height=400, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x="Q8_cultural_affinity", y="Q24_annual_spending",
                         color_discrete_sequence=["#1D9E75"],
                         labels={"Q8_cultural_affinity": "Cultural Affinity",
                                 "Q24_annual_spending": "Annual Spending (₹)"})
            fig.update_layout(title="Annual spending by cultural affinity", height=400,
                              margin=dict(t=40), xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df, x="Q25_premium_willingness", color="Q6_monthly_income",
                           barmode="group", color_discrete_sequence=COLORS,
                           category_orders={"Q25_premium_willingness": list(ORDINAL_MAPS["Q25_premium_willingness"])},
                           labels={"Q25_premium_willingness": "Premium Willingness",
                                   "Q6_monthly_income": "Income"})
        fig.update_layout(title="Premium willingness vs income", height=400, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        channel_cols = [c for c in df.columns if c.startswith("Q13_channel_")]
        ch_rates = {c.replace("Q13_channel_", "").replace("_", " ")[:40]: df[c].mean()*100
                    for c in channel_cols}
        ch_df = pd.DataFrame(ch_rates.items(), columns=["Channel", "Usage %"]).sort_values("Usage %", ascending=True)
        fig = px.bar(ch_df, x="Usage %", y="Channel", orientation="h",
                     color_discrete_sequence=["#378ADD"])
        fig.update_layout(title="Shopping & discovery channels", height=450, margin=dict(t=40, l=200))
        st.plotly_chart(fig, use_container_width=True)

        barrier_cols = [c for c in df.columns if c.startswith("Q26_barrier_")]
        br_rates = {c.replace("Q26_barrier_", "").replace("_", " ")[:40]: df[c].mean()*100
                    for c in barrier_cols}
        br_df = pd.DataFrame(br_rates.items(), columns=["Barrier", "Frequency %"]).sort_values("Frequency %", ascending=True)
        fig = px.bar(br_df, x="Frequency %", y="Barrier", orientation="h",
                     color_discrete_sequence=["#D85A30"])
        fig.update_layout(title="Purchase barriers (what stops customers)", height=400,
                          margin=dict(t=40, l=200))
        st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGE 3: CUSTOMER SEGMENTATION (CLUSTERING)
# =================================================================
elif page == "👥 Customer Segmentation":
    st.title("Customer Segmentation")
    st.caption("Diagnostic — who are our customers and why do they behave differently?")

    feat_df, feature_names = prepare_ml_features(df)
    X_clust = feat_df[feature_names].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    tab1, tab2, tab3 = st.tabs(["Optimal K Selection", "Cluster Profiles", "Cluster-wise Strategy"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            from sklearn.metrics import silhouette_score
            inertias, sil_scores = [], []
            K_range = range(2, 9)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X_scaled, km.labels_, sample_size=1000,
                                                    random_state=42))
            fig = px.line(x=list(K_range), y=inertias, markers=True,
                          labels={"x": "Number of Clusters (K)", "y": "Inertia"},
                          color_discrete_sequence=["#534AB7"])
            fig.update_layout(title="Elbow method", height=350, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=list(K_range), y=sil_scores,
                         labels={"x": "K", "y": "Silhouette Score"},
                         color_discrete_sequence=["#1D9E75"])
            fig.update_layout(title="Silhouette scores", height=350, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        optimal_k = list(K_range)[np.argmax(sil_scores)]
        st.success(f"Optimal K = **{optimal_k}** (highest silhouette score = {max(sil_scores):.3f})")

    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_scaled)
    df["Cluster"] = cluster_labels

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    with tab2:
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                         color=df["Cluster"].astype(str),
                         color_discrete_sequence=COLORS,
                         labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                                 "y": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                                 "color": "Cluster"},
                         opacity=0.6)
        fig.update_layout(title="Customer segments — PCA projection", height=500, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster profiles")
        prod_cols = [c for c in df.columns if c.startswith("Q16_product_") and "None" not in c]
        for cl in sorted(df["Cluster"].unique()):
            sub = df[df["Cluster"] == cl]
            with st.expander(f"Cluster {cl} — {len(sub)} members ({len(sub)/len(df)*100:.1f}%)", expanded=(cl == 0)):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Avg Annual Spend", f"₹{sub['Q24_annual_spending'].mean():,.0f}")
                m2.metric("Median Spend", f"₹{sub['Q24_annual_spending'].median():,.0f}")
                top_city = sub["Q4_city_tier"].mode().iloc[0] if len(sub) > 0 else "N/A"
                m3.metric("Dominant City Tier", top_city)
                pct_interested = sub["Q28_purchase_likelihood"].isin(["Highly Likely","Likely"]).mean()*100
                m4.metric("Purchase Intent", f"{pct_interested:.0f}%")

                cp1, cp2 = st.columns(2)
                with cp1:
                    top_prods = {pc.replace("Q16_product_","").replace("_"," "): sub[pc].mean()*100
                                 for pc in prod_cols}
                    tp_df = pd.DataFrame(top_prods.items(), columns=["Product","Interest %"])
                    tp_df = tp_df.sort_values("Interest %", ascending=True)
                    fig = px.bar(tp_df, x="Interest %", y="Product", orientation="h",
                                 color_discrete_sequence=[COLORS[cl % len(COLORS)]])
                    fig.update_layout(height=250, margin=dict(t=10, b=10, l=150))
                    st.plotly_chart(fig, use_container_width=True)
                with cp2:
                    demo_data = {
                        "Top Age": sub["Q1_age_group"].mode().iloc[0],
                        "Top Income": sub["Q6_monthly_income"].mode().iloc[0],
                        "Top Family": sub["Q7_family_structure"].mode().iloc[0],
                        "Top Affinity": sub["Q8_cultural_affinity"].str[:45].mode().iloc[0],
                        "Top Familiarity": sub["Q9_banarasi_familiarity"].str[:40].mode().iloc[0],
                    }
                    for k, v in demo_data.items():
                        st.write(f"**{k}:** {v}")

    with tab3:
        st.subheader("Cluster-wise discount and marketing strategy")
        strategy_data = []
        for cl in sorted(df["Cluster"].unique()):
            sub = df[df["Cluster"] == cl]
            avg_spend = sub["Q24_annual_spending"].mean()
            intent = sub["Q28_purchase_likelihood"].isin(["Highly Likely","Likely"]).mean()
            size = len(sub)
            revenue_potential = avg_spend * size * intent
            top_product = max(prod_cols, key=lambda c: sub[c].sum())
            top_product_name = top_product.replace("Q16_product_","").replace("_"," ")

            if avg_spend > 60000 and intent > 0.7:
                discount = "0-5% (Exclusivity)"
                strategy = "Artisan exclusives, early access"
            elif avg_spend > 25000:
                discount = "10-15% (Bundles)"
                strategy = "Festival combo packs, loyalty program"
            elif intent > 0.6:
                discount = "15-20% (Volume)"
                strategy = "Entry-price products, upsell later"
            else:
                discount = "5-10% (Awareness)"
                strategy = "Awareness campaigns, free samples"

            strategy_data.append({
                "Cluster": cl, "Size": size,
                "Avg Spend": f"₹{avg_spend:,.0f}",
                "Intent %": f"{intent*100:.0f}%",
                "Rev Potential": f"₹{revenue_potential/1e5:.1f}L",
                "Top Product": top_product_name[:30],
                "Discount Tier": discount,
                "Strategy": strategy
            })
        st.dataframe(pd.DataFrame(strategy_data), use_container_width=True, hide_index=True)

        st.subheader("Priority matrix — CLV vs Conversion probability")
        scatter_data = []
        for cl in sorted(df["Cluster"].unique()):
            sub = df[df["Cluster"] == cl]
            scatter_data.append({
                "Cluster": f"Cluster {cl}",
                "Avg Annual Spend (₹)": sub["Q24_annual_spending"].mean(),
                "Conversion Prob (%)": sub["Q28_purchase_likelihood"].isin(
                    ["Highly Likely","Likely"]).mean()*100,
                "Size": len(sub)
            })
        sc_df = pd.DataFrame(scatter_data)
        fig = px.scatter(sc_df, x="Avg Annual Spend (₹)", y="Conversion Prob (%)",
                         size="Size", color="Cluster", text="Cluster",
                         color_discrete_sequence=COLORS, size_max=60)
        avg_spend_mid = sc_df["Avg Annual Spend (₹)"].median()
        avg_conv_mid = sc_df["Conversion Prob (%)"].median()
        fig.add_hline(y=avg_conv_mid, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=avg_spend_mid, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_annotation(x=sc_df["Avg Annual Spend (₹)"].max()*0.95,
                           y=sc_df["Conversion Prob (%)"].max()*0.98,
                           text="LAUNCH TARGET", showarrow=False,
                           font=dict(color="#1D9E75", size=12))
        fig.add_annotation(x=sc_df["Avg Annual Spend (₹)"].min()*1.1,
                           y=sc_df["Conversion Prob (%)"].max()*0.98,
                           text="VOLUME PLAY", showarrow=False,
                           font=dict(color="#378ADD", size=12))
        fig.add_annotation(x=sc_df["Avg Annual Spend (₹)"].max()*0.95,
                           y=sc_df["Conversion Prob (%)"].min()*1.1,
                           text="NURTURE", showarrow=False,
                           font=dict(color="#EF9F27", size=12))
        fig.add_annotation(x=sc_df["Avg Annual Spend (₹)"].min()*1.1,
                           y=sc_df["Conversion Prob (%)"].min()*1.1,
                           text="DEPRIORITIZE", showarrow=False,
                           font=dict(color="#E24B4A", size=12))
        fig.update_layout(height=500, margin=dict(t=20))
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGE 4: ASSOCIATION RULE MINING
# =================================================================
elif page == "🔗 Association Rules":
    st.title("Association Rule Mining")
    st.caption("Diagnostic — what products and preferences go together?")

    tab1, tab2, tab3 = st.tabs(["Cross-Product Rules", "Within-Product Rules", "Barrier ↔ Trust Rules"])

    with tab1:
        st.subheader("Which products are bought together?")
        prod_cols = [c for c in df.columns if c.startswith("Q16_product_") and "None" not in c]
        prod_basket = df[prod_cols].copy()
        prod_basket.columns = [c.replace("Q16_product_", "").replace("_", " ") for c in prod_cols]
        prod_basket = prod_basket.astype(bool)

        min_sup = st.slider("Minimum support", 0.03, 0.30, 0.08, 0.01, key="prod_sup")
        freq_items = apriori(prod_basket, min_support=min_sup, use_colnames=True)

        if len(freq_items) > 0:
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.3,
                                     num_itemsets=len(freq_items))
            if len(rules) > 0:
                rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
                display_rules = rules[["antecedents_str", "consequents_str",
                                       "support", "confidence", "lift"]].copy()
                display_rules.columns = ["If Customer Likes", "They Also Like",
                                         "Support", "Confidence", "Lift"]
                display_rules = display_rules.sort_values("Lift", ascending=False)
                display_rules["Support"] = display_rules["Support"].map("{:.2%}".format)
                display_rules["Confidence"] = display_rules["Confidence"].map("{:.2%}".format)
                display_rules["Lift"] = display_rules["Lift"].map("{:.2f}".format)
                st.dataframe(display_rules, use_container_width=True, hide_index=True)

                st.subheader("Association network")
                fig = px.scatter(rules, x="support", y="confidence", size="lift",
                                 color="lift", hover_data=["antecedents_str", "consequents_str"],
                                 color_continuous_scale="Viridis",
                                 labels={"support": "Support", "confidence": "Confidence",
                                         "lift": "Lift"})
                fig.update_layout(height=450, margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rules found at this confidence threshold. Try lowering support.")
        else:
            st.info("No frequent itemsets found. Try lowering the minimum support.")

    with tab2:
        st.subheader("Saree type & color associations")
        saree_cols = [c for c in df.columns if c.startswith("Q17_saree_type_") or c.startswith("Q18_saree_color_")]
        saree_buyers = df[df["Q16_product_Banarasi_Silk_Sarees_and_Textiles"] == 1]
        if len(saree_buyers) > 50 and len(saree_cols) > 0:
            saree_basket = saree_buyers[saree_cols].copy()
            saree_basket.columns = [c.replace("Q17_saree_type_","Type: ").replace("Q18_saree_color_","Color: ").replace("_"," ")
                                    for c in saree_cols]
            saree_basket = saree_basket.astype(bool)
            freq_saree = apriori(saree_basket, min_support=0.06, use_colnames=True)
            if len(freq_saree) > 0:
                saree_rules = association_rules(freq_saree, metric="lift", min_threshold=1.0,
                                               num_itemsets=len(freq_saree))
                if len(saree_rules) > 0:
                    saree_rules["antecedents_str"] = saree_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                    saree_rules["consequents_str"] = saree_rules["consequents"].apply(lambda x: ", ".join(list(x)))
                    sr_disp = saree_rules[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
                    sr_disp.columns = ["If They Prefer","They Also Prefer","Support","Confidence","Lift"]
                    sr_disp = sr_disp.sort_values("Lift", ascending=False).head(20)
                    sr_disp["Support"] = sr_disp["Support"].map("{:.2%}".format)
                    sr_disp["Confidence"] = sr_disp["Confidence"].map("{:.2%}".format)
                    sr_disp["Lift"] = sr_disp["Lift"].map("{:.2f}".format)
                    st.dataframe(sr_disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No saree association rules found at current thresholds.")
            else:
                st.info("Not enough frequent saree itemsets.")

        st.subheader("Paan combination preferences")
        paan_cols = [c for c in df.columns if c.startswith("Q21_paan_")]
        paan_buyers = df[df["Q16_product_Premium_Banarasi_Paan"] == 1]
        if len(paan_buyers) > 50 and len(paan_cols) > 0:
            paan_basket = paan_buyers[paan_cols].copy()
            paan_basket.columns = [c.replace("Q21_paan_","").replace("_"," ") for c in paan_cols]
            paan_basket = paan_basket.astype(bool)
            freq_paan = apriori(paan_basket, min_support=0.08, use_colnames=True)
            if len(freq_paan) > 0:
                paan_rules = association_rules(freq_paan, metric="lift", min_threshold=1.0,
                                              num_itemsets=len(freq_paan))
                if len(paan_rules) > 0:
                    paan_rules["antecedents_str"] = paan_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                    paan_rules["consequents_str"] = paan_rules["consequents"].apply(lambda x: ", ".join(list(x)))
                    pr_disp = paan_rules[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
                    pr_disp.columns = ["Paan Choice","Also Likes","Support","Confidence","Lift"]
                    pr_disp = pr_disp.sort_values("Lift", ascending=False).head(15)
                    pr_disp["Support"] = pr_disp["Support"].map("{:.2%}".format)
                    pr_disp["Confidence"] = pr_disp["Confidence"].map("{:.2%}".format)
                    pr_disp["Lift"] = pr_disp["Lift"].map("{:.2f}".format)
                    st.dataframe(pr_disp, use_container_width=True, hide_index=True)
                else:
                    st.info("No paan rules found at this threshold.")
            else:
                st.info("Not enough frequent paan itemsets.")

    with tab3:
        st.subheader("What barriers link to which trust builders?")
        barrier_cols = [c for c in df.columns if c.startswith("Q26_barrier_") and "No_concerns" not in c]
        trust_cols = [c for c in df.columns if c.startswith("Q27_trust_") and "Nothing" not in c]
        bt_cols = barrier_cols + trust_cols
        bt_basket = df[bt_cols].copy()
        bt_basket.columns = [c.replace("Q26_barrier_","Barrier: ").replace("Q27_trust_","Trust: ").replace("_"," ")
                             for c in bt_cols]
        bt_basket = bt_basket.astype(bool)
        freq_bt = apriori(bt_basket, min_support=0.05, use_colnames=True)
        if len(freq_bt) > 0:
            bt_rules = association_rules(freq_bt, metric="confidence", min_threshold=0.4,
                                        num_itemsets=len(freq_bt))
            barrier_to_trust = bt_rules[
                bt_rules["antecedents"].apply(lambda x: any("Barrier" in str(i) for i in x)) &
                bt_rules["consequents"].apply(lambda x: any("Trust" in str(i) for i in x))
            ]
            if len(barrier_to_trust) > 0:
                barrier_to_trust["antecedents_str"] = barrier_to_trust["antecedents"].apply(lambda x: ", ".join(list(x)))
                barrier_to_trust["consequents_str"] = barrier_to_trust["consequents"].apply(lambda x: ", ".join(list(x)))
                bt_disp = barrier_to_trust[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
                bt_disp.columns = ["Customer Barrier","Trust Builder Needed","Support","Confidence","Lift"]
                bt_disp = bt_disp.sort_values("Confidence", ascending=False).head(15)
                bt_disp["Support"] = bt_disp["Support"].map("{:.2%}".format)
                bt_disp["Confidence"] = bt_disp["Confidence"].map("{:.2%}".format)
                bt_disp["Lift"] = bt_disp["Lift"].map("{:.2f}".format)
                st.dataframe(bt_disp, use_container_width=True, hide_index=True)
            else:
                st.info("No barrier → trust rules found. Try adjusting thresholds.")
        else:
            st.info("No frequent itemsets found for barrier/trust analysis.")

# =================================================================
# PAGE 5: PREDICTIVE MODELS
# =================================================================
elif page == "🤖 Predictive Models":
    st.title("Predictive Models")
    st.caption("Predictive — will they buy, and how much will they spend?")

    feat_df, feature_names = prepare_ml_features(df)
    y_class = get_binary_target(df)

    tab1, tab2 = st.tabs(["Classification: Purchase Intent", "Regression: Spending Prediction"])

    with tab1:
        st.subheader("Predicting customer purchase interest (binary)")
        X = feat_df[feature_names].fillna(0)
        y = y_class

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                             random_state=42, stratify=y)
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss",
                                     use_label_encoder=False),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }

        results = {}
        roc_data = {}

        for name, model in models.items():
            if name == "Logistic Regression":
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train_sm)
                X_te = scaler.transform(X_test)
                model.fit(X_tr, y_train_sm)
                y_pred = model.predict(X_te)
                y_proba = model.predict_proba(X_te)[:, 1]
            else:
                model.fit(X_train_sm, y_train_sm)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc_val = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            results[name] = {"Accuracy": acc, "Precision": prec,
                             "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc_val}
            roc_data[name] = (fpr, tpr)

        res_df = pd.DataFrame(results).T.reset_index()
        res_df.columns = ["Model"] + list(res_df.columns[1:])
        for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            res_df[c] = res_df[c].map("{:.4f}".format)
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC curves")
            fig = go.Figure()
            colors_roc = ["#534AB7", "#1D9E75", "#D85A30"]
            for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
                auc_val = results[name]["ROC-AUC"]
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{name} (AUC={auc_val})",
                                         line=dict(color=colors_roc[i], width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     name="Random", line=dict(color="gray", dash="dash")))
            fig.update_layout(height=400, margin=dict(t=20),
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Confusion matrix (best model)")
            best_model_name = max(results, key=lambda k: float(results[k]["F1-Score"]))
            best_model = models[best_model_name]
            if best_model_name == "Logistic Regression":
                y_pred_best = best_model.predict(scaler.transform(X_test))
            else:
                y_pred_best = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_best)
            fig = px.imshow(cm, text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Not Interested", "Interested"],
                            y=["Not Interested", "Interested"],
                            color_continuous_scale="Teal")
            fig.update_layout(height=400, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Feature importance ({best_model_name})")
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        else:
            importances = np.abs(best_model.coef_[0])
        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_imp = feat_imp.sort_values("Importance", ascending=True).tail(15)
        feat_imp["Feature"] = feat_imp["Feature"].str.replace("Q\\d+_", "", regex=True).str.replace("_", " ")
        fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=["#534AB7"])
        fig.update_layout(height=500, margin=dict(t=20, l=200))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Predicting annual spending (₹)")
        X = feat_df[feature_names].fillna(0)
        y_reg = np.log1p(df["Q24_annual_spending"])

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_reg, test_size=0.2, random_state=42)

        reg_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        }

        reg_results = {}
        best_reg_pred = None
        best_r2 = -999

        for name, model in reg_models.items():
            model.fit(X_train_r, y_train_r)
            y_pred_r = model.predict(X_test_r)
            y_actual = np.expm1(y_test_r)
            y_pred_actual = np.expm1(y_pred_r)
            y_pred_actual = np.maximum(y_pred_actual, 0)

            r2 = r2_score(y_actual, y_pred_actual)
            mae = mean_absolute_error(y_actual, y_pred_actual)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))

            reg_results[name] = {"R²": r2, "MAE (₹)": mae, "RMSE (₹)": rmse}

            if r2 > best_r2:
                best_r2 = r2
                best_reg_pred = y_pred_actual
                best_reg_actual = y_actual
                best_reg_name = name

        reg_df = pd.DataFrame(reg_results).T.reset_index()
        reg_df.columns = ["Model", "R²", "MAE (₹)", "RMSE (₹)"]
        reg_df["R²"] = reg_df["R²"].map("{:.4f}".format)
        reg_df["MAE (₹)"] = reg_df["MAE (₹)"].map("₹{:,.0f}".format)
        reg_df["RMSE (₹)"] = reg_df["RMSE (₹)"].map("₹{:,.0f}".format)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Predicted vs Actual ({best_reg_name})")
            fig = px.scatter(x=best_reg_actual, y=best_reg_pred,
                             opacity=0.4, color_discrete_sequence=["#1D9E75"],
                             labels={"x": "Actual Spending (₹)", "y": "Predicted Spending (₹)"})
            max_val = max(best_reg_actual.max(), best_reg_pred.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                     mode="lines", name="Perfect",
                                     line=dict(color="gray", dash="dash")))
            fig.update_layout(height=400, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader(f"Feature importance ({best_reg_name})")
            best_reg_model = reg_models[best_reg_name]
            if hasattr(best_reg_model, "feature_importances_"):
                reg_imp = best_reg_model.feature_importances_
            else:
                reg_imp = np.abs(best_reg_model.coef_)
            reg_feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": reg_imp})
            reg_feat_imp = reg_feat_imp.sort_values("Importance", ascending=True).tail(15)
            reg_feat_imp["Feature"] = reg_feat_imp["Feature"].str.replace("Q\\d+_", "", regex=True).str.replace("_", " ")
            fig = px.bar(reg_feat_imp, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=["#D85A30"])
            fig.update_layout(height=400, margin=dict(t=20, l=180))
            st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGE 6: PRESCRIPTIVE ACTIONS
# =================================================================
elif page == "🎯 Prescriptive Actions":
    st.title("Prescriptive Actions")
    st.caption("What should we do? — data-driven launch strategy")

    st.subheader("1. Recommended product bundles (from association rules)")
    prod_cols = [c for c in df.columns if c.startswith("Q16_product_") and "None" not in c]
    prod_basket = df[prod_cols].copy()
    prod_basket.columns = [c.replace("Q16_product_","").replace("_"," ") for c in prod_cols]
    prod_basket = prod_basket.astype(bool)
    freq_items = apriori(prod_basket, min_support=0.05, use_colnames=True)
    if len(freq_items) > 0:
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.3,
                                 num_itemsets=len(freq_items))
        if len(rules) > 0:
            rules_sorted = rules.sort_values("lift", ascending=False).head(5)
            bundle_names = [
                "Kashi Bridal Collection", "Heritage Home Bundle",
                "Artisan Gift Hamper", "Festival Essentials Pack",
                "Banaras Experience Box"
            ]
            for i, (_, rule) in enumerate(rules_sorted.iterrows()):
                items = list(rule["antecedents"]) + list(rule["consequents"])
                bname = bundle_names[i] if i < len(bundle_names) else f"Bundle {i+1}"
                conf = rule["confidence"]
                lift = rule["lift"]
                st.info(f"**{bname}:** {' + '.join(items)} — Confidence: {conf:.0%} | Lift: {lift:.2f}")
        else:
            st.write("Not enough association rules to generate bundles.")
    else:
        st.write("Not enough frequent itemsets.")

    st.divider()
    st.subheader("2. Segment-specific marketing playbook")

    feat_df, feature_names = prepare_ml_features(df)
    X_clust = feat_df[feature_names].fillna(0)
    scaler_c = StandardScaler()
    X_scaled_c = scaler_c.fit_transform(X_clust)
    from sklearn.metrics import silhouette_score
    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled_c)
        sil_scores.append(silhouette_score(X_scaled_c, labels, sample_size=1000, random_state=42))
    optimal_k = list(range(2,9))[np.argmax(sil_scores)]
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster_p6"] = km_final.fit_predict(X_scaled_c)

    channel_cols = [c for c in df.columns if c.startswith("Q13_channel_")]

    for cl in sorted(df["Cluster_p6"].unique()):
        sub = df[df["Cluster_p6"] == cl]
        avg_spend = sub["Q24_annual_spending"].mean()
        intent = sub["Q28_purchase_likelihood"].isin(["Highly Likely","Likely"]).mean()
        top_geo = sub["Q3_state"].value_counts().head(3).index.tolist()
        top_chan = max(channel_cols, key=lambda c: sub[c].sum())
        top_chan_name = top_chan.replace("Q13_channel_","").replace("_"," ")[:35]
        top_prod = max(prod_cols, key=lambda c: sub[c].sum())
        top_prod_name = top_prod.replace("Q16_product_","").replace("_"," ")

        with st.expander(f"Segment {cl}: {len(sub)} customers | ₹{avg_spend:,.0f} avg spend | {intent:.0%} conversion"):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Target Geography:** {', '.join(top_geo)}")
            c2.write(f"**Best Channel:** {top_chan_name}")
            c3.write(f"**Lead Product:** {top_prod_name}")
            clv = avg_spend * intent
            st.write(f"**Predicted CLV:** ₹{clv:,.0f}/year | **Total segment revenue potential:** ₹{clv * len(sub)/1e5:.1f}L/year")

    st.divider()
    st.subheader("3. Launch priority sequence")
    st.write("Based on the priority matrix analysis (CLV × Conversion), here is the recommended phased launch:")

    launch_data = []
    for cl in sorted(df["Cluster_p6"].unique()):
        sub = df[df["Cluster_p6"] == cl]
        avg_s = sub["Q24_annual_spending"].mean()
        conv = sub["Q28_purchase_likelihood"].isin(["Highly Likely","Likely"]).mean()
        score = avg_s * conv
        launch_data.append({"Segment": cl, "CLV Score": score, "Size": len(sub),
                            "Avg Spend": avg_s, "Conv Rate": conv})
    launch_df = pd.DataFrame(launch_data).sort_values("CLV Score", ascending=False)
    launch_df["Phase"] = [f"Phase {i+1}" for i in range(len(launch_df))]
    launch_df["Avg Spend"] = launch_df["Avg Spend"].map("₹{:,.0f}".format)
    launch_df["Conv Rate"] = launch_df["Conv Rate"].map("{:.0%}".format)
    launch_df["CLV Score"] = launch_df["CLV Score"].map("{:,.0f}".format)
    st.dataframe(launch_df[["Phase","Segment","Size","Avg Spend","Conv Rate","CLV Score"]],
                 use_container_width=True, hide_index=True)

# =================================================================
# PAGE 7: NEW CUSTOMER SCORER
# =================================================================
elif page == "📤 New Customer Scorer":
    st.title("New Customer Prediction Engine")
    st.caption("Upload new survey data → get purchase predictions + segment assignment + spending forecast")

    st.subheader("How it works")
    st.write("""
    1. Upload a CSV file with new survey respondents (same column format as original data).
    2. The trained models will predict: **Purchase Likelihood**, **Predicted Annual Spending**, and **Customer Segment**.
    3. Download the results as a CSV with predictions appended.
    """)

    st.subheader("Required columns (minimum)")
    st.code("""Q1_age_group, Q2_gender, Q3_state, Q4_city_tier, Q5_occupation,
Q6_monthly_income, Q7_family_structure, Q8_cultural_affinity,
Q9_banarasi_familiarity, Q10_sustainability_importance,
Q11_purchase_frequency, Q23_per_txn_spend, Q25_premium_willingness""", language="text")

    uploaded = st.file_uploader("Upload new customer CSV", type=["csv"])

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.write(f"Uploaded **{len(new_df)}** new records with **{len(new_df.columns)}** columns")
        st.dataframe(new_df.head(), use_container_width=True)

        with st.spinner("Training models on original data and scoring new customers..."):
            feat_df, feature_names = prepare_ml_features(df)
            y_class = get_binary_target(df)
            X_orig = feat_df[feature_names].fillna(0)

            clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            smote = SMOTE(random_state=42)
            X_sm, y_sm = smote.fit_resample(X_orig, y_class)
            clf.fit(X_sm, y_sm)

            y_reg = np.log1p(df["Q24_annual_spending"])
            reg = GradientBoostingRegressor(n_estimators=200, random_state=42)
            reg.fit(X_orig, y_reg)

            scaler_k = StandardScaler()
            X_scaled_k = scaler_k.fit_transform(X_orig)
            from sklearn.metrics import silhouette_score
            sil_scores_k = []
            for k in range(2, 9):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled_k)
                sil_scores_k.append(silhouette_score(X_scaled_k, labels, sample_size=1000, random_state=42))
            optimal_k_new = list(range(2,9))[np.argmax(sil_scores_k)]
            km_new = KMeans(n_clusters=optimal_k_new, random_state=42, n_init=10)
            km_new.fit(X_scaled_k)

            try:
                new_feat, _ = prepare_ml_features(new_df)
                available_features = [f for f in feature_names if f in new_feat.columns]
                missing_features = [f for f in feature_names if f not in new_feat.columns]
                for mf in missing_features:
                    new_feat[mf] = 0
                X_new = new_feat[feature_names].fillna(0)

                purchase_proba = clf.predict_proba(X_new)[:, 1]
                purchase_pred = clf.predict(X_new)
                spend_pred = np.expm1(reg.predict(X_new))
                spend_pred = np.maximum(spend_pred, 0)
                X_new_scaled = scaler_k.transform(X_new)
                cluster_pred = km_new.predict(X_new_scaled)

                new_df["Purchase_Probability_%"] = (purchase_proba * 100).round(1)
                new_df["Purchase_Prediction"] = np.where(purchase_pred == 1, "Interested", "Not Interested")
                new_df["Predicted_Annual_Spend_₹"] = spend_pred.round(0).astype(int)
                new_df["Customer_Segment"] = cluster_pred

                priority_score = purchase_proba * spend_pred
                new_df["Priority_Score"] = priority_score.round(0).astype(int)
                new_df = new_df.sort_values("Priority_Score", ascending=False)

                st.success("Scoring complete!")
                st.subheader("Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total New Customers", len(new_df))
                s2.metric("Predicted Interested",
                          f"{(purchase_pred == 1).sum()} ({(purchase_pred == 1).mean()*100:.0f}%)")
                s3.metric("Avg Predicted Spend", f"₹{spend_pred.mean():,.0f}")
                high_potential = ((purchase_proba > 0.7) & (spend_pred > 30000)).sum()
                s4.metric("High-Potential Leads", high_potential)

                st.subheader("Scored customers (sorted by priority)")
                display_cols = [c for c in new_df.columns if c in
                                ["respondent_id", "Q1_age_group", "Q3_state", "Q4_city_tier",
                                 "Q6_monthly_income", "Purchase_Probability_%",
                                 "Purchase_Prediction", "Predicted_Annual_Spend_₹",
                                 "Customer_Segment", "Priority_Score"]]
                if not display_cols:
                    display_cols = list(new_df.columns[-5:])
                st.dataframe(new_df[display_cols].head(20), use_container_width=True, hide_index=True)

                csv_buffer = io.StringIO()
                new_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download scored predictions (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="banarasi_new_customer_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error processing new data: {str(e)}")
                st.write("Please ensure your CSV has the same column names as the original survey data.")

    else:
        st.info("Upload a CSV file to get started. You can use a subset of the original survey data to test.")

        if st.button("Generate sample test file (10 rows from original data)"):
            sample = df.drop(columns=["persona", "Q28_purchase_likelihood",
                                       "Q24_annual_spending"], errors="ignore").sample(10, random_state=99)
            csv_buf = io.StringIO()
            sample.to_csv(csv_buf, index=False)
            st.download_button("Download sample CSV", csv_buf.getvalue(),
                               "sample_new_customers.csv", "text/csv")
