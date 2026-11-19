import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).parent
INCOME_COLUMN = "Annual Income (k$)"
SPENDING_COLUMN = "Spending Score (1-100)"

cluster_groups = {
    0: "Average Customers",
    1: "High Spenders",
    2: "High Spenders",
    3: "Low Spenders",
    4: "Low Spenders",
}

cluster_descriptions = {
    0: "Medium annual income and medium annual spending",
    1: "High annual income and high annual spending",
    2: "Low annual income and high annual spending",
    3: "High annual income and low annual spending",
    4: "Low annual income and low annual spending",
}


@st.cache_data
def load_customers():
    return pd.read_csv(PROJECT_DIR / "Mall_Customers.csv")


@st.cache_resource
def load_kmeans_model():
    with (PROJECT_DIR / "k_means.pkl").open("rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.markdown(
    """
    <style>
        div[data-testid="stMetric"] {
            background: #f4f7fb;
            border: 1px solid #e3eaf3;
            border-radius: 12px;
            padding: 16px;
            min-height: 110px;
        }
        div[data-testid="stMetricLabel"] p {
            color: #34445b;
            font-size: 1rem;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #173b67;
            font-size: 2rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

customers = load_customers()
kmeans = load_kmeans_model()
customer_features = customers[[INCOME_COLUMN, SPENDING_COLUMN]]
kmeans_labels = kmeans.predict(customer_features)
customers["K-Means Cluster"] = [
    f"Cluster {label} - {cluster_groups[int(label)]}" for label in kmeans_labels
]
customers["Customer Group"] = [
    cluster_groups[int(label)] for label in kmeans_labels
]

scaled_features = StandardScaler().fit_transform(customer_features)
dbscan_labels = DBSCAN(eps=0.40, min_samples=3).fit_predict(scaled_features)
customers["DBSCAN Cluster"] = np.where(
    dbscan_labels == -1,
    "Noise",
    "Cluster " + dbscan_labels.astype(str),
)

st.title("Mall Customer Segmentation")
st.write("Explore customer data and compare two clustering methods.")

st.sidebar.header("Dashboard filters")
age_range = st.sidebar.slider(
    "Age range",
    int(customers["Age"].min()),
    int(customers["Age"].max()),
    (int(customers["Age"].min()), int(customers["Age"].max())),
)
income_range = st.sidebar.slider(
    "Annual income range ($k)",
    int(customers[INCOME_COLUMN].min()),
    int(customers[INCOME_COLUMN].max()),
    (int(customers[INCOME_COLUMN].min()), int(customers[INCOME_COLUMN].max())),
)
genders = sorted(customers["Gender"].dropna().unique())
selected_gender = st.sidebar.selectbox("Gender", ["All"] + genders)

filtered_customers = customers[
    customers["Age"].between(age_range[0], age_range[1])
    & customers[INCOME_COLUMN].between(income_range[0], income_range[1])
]
if selected_gender != "All":
    filtered_customers = filtered_customers[
        filtered_customers["Gender"] == selected_gender
    ]

metric_columns = st.columns(3)
if filtered_customers.empty:
    metric_columns[0].metric("Total Customers", "0")
    metric_columns[1].metric("Average Annual Income", "N/A")
    metric_columns[2].metric("Average Spending Score", "N/A")
    st.info("No customers match these filters. Try widening the selected ranges.")
else:
    total_customers = len(filtered_customers)
    average_income = filtered_customers[INCOME_COLUMN].mean()
    average_spending = filtered_customers[SPENDING_COLUMN].mean()
    metric_columns[0].metric("Total Customers", f"{total_customers:,}")
    metric_columns[1].metric("Average Annual Income", f"${average_income:.1f}k")
    metric_columns[2].metric("Average Spending Score", f"{average_spending:.1f} / 100")
    st.caption("KPI values reflect the customers selected by the filters.")

    customer_tooltips = [
        alt.Tooltip(field="CustomerID", type="quantitative", title="Customer ID"),
        alt.Tooltip(field="Gender", type="nominal", title="Gender"),
        alt.Tooltip(field="Age", type="quantitative", title="Age"),
        alt.Tooltip(
            field=INCOME_COLUMN,
            type="quantitative",
            title="Annual Income ($k)",
        ),
        alt.Tooltip(
            field=SPENDING_COLUMN,
            type="quantitative",
            title="Spending Score",
        ),
    ]

    st.subheader("Customer distribution")
    distribution_chart = (
        alt.Chart(filtered_customers)
        .mark_circle(size=85, opacity=0.8)
        .encode(
            x=alt.X(
                field=INCOME_COLUMN,
                type="quantitative",
                title="Annual Income ($k)",
            ),
            y=alt.Y(field="Age", type="quantitative", title="Age (years)"),
            color=alt.Color(field="Gender", type="nominal", title="Gender"),
            tooltip=customer_tooltips,
        )
        .properties(
            title="Customer Distribution by Annual Income and Age",
            height=360,
        )
        .interactive()
    )
    st.altair_chart(distribution_chart, use_container_width=True)
    st.caption(
        "Each dot represents one customer. Hover over a dot to see their age, "
        "gender, income, and spending score."
    )

    st.subheader("K-Means customer clusters")
    cluster_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    cluster_order = [
        f"Cluster {label} - {cluster_groups[label]}" for label in range(5)
    ]
    kmeans_chart = (
        alt.Chart(filtered_customers)
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X(
                field=INCOME_COLUMN,
                type="quantitative",
                title="Annual Income ($k)",
            ),
            y=alt.Y(
                field=SPENDING_COLUMN,
                type="quantitative",
                title="Spending Score (1-100)",
            ),
            color=alt.Color(
                field="K-Means Cluster",
                type="nominal",
                title="Customer Group",
                scale=alt.Scale(domain=cluster_order, range=cluster_colors),
            ),
            tooltip=customer_tooltips
            + [
                alt.Tooltip(
                    field="K-Means Cluster",
                    type="nominal",
                    title="K-Means Cluster",
                ),
                alt.Tooltip(
                    field="Customer Group",
                    type="nominal",
                    title="Customer Group",
                ),
            ],
        )
        .properties(title="K-Means Customer Segments", height=360)
        .interactive()
    )
    st.altair_chart(kmeans_chart, use_container_width=True)
    st.caption(
        "Colors distinguish the five K-Means clusters. Hover over a customer "
        "to see their details and simple group name."
    )

    st.subheader("DBSCAN customer clusters")
    dbscan_chart = (
        alt.Chart(filtered_customers)
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X(
                field=INCOME_COLUMN,
                type="quantitative",
                title="Annual Income ($k)",
            ),
            y=alt.Y(
                field=SPENDING_COLUMN,
                type="quantitative",
                title="Spending Score (1-100)",
            ),
            color=alt.Color(
                field="DBSCAN Cluster",
                type="nominal",
                title="DBSCAN Cluster",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=customer_tooltips
            + [
                alt.Tooltip(
                    field="DBSCAN Cluster",
                    type="nominal",
                    title="DBSCAN Cluster",
                )
            ],
        )
        .properties(title="DBSCAN Customer Segments", height=360)
        .interactive()
    )
    st.altair_chart(dbscan_chart, use_container_width=True)
    st.caption(
        "DBSCAN groups nearby customers and labels outliers as Noise. Hover "
        "over a customer to see their details and cluster."
    )

st.sidebar.header("Predict a customer segment")
st.sidebar.write("Enter the customer's annual income and spending score.")
annual_income = st.sidebar.slider("Annual Income in $k", 0, 200, 1)
spending_score = st.sidebar.slider("Spending Score", 0, 100, 50)

if st.sidebar.button("Predict Cluster"):
    user_data = pd.DataFrame(
        [[annual_income, spending_score]],
        columns=[INCOME_COLUMN, SPENDING_COLUMN],
    )
    predicted_cluster = int(kmeans.predict(user_data)[0])
    st.sidebar.subheader(
        f"Customer Group: {cluster_groups.get(predicted_cluster, 'Unknown')}"
    )
    st.sidebar.write(f"Cluster {predicted_cluster}")
    st.sidebar.write(
        cluster_descriptions.get(predicted_cluster, "Description not available")
    )
