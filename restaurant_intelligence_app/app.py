import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from html import escape
import io
import os

st.set_page_config(page_title="Restaurant Intelligence", page_icon="🍽️", layout="wide")

# Warm, high-contrast styling shared by every module.
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8f8f3 0%, #f1f5f0 100%);
    color: #203b32;
}
[data-testid="stHeader"] {background: rgba(248, 248, 243, 0.88);}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #173f35 0%, #102e28 100%);
}
[data-testid="stSidebar"] * {color: #f3f5e9;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: #c4d6ca;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #a9c2b2;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {color: #f3f5e9;}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    width: 100%;
    min-height: 2.65rem;
    justify-content: flex-start;
    text-align: left;
    padding-left: 0.9rem;
    border-radius: 10px;
    box-shadow: none;
    transform: none;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"] {
    background: transparent;
    color: #d8e6dc;
    border-color: transparent;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(235, 246, 235, 0.1);
    color: #ffffff;
    border-color: rgba(235, 246, 235, 0.12);
    box-shadow: none;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
    background: #e7b66b;
    color: #183c31;
    border-color: #e7b66b;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] * {
    color: #183c31;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {
    background: #f0c77f;
    color: #183c31;
    border-color: #f0c77f;
    box-shadow: none;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover * {
    color: #183c31;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: rgba(244, 248, 243, 0.08);
    border: 1px dashed rgba(216, 230, 220, 0.45);
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {color: #d8e6dc;}
[data-testid="stSidebar"] hr {border-color: rgba(216, 230, 220, 0.18);}
h1, h2, h3 {color: #173f35; letter-spacing: -0.025em;}
h1 {font-weight: 800;}
h2, h3 {font-weight: 700;}
[data-testid="stCaptionContainer"] {color: #687b70;}
div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 18px 20px;
    border-radius: 16px;
    border: 1px solid #dce7dc;
    box-shadow: 0 5px 16px rgba(27, 62, 49, 0.06);
}
[data-testid="stMetricLabel"] {color: #64766a;}
[data-testid="stMetricValue"] {color: #173f35; font-weight: 750;}
.stButton > button {
    background: #176b50;
    color: #ffffff;
    border: 1px solid #176b50;
    border-radius: 11px;
    min-height: 2.8rem;
    font-weight: 700;
    transition: background-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}
.stButton > button:hover {
    background: #10563f;
    color: #ffffff;
    border-color: #10563f;
    box-shadow: 0 5px 14px rgba(23, 107, 80, 0.2);
    transform: translateY(-1px);
}
.stButton > button:focus-visible {outline: 3px solid #e6a25c; outline-offset: 2px;}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border-color: #dce7dc !important;
    border-radius: 17px;
    box-shadow: 0 6px 20px rgba(27, 62, 49, 0.055);
    transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #a7cbb4 !important;
    box-shadow: 0 10px 24px rgba(27, 62, 49, 0.1);
    transform: translateY(-2px);
}
[data-testid="stAlert"] {border-radius: 14px;}
[data-testid="stFileUploader"] section {
    background: #f4f8f3;
    border: 1px dashed #9bb9a2;
    border-radius: 14px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {border-radius: 12px; overflow: hidden;}
hr {border-color: #dce7dc;}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {transition-duration: 0.01ms !important; scroll-behavior: auto !important;}
}
</style>
""", unsafe_allow_html=True)

st.title("🍽️ Restaurant Intelligence")
st.caption("A smarter workspace for restaurant data, predictions, and visual analysis")

if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"
if "restaurant_df" not in st.session_state:
    st.session_state.restaurant_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

def navigate_to(page_name):
    st.session_state.active_page = page_name

def load_data(uploaded):
    if uploaded is not None:
        try:
            return pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    return None

def device_file_uploader(label, file_types, key):
    return st.file_uploader(
        label,
        type=file_types,
        key=key,
        help="Choose a file from this device. On a phone, use the Photos or Files picker; on a PC, browse to a file.",
    )

@st.cache_resource(show_spinner=False)
def load_food_classifier():
    from transformers import pipeline

    return pipeline("image-classification", model="nateraw/food")

def normalize_col(df, candidates):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        cl = str(c).lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None

navigation_groups = [
    ("WORKSPACE", [("🏠", "Overview")]),
    ("RESTAURANT DATA", [("📊", "Restaurant Analytics")]),
    ("PREDICTIONS", [("📈", "Rating Prediction")]),
    ("DISCOVERY", [("🍴", "Restaurant Finder")]),
    ("IMAGE STUDIO", [
        ("🍲", "Food Image Recognition"),
        ("🧾", "Menu OCR Scanner"),
        ("🎨", "Visual Image Analysis"),
    ]),
    ("🧪 REPORTS", [("📋", "Model Evaluation")]),
]
available_pages = {
    destination
    for _, destinations in navigation_groups
    for _, destination in destinations
}
if st.session_state.active_page not in available_pages:
    st.session_state.active_page = "Overview"

with st.sidebar:
    st.markdown("## 🍽️ Restaurant IQ")
    st.caption("INSIGHTS WORKSPACE")
    st.divider()

    for group_name, destinations in navigation_groups:
        st.markdown(
            f'<p style="font-size:0.68rem;letter-spacing:0.12em;'
            f'font-weight:700;color:#9eb9a8;margin:0.8rem 0 0.35rem;">'
            f'{group_name}</p>',
            unsafe_allow_html=True,
        )
        for icon, destination in destinations:
            st.button(
                f"{icon}   {destination}",
                key=f"sidebar_nav_{destination}",
                type="primary" if st.session_state.active_page == destination else "secondary",
                use_container_width=True,
                on_click=navigate_to,
                args=(destination,),
            )

    st.divider()
    st.markdown("#### 📊 Dataset")
    uploaded_sidebar = st.file_uploader(
        "Upload a restaurant CSV",
        type=["csv"],
        key="sidebar_csv",
        help="Upload once and use this dataset across analytics and prediction modules.",
        label_visibility="collapsed",
    )
    if uploaded_sidebar is not None:
        loaded = load_data(uploaded_sidebar)
        if loaded is not None:
            st.session_state.restaurant_df = loaded
            st.session_state.dataset_name = uploaded_sidebar.name

    sidebar_df = st.session_state.restaurant_df
    if sidebar_df is None:
        st.caption("No dataset loaded yet. Upload a CSV here to unlock analytics and predictions.")
    else:
        dataset_name = escape(str(st.session_state.dataset_name or "Restaurant dataset"))
        st.markdown(
            f'<div style="background:rgba(232,246,233,0.09);border:1px solid '
            f'rgba(214,235,218,0.18);border-radius:12px;padding:0.75rem 0.85rem;">'
            f'<div style="font-size:0.82rem;font-weight:700;color:#f3f5e9;'
            f'overflow-wrap:anywhere;">✓ {dataset_name}</div>'
            f'<div style="font-size:0.75rem;color:#c4d6ca;margin-top:0.3rem;">'
            f'{len(sidebar_df):,} rows · {len(sidebar_df.columns):,} columns</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Your uploaded data stays in this browser session.")

page = st.session_state.active_page

if page == "Overview":
    st.subheader("Your restaurant data, made useful")
    st.write("Explore restaurant trends, estimate ratings, classify cuisine labels, scan menu images, and inspect image quality from one place.")
    c1,c2,c3,c4 = st.columns(4)
    df = st.session_state.restaurant_df
    c1.metric("Dataset", "Loaded" if df is not None else "Not loaded")
    c2.metric("Rows", f"{len(df):,}" if df is not None else "—")
    c3.metric("Modules", "7")
    c4.metric("Image tools", "3")
    st.markdown("### Pick a place to start")
    st.caption("Choose any workspace below. Your uploaded dataset stays available as you move between modules.")
    shortcuts = [
        ("Restaurant Analytics", "Explore ratings, votes, costs, cuisines, and cities.", "Explore dataset"),
        ("Rating Prediction", "Train a model and estimate ratings for a custom restaurant.", "Build a rating model"),
        ("Restaurant Finder", "Find highly rated restaurants within your budget.", "Find restaurants"),
        ("Food Image Recognition", "Get likely dish labels from an uploaded food photo.", "Recognise a dish"),
        ("Menu OCR Scanner", "Extract editable text from a menu image.", "Scan a menu"),
        ("Visual Image Analysis", "Measure brightness, contrast, sharpness, and colour.", "Analyse an image"),
        ("Model Evaluation", "Review reference results and your latest trained model.", "View model results"),
    ]
    for start in range(0, len(shortcuts), 3):
        columns = st.columns(3)
        for column, (target, description, action) in zip(columns, shortcuts[start:start + 3]):
            with column.container(border=True):
                st.markdown(f"#### {target}")
                st.write(description)
                st.button(
                    action,
                    key=f"navigate_{target}",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=(target,),
                )
    st.info("Predictions depend on the dataset and models available. Treat them as estimates, not guarantees.")

elif page == "Restaurant Analytics":
    st.subheader("Restaurant Analytics")
    uploaded = device_file_uploader("Upload restaurant dataset (CSV)", ["csv"], "analytics_csv")
    if uploaded:
        df = load_data(uploaded)
        if df is not None:
            st.session_state.restaurant_df = df
            st.session_state.dataset_name = uploaded.name
    df = st.session_state.restaurant_df
    if df is None:
        st.info("Upload your Zomato CSV to begin. The app does not invent sample business data.")
    else:
        rating_col = normalize_col(df, ["Aggregate rating", "rating"])
        votes_col = normalize_col(df, ["Votes", "vote"])
        cost_col = normalize_col(df, ["Average Cost for two", "cost for two", "average cost"])
        cuisine_col = normalize_col(df, ["Cuisines", "Cuisine"])
        city_col = normalize_col(df, ["City"])
        x1,x2,x3 = st.columns(3)
        x1.metric("Restaurant records", f"{len(df):,}")
        x2.metric("Columns / features", f"{len(df.columns)}")
        x3.metric("Average rating", f"{pd.to_numeric(df[rating_col], errors='coerce').mean():.2f}" if rating_col else "Not found")
        if rating_col:
            ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()
            st.markdown("#### Rating distribution")
            st.bar_chart(ratings.value_counts().sort_index())
        left,right = st.columns(2)
        if city_col and rating_col:
            city_stats = df.assign(_rating=pd.to_numeric(df[rating_col], errors="coerce")).groupby(city_col)["_rating"].mean().dropna().sort_values(ascending=False).head(15)
            left.markdown("#### Highest average ratings by city")
            left.bar_chart(city_stats)
        if cuisine_col:
            top_cuisines = df[cuisine_col].fillna("Unknown").astype(str).str.split(",").explode().str.strip().value_counts().head(15)
            right.markdown("#### Most common cuisine labels")
            right.bar_chart(top_cuisines)
        if votes_col and rating_col:
            scatter = df[[votes_col, rating_col]].copy()
            scatter[votes_col] = pd.to_numeric(scatter[votes_col], errors="coerce")
            scatter[rating_col] = pd.to_numeric(scatter[rating_col], errors="coerce")
            st.markdown("#### Votes vs rating (sample)")
            st.scatter_chart(scatter.dropna().head(3000), x=votes_col, y=rating_col)
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(100), use_container_width=True)

elif page == "Rating Prediction":
    st.subheader("Rating Prediction")
    df = st.session_state.restaurant_df
    if df is None:
        up = device_file_uploader("Upload restaurant CSV", ["csv"], "rating_csv")
        if up:
            df = load_data(up)
            st.session_state.restaurant_df = df
            if df is not None:
                st.session_state.dataset_name = up.name
    if df is None:
        st.info("Upload a dataset in Restaurant Analytics first.")
    else:
        target = normalize_col(df, ["Aggregate rating", "rating"])
        if not target:
            st.error("A target column such as 'Aggregate rating' is required.")
        else:
            from sklearn.model_selection import KFold, cross_validate, train_test_split
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.linear_model import LinearRegression
            work = df.copy()
            work = work.dropna(subset=[target])
            y = pd.to_numeric(work[target], errors="coerce")
            work = work.loc[y.notna()].copy()
            y = pd.to_numeric(work[target], errors="coerce")
            drop_candidates = [target]
            for c in work.columns:
                if str(c).strip().lower() in ["restaurant id", "restaurant name", "address", "url", "phone"]:
                    drop_candidates.append(c)
            X = work.drop(columns=list(set(drop_candidates)), errors="ignore")
            # Avoid leakage from fields that directly encode the target.
            X = X.select_dtypes(include=["number", "object", "bool"])
            if len(X) < 30 or X.shape[1] == 0:
                st.error("Not enough usable data/features to train a model.")
            else:
                num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
                cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
                pre = ColumnTransformer([
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), cat_cols)
                ], remainder="drop")
                model_choice = st.selectbox(
                    "Choose model",
                    ["Auto-select best", "Decision Tree", "Linear Regression", "Random Forest", "Extra Trees"],
                )
                if st.button("Train and evaluate model"):
                    from sklearn.tree import DecisionTreeRegressor
                    estimators = {
                        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=42),
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1),
                        "Extra Trees": ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1),
                    }
                    candidates = estimators if model_choice == "Auto-select best" else {model_choice: estimators[model_choice]}
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    comparison = []
                    for name, estimator in candidates.items():
                        candidate_pipe = Pipeline([("preprocess", pre), ("model", estimator)])
                        scores = cross_validate(
                            candidate_pipe,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring="neg_root_mean_squared_error",
                            n_jobs=1,
                        )
                        comparison.append({
                            "Model": name,
                            "5-fold CV RMSE": -float(scores["test_score"].mean()),
                        })
                    comparison.sort(key=lambda row: row["5-fold CV RMSE"])
                    best_model = comparison[0]["Model"]
                    pipe = Pipeline([("preprocess", pre), ("model", estimators[best_model])])
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)
                    mse = mean_squared_error(y_test, pred)
                    r2 = r2_score(y_test, pred)
                    st.session_state.rating_pipe = pipe
                    st.session_state.rating_columns = X.columns.tolist()
                    st.session_state.rating_metrics = {
                        "model": best_model,
                        "mse": mse,
                        "r2": r2,
                        "cv_rmse": comparison[0]["5-fold CV RMSE"],
                    }
                    st.markdown("#### Cross-validation model comparison")
                    st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
                    a,b,c = st.columns(3)
                    a.metric("Selected model", best_model)
                    b.metric("Holdout RMSE", f"{mse ** 0.5:.4f}")
                    c.metric("Holdout R²", f"{r2:.4f}")
                    st.caption("The model is selected using 5-fold cross-validation on the training split, then evaluated on a separate 20% holdout. Results depend on your uploaded data.")
                if "rating_pipe" in st.session_state:
                    st.markdown("#### Predict for a custom restaurant")
                    pipe = st.session_state.rating_pipe
                    vals = {}
                    cols = st.columns(2)
                    for i, col in enumerate(st.session_state.rating_columns):
                        s = df[col]
                        with cols[i % 2]:
                            if pd.api.types.is_numeric_dtype(s):
                                vals[col] = st.number_input(str(col), value=float(pd.to_numeric(s, errors="coerce").median()) if pd.notna(pd.to_numeric(s, errors="coerce").median()) else 0.0)
                            else:
                                opts = s.dropna().astype(str).value_counts().head(30).index.tolist()
                                vals[col] = st.selectbox(str(col), opts) if opts else ""
                    if st.button("Predict rating"):
                        pred = pipe.predict(pd.DataFrame([vals]))[0]
                        st.success(f"Estimated rating: {pred:.2f} / 5")
                        st.caption("This is a model estimate and may not match a real customer rating.")

elif page == "Restaurant Finder":
    st.subheader("Find restaurants by budget and rating")
    df = st.session_state.restaurant_df
    if df is None:
        st.info("Upload your restaurant CSV from the sidebar to find matching restaurants.")
    else:
        cost_col = normalize_col(df, ["Average Cost for two", "cost for two", "average cost"])
        rating_col = normalize_col(df, ["Aggregate rating", "rating"])
        name_col = normalize_col(df, ["Restaurant Name", "Name"])
        city_col = normalize_col(df, ["City"])
        cuisine_col = normalize_col(df, ["Cuisines", "Cuisine"])

        if not cost_col or not rating_col:
            missing = []
            if not cost_col:
                missing.append("average cost for two")
            if not rating_col:
                missing.append("rating")
            st.warning(
                "The uploaded CSV needs these columns to find restaurants: "
                + ", ".join(missing)
                + "."
            )
        else:
            costs = pd.to_numeric(
                df[cost_col].astype(str).str.replace(r"[^0-9.-]", "", regex=True),
                errors="coerce",
            )
            ratings = pd.to_numeric(df[rating_col], errors="coerce")
            usable = costs.notna() & ratings.notna()
            if not usable.any():
                st.warning("No rows have both a valid cost-for-two value and a rating.")
            else:
                valid_costs = costs[usable]
                valid_ratings = ratings[usable]
                cost_min = max(0, int(np.floor(valid_costs.min())))
                cost_max = max(cost_min + 1, int(np.ceil(valid_costs.max())))
                rating_min = float(np.floor(valid_ratings.min()))
                rating_max = float(np.ceil(valid_ratings.max()))
                if rating_max <= rating_min:
                    rating_max = rating_min + 1.0
                rating_default = min(max(rating_min + 0.7 * (rating_max - rating_min), rating_min), rating_max)

                st.write(
                    "Choose a cost range for two people and a minimum rating. "
                    "Results come from matching restaurants in your uploaded dataset, not generated predictions."
                )
                with st.form("restaurant_finder_filters"):
                    cost_columns = st.columns(2)
                    with cost_columns[0]:
                        minimum_cost = st.number_input(
                            "Minimum cost for two (₹)",
                            min_value=cost_min,
                            max_value=cost_max,
                            value=cost_min,
                            step=100,
                        )
                    with cost_columns[1]:
                        maximum_cost = st.number_input(
                            "Maximum cost for two (₹)",
                            min_value=cost_min,
                            max_value=cost_max,
                            value=cost_max,
                            step=100,
                        )

                    filter_columns = st.columns(2)
                    with filter_columns[0]:
                        minimum_rating = st.slider(
                            "Minimum rating",
                            min_value=rating_min,
                            max_value=rating_max,
                            value=rating_default,
                            step=0.1,
                        )
                    with filter_columns[1]:
                        if city_col:
                            cities = sorted(df[city_col].dropna().astype(str).unique().tolist())
                            selected_city = st.selectbox("City", ["Any city"] + cities)
                        else:
                            selected_city = "Any city"
                            st.caption("City filter unavailable: no City column found.")
                    search = st.form_submit_button("Find restaurants", use_container_width=True)

                if search:
                    if minimum_cost > maximum_cost:
                        st.session_state.restaurant_finder_last_search = None
                        st.error("Minimum cost must be less than or equal to maximum cost.")
                    else:
                        st.session_state.restaurant_finder_last_search = {
                            "minimum_cost": minimum_cost,
                            "maximum_cost": maximum_cost,
                            "minimum_rating": minimum_rating,
                            "city": selected_city,
                        }

                finder_filters = st.session_state.get("restaurant_finder_last_search")
                if finder_filters:
                    matches = df.loc[
                        usable
                        & costs.between(
                            finder_filters["minimum_cost"],
                            finder_filters["maximum_cost"],
                        )
                        & ratings.ge(finder_filters["minimum_rating"])
                    ].copy()
                    matches["_finder_rating"] = ratings.loc[matches.index]
                    matches["_finder_cost"] = costs.loc[matches.index]
                    if finder_filters["city"] != "Any city" and city_col:
                        matches = matches[
                            matches[city_col].astype(str).eq(finder_filters["city"])
                        ]

                    matches = matches.sort_values(
                        ["_finder_rating", "_finder_cost"],
                        ascending=[False, True],
                    ).head(6)

                    if matches.empty:
                        st.info("No restaurants match these filters. Try a wider budget range or a lower minimum rating.")
                    else:
                        st.markdown(f"#### Top {len(matches)} matching restaurants")
                        st.caption(
                            "Sorted by highest rating, then lower cost. Costs are shown as rupees, "
                            "assuming the uploaded dataset uses INR."
                        )
                        for start in range(0, len(matches), 2):
                            result_columns = st.columns(2)
                            for column, (row_index, restaurant) in zip(
                                result_columns, matches.iloc[start:start + 2].iterrows()
                            ):
                                name = (
                                    str(restaurant[name_col]).strip()
                                    if name_col and pd.notna(restaurant[name_col])
                                    else f"Restaurant record {row_index}"
                                )
                                with column.container(border=True):
                                    st.markdown(f"#### {name}")
                                    metric_columns = st.columns(2)
                                    metric_columns[0].metric(
                                        "Rating",
                                        f"{restaurant['_finder_rating']:.1f} / 5",
                                    )
                                    metric_columns[1].metric(
                                        "Cost for two",
                                        f"₹{restaurant['_finder_cost']:,.0f}",
                                    )
                                    if cuisine_col and pd.notna(restaurant[cuisine_col]):
                                        st.caption(f"🍴 {restaurant[cuisine_col]}")
                                    if city_col and pd.notna(restaurant[city_col]):
                                        st.caption(f"📍 {restaurant[city_col]}")

elif page == "Food Image Recognition":
    st.subheader("Food Image Recognition")
    st.write("Upload a food photo for predictions from a model trained on the Food-101 dataset.")
    img_file = device_file_uploader("Choose a food image", ["png", "jpg", "jpeg", "webp"], "food_image")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Recognise food"):
            with st.spinner("Loading image model and analysing image..."):
                try:
                    classifier = load_food_classifier()
                    results = classifier(image, top_k=5)
                    st.markdown("#### Model predictions")
                    for r in results:
                        st.write(f"**{r['label']}** — {r['score']:.1%}")
                    st.caption("Food-101 covers a fixed set of dishes; predictions may be incorrect for dishes outside those labels.")
                except Exception as e:
                    st.error("Food recognition needs the optional Transformers/PyTorch packages and an internet connection on first run.")
                    st.code(str(e))

elif page == "Menu OCR Scanner":
    st.subheader("Smart Menu Scanner")
    st.write("Upload a menu image or scanned menu page. OCR extracts visible text; review the result because prices and words can be misread.")
    menu_file = device_file_uploader("Upload menu image", ["png", "jpg", "jpeg", "webp"], "menu_image")
    if menu_file:
        image = Image.open(menu_file).convert("RGB")
        st.image(image, caption="Uploaded menu", use_container_width=True)
        if st.button("Extract menu text"):
            with st.spinner("Running OCR..."):
                try:
                    import easyocr
                    reader = easyocr.Reader(["en"], gpu=False)
                    result = reader.readtext(np.array(image), detail=1, paragraph=False)
                    rows = []
                    for box, txt, conf in result:
                        rows.append({"Extracted text": txt, "Confidence": round(float(conf), 3)})
                    if rows:
                        out = pd.DataFrame(rows)
                        st.dataframe(out, use_container_width=True)
                        st.markdown("#### Editable menu items")
                        st.caption("OCR does not always know which number is a price. Check and correct the extracted items.")
                        st.text_area("Copy/edit extracted menu text", value="\n".join(r["Extracted text"] for r in rows), height=220)
                        st.download_button("Download OCR results CSV", out.to_csv(index=False).encode("utf-8"), "menu_ocr_results.csv", "text/csv")
                    else:
                        st.warning("No text was detected. Try a sharper, brighter image.")
                except Exception as e:
                    st.error("Menu OCR needs EasyOCR and its model files. First run may download model weights.")
                    st.code(str(e))

elif page == "Visual Image Analysis":
    st.subheader("Visual Image Analysis")
    st.write("Measure image properties. These measurements describe the image, not food safety, taste, or actual food quality.")
    file = device_file_uploader("Upload food or menu image", ["png", "jpg", "jpeg", "webp"], "visual_image")
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Image to analyse", use_container_width=True)
        arr = np.asarray(image).astype(np.float32)
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        # Variance of Laplacian is a common rough sharpness indicator.
        try:
            import cv2
            sharpness = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
            sharpness_method = "OpenCV Laplacian variance"
        except Exception:
            edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            sharpness = float(edges.var())
            sharpness_method = "Pillow edge variance (fallback)"
        r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        warmth = float((r.mean() - b.mean()))
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Brightness", f"{brightness:.1f}/255")
        c2.metric("Contrast", f"{contrast:.1f}")
        c3.metric("Sharpness indicator", f"{sharpness:.1f}")
        c4.metric("Red-blue balance", f"{warmth:.1f}")
        st.markdown("#### Interpretation")
        st.write("**Brightness:** average pixel intensity; very low values may indicate a dark image.")
        st.write("**Contrast:** variation between light and dark pixels.")
        st.write("**Sharpness indicator:** a rough measure of edge detail; compare images using the same method and similar dimensions.")
        st.write("**Red-blue balance:** a basic colour statistic, not a measure of food quality.")
        st.caption(f"Sharpness method: {sharpness_method}. These are simple visual measurements, not a trained food-quality score.")

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.write("The report's previously recorded rating-model results are shown as reference values. Train your own model on the uploaded dataset in Rating Prediction to generate fresh metrics.")
    ref = pd.DataFrame([
        {"Model": "Linear Regression", "MSE": 1.5242, "R²": 0.3344},
        {"Model": "Decision Tree Regression", "MSE": 0.1727, "R²": 0.9246},
    ])
    st.dataframe(ref, use_container_width=True)
    st.bar_chart(ref.set_index("Model")[["MSE", "R²"]])
    st.warning("These are the values reported in the presentation, not a guarantee that a newly trained model will reproduce them. Check for data leakage and validate with cross-validation before making strong claims.")
    if "rating_metrics" in st.session_state:
        st.markdown("#### Latest model trained in this session")
        st.json(st.session_state.rating_metrics)

st.divider()
st.caption("Restaurant Intelligence • Student ML project • Verify all model outputs before using them for business decisions.")
