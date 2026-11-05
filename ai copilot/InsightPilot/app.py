"""InsightPilot — Integrated AI Data Analytics Platform
Run: streamlit run app.py
"""
import io, re, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="InsightPilot | Analytics Platform", page_icon="📊", layout="wide")

# ---------- Styling ----------
st.markdown("""<style>
.block-container{padding-top:1.5rem;padding-bottom:2rem}
[data-testid="stMetric"]{background:rgba(120,120,120,.08);padding:14px;border-radius:12px;border:1px solid rgba(120,120,120,.15)}
.ip-card{padding:16px;border-radius:14px;border:1px solid rgba(120,120,120,.2);background:rgba(120,120,120,.04)}
</style>""", unsafe_allow_html=True)
st.title("📊 InsightPilot")
st.caption("Integrated AI-powered data analytics · Explore, prepare, model and generate insights")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧭 Workspace")
    section = st.radio(
        "Workspace",
        ["Data Explorer", "Data Preparation", "Machine Learning Studio", "AI Copilot", "Reports & Export"],
        label_visibility="collapsed",
        key="ip_navigation_section"
    )

    # Group related tools under a single, cleaner navigation section.
    section_pages = {
        "Data Explorer": {
            "Overview": "Dataset Explorer",
            "Dataset Intelligence": "Dataset Intelligence",
            "Dashboard & Visuals": "Advanced Dashboard",
            "Visualization & Insights": "Visualization & Insights",
        },
        "Data Preparation": {
            "Cleaning Studio": "Data Cleaning Studio",
            "AI Cleaning Advisor": "AI Cleaning Advisor",
            "Feature Engineering": "Feature Engineering Studio",
        },
        "Machine Learning Studio": {
            "Train & Compare Models": "Machine Learning Lab",
            "AutoML": "AutoML Assistant",
            "Deploy & Predict": "Model Deployment & Prediction",
        },
        "AI Copilot": {
            "Dataset Copilot (Ask about data)": "AI Knowledge Copilot",
            "Analytics Assistant (Dataset)": "AI Analytics Assistant",
        },
        "Reports & Export": {
            "Report & Export": "Report & Export",
        },
    }
    chosen_tool = st.selectbox("Tool", list(section_pages[section].keys()), key="ip_navigation_tool")
    page = section_pages[section][chosen_tool]

    st.divider()
    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json"])
    st.caption("CSV, Excel or JSON files")

@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename):
    ext = filename.rsplit('.',1)[-1].lower()
    if ext == 'csv': return pd.read_csv(io.BytesIO(file_bytes))
    if ext in ('xlsx','xls'): return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_json(io.BytesIO(file_bytes))

if uploaded:
    try:
        original = load_data(uploaded.getvalue(), uploaded.name)
        st.session_state.setdefault("ip_original", original.copy())
        st.session_state.setdefault("ip_data", original.copy())
        st.session_state["ip_filename"] = uploaded.name
    except Exception as e:
        st.error(f"Could not read this file: {e}"); st.stop()
else:
    st.info("Upload a dataset from the sidebar to begin. You can also explore InsightPilot with a sample dataset.")
    if st.button("Load sample dataset"):
        sample = pd.DataFrame({"Student": ["Asha","Ben","Cara","Dev","Eva","Finn","Gia","Hari"], "Department":["AI","CS","AI","DS","CS","DS","AI","CS"], "Score":[88,72,95,64,81,77,91,69], "Attendance":[92,78,98,65,85,80,96,73], "Hours_Studied":[12,7,15,5,10,8,14,6], "Placement":["Yes","No","Yes","No","Yes","No","Yes","No"]})
        st.session_state.ip_original=sample.copy(); st.session_state.ip_data=sample.copy(); st.session_state.ip_filename="sample_students.csv"
        st.rerun()
    st.stop()

# Use persistent cleaned dataframe
if "ip_data" not in st.session_state: st.session_state.ip_data = original.copy()
df = st.session_state.ip_data
if df.empty: st.warning("Dataset is empty."); st.stop()

# Shared helper
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if page == "Advanced Dashboard":
    st.header("📊 Advanced Analytics Dashboard")
    st.caption("Interactive charts, correlations, distributions and category comparisons.")
    ncols=df.select_dtypes(include=np.number).columns.tolist()
    ccols=df.select_dtypes(exclude=np.number).columns.tolist()
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Rows",f"{len(df):,}"); m2.metric("Columns",df.shape[1])
    m3.metric("Missing cells",int(df.isna().sum().sum())); m4.metric("Duplicate rows",int(df.duplicated().sum()))
    tab1,tab2,tab3,tab4=st.tabs(["Correlation","Distributions","Categories","Relationships"])
    with tab1:
        st.subheader("Numeric correlation matrix")
        if len(ncols)>=2:
            corr=df[ncols].corr(numeric_only=True)
            fig,ax=plt.subplots(figsize=(max(7,len(ncols)*0.7),max(5,len(ncols)*0.55)))
            im=ax.imshow(corr.values,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm")
            ax.set_xticks(range(len(ncols))); ax.set_xticklabels(ncols,rotation=45,ha="right")
            ax.set_yticks(range(len(ncols))); ax.set_yticklabels(ncols)
            fig.colorbar(im,ax=ax,label="Correlation")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.dataframe(corr.style.format("{:.2f}"),use_container_width=True)
            pairs=corr.abs().where(np.triu(np.ones(corr.shape),k=1).astype(bool)).stack().sort_values(ascending=False)
            st.write("**Strongest absolute correlations**")
            st.dataframe(pairs.head(10).rename("Absolute correlation").reset_index(),use_container_width=True)
        else: st.info("Need at least two numeric columns for correlation.")
    with tab2:
        if ncols:
            col=st.selectbox("Numeric column",ncols,key="adv_dist_col")
            fig,ax=plt.subplots(); ax.hist(df[col].dropna(),bins=30)
            ax.set_title(f"Distribution of {col}"); ax.set_xlabel(col); ax.set_ylabel("Frequency")
            st.pyplot(fig); plt.close(fig)
            st.write(df[col].describe().to_frame("Statistics"))
            fig,ax=plt.subplots(); ax.boxplot(df[col].dropna(),vert=True); ax.set_title(f"Box plot: {col}")
            st.pyplot(fig); plt.close(fig)
        else: st.info("No numeric columns available.")
    with tab3:
        if ccols:
            col=st.selectbox("Categorical column",ccols,key="adv_cat_col")
            counts=df[col].astype(str).value_counts().head(20)
            st.bar_chart(counts)
            st.dataframe(counts.rename("Count").reset_index().rename(columns={"index":"Category"}),use_container_width=True)
        else: st.info("No categorical columns available.")
    with tab4:
        if len(ncols)>=2:
            xcol=st.selectbox("X axis",ncols,key="adv_x")
            ycol=st.selectbox("Y axis",[c for c in ncols if c!=xcol],key="adv_y")
            color_col=st.selectbox("Color/group (optional)",["None"]+ccols,key="adv_color")
            plotdf=df[[xcol,ycol]+([color_col] if color_col!="None" else [])].dropna().sample(min(3000,len(df.dropna())),random_state=42)
            fig,ax=plt.subplots()
            if color_col=="None": ax.scatter(plotdf[xcol],plotdf[ycol],alpha=.65)
            else:
                for label,group in plotdf.groupby(color_col): ax.scatter(group[xcol],group[ycol],alpha=.65,label=str(label))
                ax.legend()
            ax.set_xlabel(xcol);ax.set_ylabel(ycol);ax.set_title(f"{ycol} vs {xcol}")
            st.pyplot(fig);plt.close(fig)
        else: st.info("Need at least two numeric columns for relationships.")

# ---------- AutoML Assistant ----------
if page == "AutoML Assistant":
    st.header("⚙️ AutoML Assistant")
    st.caption("Use dataset characteristics to recommend candidate algorithms, train them, and rank their results.")
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    except Exception as e:
        st.error(f"Install scikit-learn: {e}"); st.stop()
    target=st.selectbox("Target column",df.columns,key="automl_target")
    task=st.radio("Task",["Classification","Regression"],horizontal=True,key="automl_task")
    feature_candidates=[c for c in df.columns if c!=target]
    features=st.multiselect("Features",feature_candidates,default=feature_candidates,key="automl_features")
    st.info("AutoML here is a transparent baseline selector—not a guarantee of a globally optimal model. Use a separate test set and domain validation before deployment.")
    if task=="Classification":
        choices={"Logistic Regression":LogisticRegression(max_iter=1000),"Decision Tree":DecisionTreeClassifier(random_state=42),
                 "Random Forest":RandomForestClassifier(n_estimators=120,random_state=42),
                 "Extra Trees":ExtraTreesClassifier(n_estimators=120,random_state=42),
                 "Gradient Boosting":GradientBoostingClassifier(random_state=42),
                 "KNN":KNeighborsClassifier()}
        metric="F1 weighted"
    else:
        choices={"Ridge":Ridge(),"Decision Tree":DecisionTreeRegressor(random_state=42),
                 "Random Forest":RandomForestRegressor(n_estimators=120,random_state=42),
                 "Extra Trees":ExtraTreesRegressor(n_estimators=120,random_state=42),
                 "Gradient Boosting":GradientBoostingRegressor(random_state=42),
                 "KNN":KNeighborsRegressor()}
        metric="R²"
    # Heuristic recommendation from row count, feature count and target type
    n=len(df); p=max(len(features),1); unique=df[target].nunique(dropna=True)
    if task=="Classification":
        recommended=["Logistic Regression","Random Forest","Extra Trees"] if n<5000 else ["Random Forest","Extra Trees","Gradient Boosting"]
        if unique>20: st.warning("Many target classes detected. Check whether this is truly a classification task.")
        if df[target].value_counts(dropna=True).min()<5: st.warning("Some classes are rare; use stratified validation or consider collecting more examples.")
    else:
        recommended=["Ridge","Random Forest","Extra Trees"] if n<5000 else ["Random Forest","Extra Trees","Gradient Boosting"]
    st.write("**Suggested candidates based on simple dataset heuristics:**",", ".join(recommended))
    selected=st.multiselect("Algorithms to run",list(choices),default=[x for x in recommended if x in choices],key="automl_models")
    test_pct=st.slider("Test set (%)",10,40,20,key="automl_test")/100
    if st.button("🚀 Run AutoML comparison",type="primary"):
        if not features or not selected: st.error("Select features and at least one algorithm.")
        else:
            work=df[features+[target]].dropna(subset=[target]).copy(); X=work[features]; y=work[target]
            if task=="Classification" and (y.nunique()<2 or y.value_counts().min()<2):
                st.error("Classification needs at least two classes and two samples per class.")
            else:
                numeric=X.select_dtypes(include=np.number).columns.tolist(); categorical=X.select_dtypes(exclude=np.number).columns.tolist()
                trans=[]
                if numeric: trans.append(("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric))
                if categorical: trans.append(("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),categorical))
                prep=ColumnTransformer(trans)
                try: Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_pct,random_state=42,stratify=y if task=="Classification" else None)
                except Exception as e: st.error(f"Split failed: {e}"); st.stop()
                rows=[]; trained={}
                bar=st.progress(0)
                for i,name in enumerate(selected):
                    try:
                        pipe=Pipeline([("preprocess",prep),("model",choices[name])]);pipe.fit(Xtr,ytr);pred=pipe.predict(Xte)
                        if task=="Classification": score=f1_score(yte,pred,average="weighted",zero_division=0);rows.append({"Algorithm":name,"Accuracy":accuracy_score(yte,pred),"F1 weighted":score})
                        else: score=r2_score(yte,pred);rows.append({"Algorithm":name,"MAE":mean_absolute_error(yte,pred),"RMSE":float(np.sqrt(mean_squared_error(yte,pred))),"R²":score})
                        trained[name]=pipe
                    except Exception as e: st.warning(f"{name} failed: {e}")
                    bar.progress((i+1)/len(selected))
                if rows:
                    res=pd.DataFrame(rows).sort_values(metric,ascending=False);st.subheader("AutoML leaderboard");st.dataframe(res,use_container_width=True)
                    winner=res.iloc[0]["Algorithm"];st.success(f"Recommended best candidate: {winner} · {metric}={res.iloc[0][metric]:.4f}")
                    st.session_state.ip_model=trained[winner];st.session_state.ip_model_name=winner
                    st.download_button("Download leaderboard CSV",res.to_csv(index=False).encode(),"automl_leaderboard.csv","text/csv")
                    st.caption("Ranking reflects this one train/test split. Confirm with cross-validation, leakage checks and a business-relevant metric.")

# ---------- AI Cleaning Advisor ----------
if page == "AI Cleaning Advisor":
    st.header("🧠 AI Cleaning Advisor")
    st.caption("Explain data-quality issues and produce a cleaning plan. Review every change before applying it.")
    issues=[]
    for col in df.columns:
        missing=int(df[col].isna().sum())
        if missing: issues.append({"Column":str(col),"Issue":"Missing values","Count":missing,"Suggested action":"Impute or review missing records"})
        if df[col].nunique(dropna=True)<=1: issues.append({"Column":str(col),"Issue":"Constant column","Count":int(df[col].nunique(dropna=True)),"Suggested action":"Consider dropping if not meaningful"})
        if df[col].dtype=="object":
            spaces=int(df[col].astype("string").str.strip().ne(df[col].astype("string")).fillna(False).sum())
            if spaces: issues.append({"Column":str(col),"Issue":"Leading/trailing spaces","Count":spaces,"Suggested action":"Trim text"})
    duplicates=int(df.duplicated().sum())
    if duplicates: issues.append({"Column":"—","Issue":"Duplicate rows","Count":duplicates,"Suggested action":"Review/remove duplicates"})
    st.subheader("Detected issues")
    if issues: st.dataframe(pd.DataFrame(issues),use_container_width=True)
    else: st.success("No basic issues detected.")
    st.subheader("Cleaning plan")
    plan=[]
    for col in df.columns:
        n=int(df[col].isna().sum())
        if n:
            method="median" if pd.api.types.is_numeric_dtype(df[col]) else "most frequent"
            plan.append(f"Fill {n} missing values in **{col}** using {method}.")
        if df[col].dtype=="object" and int(df[col].astype("string").str.strip().ne(df[col].astype("string")).fillna(False).sum()):
            plan.append(f"Trim whitespace in **{col}**.")
        if df[col].nunique(dropna=True)<=1: plan.append(f"Consider dropping constant column **{col}**.")
    if duplicates: plan.append(f"Consider removing {duplicates} duplicate rows.")
    if plan:
        for item in plan: st.markdown("- "+item)
    else: st.info("No recommended basic cleaning actions.")
    with st.expander("Ask local Ollama to explain the cleaning plan"):
        ollama_url=st.text_input("Ollama endpoint","http://localhost:11434/api/generate",key="clean_ollama_url")
        ollama_model=st.text_input("Model","llama3.2",key="clean_ollama_model")
        if st.button("Explain issues with Ollama"):
            try:
                import requests
                profile={"shape":df.shape,"missing":df.isna().sum().to_dict(),"duplicates":duplicates,"issues":issues,"plan":plan}
                prompt="Explain this data quality profile in beginner-friendly language. Explain risks and safe cleaning choices; do not claim changes were applied. Profile: "+json.dumps(profile,default=str)
                resp=requests.post(ollama_url,json={"model":ollama_model,"prompt":prompt,"stream":False},timeout=120)
                resp.raise_for_status(); st.markdown(resp.json().get("response","No response"))
            except Exception as e: st.error(f"Ollama request failed: {e}")



# ---------- AI Dataset Copilot ----------
if page == "AI Knowledge Copilot":
    st.header("📊 AI Dataset Copilot")
    st.caption("Ask questions about the dataset currently uploaded in InsightPilot. Answers use your dataset profile, statistics, and sample records.")

    st.success(f"Active dataset: {st.session_state.get('ip_filename', 'uploaded dataset')} · {len(df):,} rows × {df.shape[1]} columns")

    # Build a compact, privacy-conscious context from the active dataframe.
    profile = pd.DataFrame({
        "column": df.columns.astype(str),
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    numeric_summary = df.select_dtypes(include=np.number).describe().round(3).to_dict()
    sample = df.head(15).replace({np.nan: None}).to_dict(orient="records")

    with st.expander("Dataset context sent to the local AI", expanded=False):
        st.write("Column profile")
        st.dataframe(profile, use_container_width=True)
        st.write("First 15 records")
        st.dataframe(df.head(15), use_container_width=True)
        if numeric_summary:
            st.write("Numeric summary")
            st.json(numeric_summary)

    question = st.text_area(
        "Ask your dataset assistant",
        placeholder="Examples: Explain this dataset in simple terms; which columns have missing values?; What patterns can you see?",
        height=100,
        key="ip_dataset_copilot_question",
    )
    ollama_model = st.text_input("Ollama model", value="llama3.2", key="ip_dataset_copilot_model")
    ollama_url = st.text_input("Ollama endpoint", value="http://localhost:11434/api/generate", key="ip_dataset_copilot_url")

    if st.button("🤖 Ask Dataset Assistant", type="primary", key="ip_dataset_copilot_ask") and question.strip():
        try:
            import requests
            dataset_context = {
                "filename": st.session_state.get("ip_filename", "dataset"),
                "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
                "columns": profile.to_dict(orient="records"),
                "numeric_summary": numeric_summary,
                "sample_records_first_15": sample,
                "missing_by_column": df.isna().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
            }
            prompt = (
                "You are InsightPilot's dataset assistant. Answer the user's question in clear, beginner-friendly language. "
                "Use ONLY the dataset context supplied below. Explain what the available statistics and sample can support. "
                "Do not invent exact values, claim to have inspected every row when only a sample/statistics are supplied, "
                "or infer column meanings with certainty. If the context is insufficient, say so and suggest a relevant "
                "InsightPilot analysis (such as Dataset Explorer, Visualization, Analytics Assistant, or Machine Learning Lab). "
                "\nDATASET CONTEXT:\n" + json.dumps(dataset_context, default=str) +
                "\n\nUSER QUESTION:\n" + question.strip()
            )
            with st.spinner("Ollama is analysing your dataset context..."):
                response = requests.post(
                    ollama_url,
                    json={"model": ollama_model, "prompt": prompt, "stream": False},
                    timeout=180,
                )
                response.raise_for_status()
                answer = response.json().get("response", "No response returned.")
            st.session_state["ip_dataset_copilot_answer"] = answer
            st.session_state["ip_dataset_copilot_answer_question"] = question.strip()
        except Exception as e:
            st.error(f"Ollama request failed. Check that Ollama is running and the model is installed. Details: {e}")

    if st.session_state.get("ip_dataset_copilot_answer_question") == question.strip():
        st.subheader("Assistant answer")
        st.markdown(st.session_state.get("ip_dataset_copilot_answer", ""))

    st.info(
        "This assistant is dataset-aware: it uses column metadata, descriptive statistics, missing-value counts, "
        "duplicate counts, and the first 15 records. For exact calculations across all rows, use AI Analytics Assistant "
        "or the relevant analysis module. The LLM response is an explanation, not a substitute for computed results."
    )


# ---------- Model Deployment & Prediction Playground ----------
if page == "Model Deployment & Prediction":
    st.header("🚀 Model Deployment & Prediction")
    st.caption("Export a trained pipeline and use it to predict outcomes for new records.")
    trained = st.session_state.get("ip_model")
    model_name = st.session_state.get("ip_model_name", "Trained model")
    if trained is None:
        st.warning("No trained model is available yet. Train a model in Machine Learning Lab first.")
        st.stop()
    st.success(f"Ready: {model_name}")
    st.write("This exports the complete preprocessing + model pipeline, so incoming data is transformed consistently.")
    try:
        import joblib
        model_buffer = io.BytesIO()
        joblib.dump(trained, model_buffer)
        model_buffer.seek(0)
        st.download_button("⬇️ Download trained pipeline (.joblib)", model_buffer,
                           file_name="insightpilot_model.joblib",
                           mime="application/octet-stream")
    except Exception as e:
        st.error(f"Model export unavailable: {e}")

    st.divider()
    st.subheader("Predict on a new dataset")
    incoming = st.file_uploader("Upload prediction data (CSV or Excel)", type=["csv", "xlsx", "xls"], key="ip_prediction_upload")
    if incoming is not None:
        try:
            new_df = pd.read_csv(incoming) if incoming.name.lower().endswith(".csv") else pd.read_excel(incoming)
            st.write("Incoming data preview")
            st.dataframe(new_df.head(10), use_container_width=True)
            try:
                required = list(trained.feature_names_in_)
            except Exception:
                required = None
            if required:
                missing = [c for c in required if c not in new_df.columns]
                extra = [c for c in new_df.columns if c not in required]
                if missing:
                    st.error("Missing required feature columns: " + ", ".join(map(str, missing)))
                else:
                    if extra:
                        st.caption("Extra columns will be ignored: " + ", ".join(map(str, extra)))
                    Xnew = new_df[required]
                    if st.button("🔮 Generate predictions", type="primary"):
                        pred = trained.predict(Xnew)
                        result = new_df.copy()
                        result["Prediction"] = pred
                        if hasattr(trained, "predict_proba"):
                            try:
                                probs = trained.predict_proba(Xnew)
                                if probs.ndim == 2 and probs.shape[1] == 2:
                                    result["Probability (positive class)"] = probs[:, 1]
                                elif probs.ndim == 2:
                                    result["Prediction confidence"] = probs.max(axis=1)
                            except Exception:
                                pass
                        st.subheader("Prediction results")
                        st.dataframe(result, use_container_width=True)
                        st.download_button("⬇️ Download predictions (CSV)", result.to_csv(index=False).encode("utf-8"),
                                           "insightpilot_predictions.csv", "text/csv")
            else:
                st.info("This model does not expose its original feature names. Ensure the uploaded columns match the training features.")
                if st.button("🔮 Predict using uploaded columns", type="primary"):
                    pred = trained.predict(new_df)
                    result = new_df.copy()
                    result["Prediction"] = pred
                    st.dataframe(result, use_container_width=True)
                    st.download_button("⬇️ Download predictions (CSV)", result.to_csv(index=False).encode("utf-8"),
                                      "insightpilot_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read or predict from this file: {e}")


# ---------- Feature Engineering Studio ----------
if page == "Feature Engineering Studio":
    st.header("🧩 Feature Engineering Studio")
    st.caption("Prepare model features with encoding, scaling, and optional feature selection.")
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    categorical=df.select_dtypes(exclude=np.number).columns.tolist()
    ftab1,ftab2,ftab3=st.tabs(["Encoding","Scaling","Feature selection"])
    with ftab1:
        st.write("Categorical values can be converted into numeric columns for machine learning.")
        enc_cols=st.multiselect("Categorical columns to one-hot encode",categorical,default=categorical,key="fe_enc_cols")
        if st.button("Preview one-hot encoding"):
            preview=pd.get_dummies(df,columns=enc_cols,dummy_na=False)
            st.write("Preview shape:",preview.shape);st.dataframe(preview.head(30),use_container_width=True)
            st.download_button("Download encoded dataset",preview.to_csv(index=False).encode(),"insightpilot_encoded.csv","text/csv")
    with ftab2:
        st.write("Scaling is useful for distance-based and gradient-based models (e.g. KNN, SVM, Logistic Regression).")
        scale_cols=st.multiselect("Numeric columns to scale",numeric,default=numeric,key="fe_scale_cols")
        scaler_choice=st.selectbox("Scaler",["StandardScaler","MinMaxScaler","RobustScaler"],key="fe_scaler")
        if st.button("Preview scaling"):
            from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
            scaler={"StandardScaler":StandardScaler(),"MinMaxScaler":MinMaxScaler(),"RobustScaler":RobustScaler()}[scaler_choice]
            preview=df.copy()
            if scale_cols:
                valid=preview[scale_cols].replace([np.inf,-np.inf],np.nan)
                preview[scale_cols]=scaler.fit_transform(valid.fillna(valid.median()))
            st.dataframe(preview.head(30),use_container_width=True)
            st.download_button("Download scaled dataset",preview.to_csv(index=False).encode(),"insightpilot_scaled.csv","text/csv")
    with ftab3:
        st.write("Select features using a supervised score. Choose a target and keep only the highest scoring predictors.")
        target=st.selectbox("Target",df.columns,key="fe_target")
        k=st.slider("Number of features to keep",1,max(1,len([c for c in df.columns if c!=target])),min(5,max(1,len(df.columns)-1)),key="fe_k")
        if st.button("Run feature selection"):
            try:
                from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.impute import SimpleImputer
                from sklearn.pipeline import Pipeline
                work=df.dropna(subset=[target]).copy(); X=work.drop(columns=[target]); y=work[target]
                cats=X.select_dtypes(exclude=np.number).columns.tolist(); nums=X.select_dtypes(include=np.number).columns.tolist()
                trans=[]
                if nums: trans.append(("num",SimpleImputer(strategy="median"),nums))
                if cats: trans.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cats))
                prep=ColumnTransformer(trans); Xp=prep.fit_transform(X)
                score_func=f_classif if y.dtype=="object" or y.nunique()<=20 else f_regression
                selector=SelectKBest(score_func=score_func,k=min(k,Xp.shape[1]));selector.fit(Xp,y)
                names=prep.get_feature_names_out(); selected_names=names[selector.get_support()]
                scores=pd.DataFrame({"Feature":names,"Score":selector.scores_}).sort_values("Score",ascending=False)
                st.write("Selected features");st.dataframe(scores.head(k),use_container_width=True)
                st.session_state["ip_selected_features"]=selected_names.tolist()
                st.success(f"Selected {len(selected_names)} encoded feature(s).")
            except Exception as e: st.error(f"Feature selection failed: {e}")

# ---------- Dataset Explorer ----------
if page == "Dataset Explorer":
    st.header("🔎 Dataset Explorer")
    a,b,c,d = st.columns(4)
    a.metric("Rows", f"{len(df):,}"); b.metric("Columns", f"{df.shape[1]:,}"); c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}"); d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)
    with st.expander("Dataset profile", expanded=True):
        x,y=st.columns(2)
        with x: st.write("**Column types**"); st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
        with y: st.write("**Missing values by column**"); st.dataframe(df.isna().sum().rename("missing"), use_container_width=True)
    if num_cols:
        st.subheader("Numerical summary"); st.dataframe(df[num_cols].describe().T, use_container_width=True)
    if cat_cols:
        st.subheader("Categorical summary"); st.dataframe(df[cat_cols].describe().T, use_container_width=True)


# ---------- Dataset Intelligence ----------
elif page == "Dataset Intelligence":
    st.header("🧠 Dataset Intelligence")
    st.caption("Automatic dataset profiling, data-quality checks, target suggestions and ML recommendations.")

    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    missing_pct = (missing_total / max(rows * cols, 1)) * 100
    quality_score = max(0, 100 - missing_pct - (duplicate_total / max(rows, 1) * 100))
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{rows:,}")
    b.metric("Columns", f"{cols:,}")
    c.metric("Missing cells", f"{missing_total:,}", f"{missing_pct:.1f}%")
    d.metric("Duplicate rows", f"{duplicate_total:,}")
    st.metric("Indicative data completeness score", f"{100-missing_pct:.1f}%")
    st.caption("Completeness measures non-missing cells only; it is not a full data-quality certification.")

    st.subheader("1 · Dataset overview")
    st.write(f"Your dataset contains **{rows:,} records** and **{cols} columns**.")
    st.write(f"It has **{len(num_cols)} numerical columns** and **{len(cat_cols)} categorical/non-numerical columns**.")
    if missing_total == 0 and duplicate_total == 0:
        st.success("No missing cells or exact duplicate rows were detected.")
    else:
        st.warning(f"Detected {missing_total:,} missing cells and {duplicate_total:,} exact duplicate rows. Cleaning may be required.")

    st.subheader("2 · Column-by-column profile")
    profile = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Role (heuristic)": ["Numeric feature" if c in num_cols else "Categorical/text feature" for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(float(df[c].isna().mean()*100), 2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "Constant?": [df[c].nunique(dropna=True) <= 1 for c in df.columns],
    })
    st.dataframe(profile, use_container_width=True)
    with st.expander("Column meaning (data dictionary)", expanded=False):
        st.info("Column meanings cannot be reliably inferred from names alone. Add domain descriptions here or use the AI explanation below for suggested interpretations.")
        dictionary = pd.DataFrame({"Column":df.columns, "Your description":[""]*len(df.columns)})
        st.data_editor(dictionary, hide_index=True, use_container_width=True, key="ip_dictionary")

    st.subheader("3 · Data-quality checks")
    missing_table = pd.DataFrame({"Missing count":df.isna().sum(), "Missing %":(df.isna().mean()*100).round(2)})
    missing_table = missing_table[missing_table["Missing count"] > 0].sort_values("Missing %",ascending=False)
    if missing_table.empty:
        st.success("Missing-value check: none found.")
    else:
        st.warning("Missing values found. Suggested treatment depends on the column and the meaning of missingness.")
        st.dataframe(missing_table, use_container_width=True)
        st.write("**Typical options:** numeric columns → median (robust) or mean; categorical columns → most-frequent value or 'Unknown'; drop rows only when safe and justified.")
    if duplicate_total:
        st.warning(f"Duplicate check: {duplicate_total} exact duplicate rows. Review before removing; duplicates can sometimes be valid repeated observations.")
    else:
        st.success("Duplicate check: no exact duplicate rows.")
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        st.info("Constant/near-constant candidates: " + ", ".join(map(str,constant_cols)) + ". These usually add little predictive information.")
    else:
        st.success("No constant columns detected.")

    outlier_rows = []
    for c in num_cols:
        series = df[c].dropna()
        if len(series) >= 4:
            q1, q3 = series.quantile(.25), series.quantile(.75)
            iqr = q3-q1
            count = int(((series < q1-1.5*iqr) | (series > q3+1.5*iqr)).sum()) if iqr > 0 else 0
            outlier_rows.append({"Column":c,"IQR outlier count":count,"IQR outlier %":round(count/max(len(series),1)*100,2)})
    if outlier_rows:
        out_df = pd.DataFrame(outlier_rows).sort_values("IQR outlier count",ascending=False)
        st.write("**Potential numerical outliers (IQR rule)**")
        st.dataframe(out_df, use_container_width=True)
        st.caption("Outliers are candidates for investigation, not automatically errors. Do not remove them without domain reasoning.")

    st.subheader("4 · Target-column suggestions")
    st.write("A target is the outcome you want your model to predict. It is chosen from your business/research question; this page only proposes candidates.")
    target_candidates=[]
    for c in df.columns:
        nunique=int(df[c].nunique(dropna=True)); dtype=df[c].dtype
        if c in num_cols:
            task_hint="Regression candidate" if nunique > 10 else "Could be classification if values are labels"
        else:
            task_hint="Classification candidate" if 2 <= nunique <= 30 else "Possible ID/text or high-cardinality field"
        target_candidates.append({"Column":c,"Unique values":nunique,"Type":str(dtype),"Heuristic suggestion":task_hint})
    st.dataframe(pd.DataFrame(target_candidates),use_container_width=True)
    st.warning("Avoid IDs, names, post-outcome fields, and columns that reveal the answer (data leakage) as targets/features unless they are genuinely part of the prediction problem.")

    st.subheader("5 · Which analytical / ML method fits?")
    goal = st.selectbox("What do you want to achieve?",[
        "Understand patterns without a target",
        "Predict a category / label",
        "Predict a numeric value",
        "Find unusual observations",
        "Summarise or segment groups"
    ])
    if goal == "Understand patterns without a target":
        st.success("Suggested: Exploratory Data Analysis (EDA), correlation analysis, PCA (after scaling), and clustering such as K-Means if grouping is meaningful.")
    elif goal == "Predict a category / label":
        st.success("Suggested: Classification. Start with Logistic Regression as a baseline; compare Decision Tree, Random Forest, Extra Trees, and (where suitable) SVM. Check class balance and use F1/precision/recall as well as accuracy.")
    elif goal == "Predict a numeric value":
        st.success("Suggested: Regression. Start with Linear/Ridge Regression as a baseline; compare Decision Tree, Random Forest, Gradient Boosting and SVR. Evaluate MAE, RMSE and R².")
    elif goal == "Find unusual observations":
        st.success("Suggested: IQR/z-score for simple numeric screening; Isolation Forest for multivariate anomaly detection. Investigate flagged records before treating them as errors.")
    else:
        st.success("Suggested: group-by summaries for known categories; K-Means or hierarchical clustering for exploratory segmentation after encoding/scaling and checking feature suitability.")
    st.caption("These are starting recommendations, not a claim that one model is universally best. Use a held-out test set or cross-validation to compare models.")

    st.subheader("6 · Cleaning readiness checklist")
    checklist = [
        ("Missing values", missing_total == 0, "Review imputation/drop strategy"),
        ("Exact duplicate rows", duplicate_total == 0, "Review and remove only if duplicates are invalid"),
        ("Constant columns", len(constant_cols) == 0, "Consider excluding constant columns"),
        ("Target defined", False, "Choose a target based on your prediction objective"),
        ("Leakage reviewed", False, "Check features for information unavailable at prediction time"),
    ]
    for label, ok, action in checklist:
        st.write(("✅ " if ok else "⚠️ ") + f"**{label}:** " + ("Passed basic check" if ok else action))
    st.info("This checklist is an initial automated assessment. Dataset readiness also depends on domain context, correct labels, representative samples, and train/test design.")

    st.subheader("7 · AI explanation (optional, Ollama)")
    st.write("Generate a natural-language explanation using the local Ollama service. Computed profile details are sent to the model—not the full dataset.")
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
    ollama_model = st.text_input("Model", "llama3.2")
    if st.button("Explain my dataset with Ollama"):
        try:
            import requests
            payload = {
                "dataset_shape":{"rows":rows,"columns":cols},
                "column_profile":profile.to_dict(orient="records"),
                "missing_by_column":df.isna().sum().to_dict(),
                "duplicate_rows":duplicate_total,
                "numeric_summary":df[num_cols].describe().to_dict() if num_cols else {},
                "user_goal":goal
            }
            prompt = ("You are a careful data analyst. Explain this dataset profile in plain English. "
                      "Cover dataset structure, quality/missing values, likely target candidates (clearly label as hypotheses), "
                      "suitable methods for the stated goal, cleaning recommendations, limitations, and next steps. "
                      "Do not claim column meanings or a best model without evidence. Profile JSON:\\n"+json.dumps(payload,default=str))
            response=requests.post(ollama_url.rstrip("/")+"/api/generate",
                json={"model":ollama_model,"prompt":prompt,"stream":False},timeout=180)
            response.raise_for_status()
            st.markdown(response.json().get("response","No response returned."))
        except Exception as e:
            st.error(f"Ollama request failed: {e}. Confirm Ollama is running and the model is installed.")

# ---------- Cleaning ----------
elif page == "Data Cleaning Studio":
    st.header("🧹 Data Cleaning Studio")
    st.caption("Review data-quality recommendations, preview changes, and apply cleaning safely.")
    missing_by_col = df.isna().sum()
    missing_cols = missing_by_col[missing_by_col > 0].index.tolist()
    duplicate_count = int(df.duplicated().sum())
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    st.subheader("Automatic cleaning recommendations")
    recommendations = []
    if missing_cols:
        for c in missing_cols:
            pct = float(df[c].isna().mean()*100)
            if c in num_cols:
                recommendations.append(f"{c}: {int(missing_by_col[c])} missing ({pct:.1f}%). Consider median imputation; mean if distribution is symmetric.")
            else:
                recommendations.append(f"{c}: {int(missing_by_col[c])} missing ({pct:.1f}%). Consider most-frequent value or 'Unknown'.")
    else:
        recommendations.append("No missing values detected.")
    if duplicate_count:
        recommendations.append(f"{duplicate_count} exact duplicate rows detected. Remove only if they are accidental.")
    else:
        recommendations.append("No exact duplicate rows detected.")
    if constant_cols:
        recommendations.append("Constant columns may be removable: " + ", ".join(map(str,constant_cols)))
    if not missing_cols and not duplicate_count and not constant_cols:
        st.success("No basic cleaning actions are suggested by these checks.")
    else:
        for rec in recommendations: st.write("• " + rec)

    st.subheader("One-click cleaning")
    with st.form("auto_clean_form"):
        fill_numeric = st.selectbox("Fill numeric missing values", ["Do not fill", "Median", "Mean"])
        fill_categorical = st.selectbox("Fill categorical missing values", ["Do not fill", "Most frequent", "Unknown"])
        remove_duplicates = st.checkbox("Remove exact duplicate rows", value=False)
        remove_constants = st.checkbox("Remove constant columns", value=False)
        trim_text = st.checkbox("Trim leading/trailing whitespace in text", value=True)
        submitted = st.form_submit_button("Preview & apply selected cleaning", type="primary")
    if submitted:
        cleaned=df.copy()
        if fill_numeric != "Do not fill":
            strategy = "median" if fill_numeric=="Median" else "mean"
            for c in cleaned.select_dtypes(include=np.number).columns:
                if cleaned[c].isna().any():
                    value = cleaned[c].median() if strategy=="median" else cleaned[c].mean()
                    cleaned[c]=cleaned[c].fillna(value)
        if fill_categorical != "Do not fill":
            for c in cleaned.select_dtypes(exclude=np.number).columns:
                if cleaned[c].isna().any():
                    value = "Unknown" if fill_categorical=="Unknown" else (cleaned[c].mode(dropna=True).iloc[0] if not cleaned[c].mode(dropna=True).empty else "Unknown")
                    cleaned[c]=cleaned[c].fillna(value)
        if remove_duplicates: cleaned=cleaned.drop_duplicates()
        if remove_constants:
            const=[c for c in cleaned.columns if cleaned[c].nunique(dropna=True)<=1]
            cleaned=cleaned.drop(columns=const)
        if trim_text:
            for c in cleaned.select_dtypes(include="object").columns:
                cleaned[c]=cleaned[c].map(lambda v:v.strip() if isinstance(v,str) else v)
        st.session_state.ip_cleaning_preview=cleaned
        st.session_state.ip_cleaning_changes={"before":df.shape,"after":cleaned.shape,
            "missing_before":int(df.isna().sum().sum()),"missing_after":int(cleaned.isna().sum().sum()),
            "duplicates_before":int(df.duplicated().sum()),"duplicates_after":int(cleaned.duplicated().sum())}

    if "ip_cleaning_preview" in st.session_state:
        preview=st.session_state.ip_cleaning_preview
        changes=st.session_state.ip_cleaning_changes
        st.subheader("Cleaning preview")
        a,b,c=st.columns(3)
        a.metric("Rows",f"{changes['before'][0]} → {changes['after'][0]}")
        b.metric("Columns",f"{changes['before'][1]} → {changes['after'][1]}")
        c.metric("Missing cells",f"{changes['missing_before']} → {changes['missing_after']}")
        st.dataframe(preview.head(50),use_container_width=True)
        x,y=st.columns(2)
        with x:
            if st.button("Confirm and replace active dataset",type="primary"):
                st.session_state.ip_data=preview.copy()
                st.session_state.pop("ip_cleaning_preview",None)
                st.session_state.pop("ip_cleaning_changes",None)
                st.success("Cleaned dataset applied."); st.rerun()
        with y:
            if st.button("Discard cleaning preview"):
                st.session_state.pop("ip_cleaning_preview",None)
                st.session_state.pop("ip_cleaning_changes",None)
                st.rerun()
    if st.button("Reset to original upload"):
        st.session_state.ip_data=st.session_state.ip_original.copy()
        st.session_state.pop("ip_cleaning_preview",None)
        st.session_state.pop("ip_cleaning_changes",None)
        st.rerun()
    st.caption("Always review cleaning decisions: imputation and duplicate removal can change analytical meaning.")

# ---------- Visualization ----------
elif page == "Visualization & Insights":
    st.header("📈 Visualization & Insights")
    chart_type=st.selectbox("Chart type", ["Histogram","Bar chart","Scatter plot","Box plot","Correlation heatmap"])
    if chart_type=="Histogram":
        if not num_cols: st.warning("No numeric columns available.")
        else:
            col=st.selectbox("Numeric column",num_cols); bins=st.slider("Bins",5,60,20)
            fig,ax=plt.subplots(); ax.hist(df[col].dropna(),bins=bins); ax.set_title(f"Distribution of {col}"); ax.set_xlabel(col); ax.set_ylabel("Frequency"); st.pyplot(fig); plt.close(fig)
    elif chart_type=="Bar chart":
        if not cat_cols: st.warning("No categorical columns available.")
        else:
            col=st.selectbox("Category",cat_cols); top=st.slider("Categories to show",3,30,10)
            counts=df[col].astype(str).value_counts().head(top); fig,ax=plt.subplots(); counts.plot(kind="bar",ax=ax); ax.set_ylabel("Count"); st.pyplot(fig); plt.close(fig)
    elif chart_type=="Scatter plot":
        if len(num_cols)<2: st.warning("Need at least two numeric columns.")
        else:
            x=st.selectbox("X axis",num_cols); y=st.selectbox("Y axis",[z for z in num_cols if z!=x]); fig,ax=plt.subplots(); ax.scatter(df[x],df[y],alpha=.7); ax.set_xlabel(x); ax.set_ylabel(y); st.pyplot(fig); plt.close(fig)
    elif chart_type=="Box plot":
        if not num_cols: st.warning("No numeric columns available.")
        else:
            col=st.selectbox("Numeric column",num_cols); fig,ax=plt.subplots(); ax.boxplot(df[col].dropna()); ax.set_ylabel(col); st.pyplot(fig); plt.close(fig)
    else:
        if len(num_cols)<2: st.warning("Need at least two numeric columns.")
        else:
            fig,ax=plt.subplots(figsize=(9,6)); corr=df[num_cols].corr(); im=ax.imshow(corr,vmin=-1,vmax=1); ax.set_xticks(range(len(num_cols)),num_cols,rotation=45,ha="right"); ax.set_yticks(range(len(num_cols)),num_cols); fig.colorbar(im); st.pyplot(fig); plt.close(fig); st.dataframe(corr, use_container_width=True)
    st.subheader("Quick insights")
    if num_cols:
        insight=pd.DataFrame({"Column":num_cols,"Mean":df[num_cols].mean().values,"Median":df[num_cols].median().values,"Std":df[num_cols].std().values}).sort_values("Mean",ascending=False)
        st.dataframe(insight, use_container_width=True)

# ---------- Copilot ----------
elif page == "AI Analytics Assistant":
    st.header("🤖 AI Analytics Assistant")
    st.caption("Rule-based Pandas assistant: computes answers directly from your data (no external LLM required).")
    question=st.text_input("Ask about your dataset", placeholder="e.g. average Score, missing values, top Department")
    if st.button("Analyze question", type="primary") and question:
        q=question.lower(); answer=None
        if "missing" in q or "null" in q or "na " in q: answer=df.isna().sum().sort_values(ascending=False).to_frame("Missing values")
        elif "duplicate" in q: answer=f"There are {int(df.duplicated().sum())} duplicate rows."
        elif "correlation" in q or "correlat" in q:
            if len(num_cols)>1: answer=df[num_cols].corr()
            else: answer="At least two numeric columns are needed for correlation."
        elif "count" in q or "frequency" in q or "top " in q:
            match=next((c for c in df.columns if c.lower() in q),None)
            if match and match in cat_cols: answer=df[match].value_counts().head(15).to_frame("Count")
        else:
            match=next((c for c in df.columns if c.lower() in q),None)
            if match and match in num_cols:
                if "median" in q: answer=f"Median {match}: {df[match].median():.4g}"
                elif "max" in q or "highest" in q: answer=f"Maximum {match}: {df[match].max():.4g}"
                elif "min" in q or "lowest" in q: answer=f"Minimum {match}: {df[match].min():.4g}"
                elif "sum" in q or "total" in q: answer=f"Total {match}: {df[match].sum():.4g}"
                else: answer=f"Mean {match}: {df[match].mean():.4g}"
        if answer is None: st.info("Try mentioning a column name with mean, median, max, min, sum, count, or ask about missing values, duplicates, or correlation.")
        elif isinstance(answer,(pd.DataFrame,pd.Series)): st.dataframe(answer, use_container_width=True)
        else: st.success(str(answer))
    st.markdown("**Suggested questions**")
    st.write("• Which columns have missing values?  • Show duplicate rows count  • What is the mean of [numeric column]?  • Show correlation")

# ---------- ML ----------
elif page == "Machine Learning Lab":
    st.header("🧪 Machine Learning Lab")
    st.caption("Train and compare multiple classification or regression algorithms.")
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.naive_bayes import GaussianNB
    except Exception as e:
        st.error(f"Install scikit-learn to use this module: {e}")
        st.stop()

    target = st.selectbox("Target column", df.columns.tolist())
    task = st.radio("Problem type", ["Classification", "Regression"], horizontal=True)
    features = st.multiselect("Feature columns", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
    if task == "Classification":
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(), "Support Vector Machine": SVC(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42), "Naive Bayes": GaussianNB()
        }
    else:
        model_options = {
            "Linear Regression": LinearRegression(), "Ridge Regression": Ridge(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(), "Support Vector Regression": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42)
        }
    selected_models = st.multiselect("Models to compare", list(model_options), default=["Random Forest"])
    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    if st.button("Train & Compare Models", type="primary"):
        if not features: st.error("Select at least one feature column.")
        elif not selected_models: st.error("Select at least one model.")
        else:
            work = df[features + [target]].dropna(subset=[target]).copy()
            X, y = work[features], work[target]
            if task == "Classification" and y.nunique() < 2: st.error("Classification target must contain at least two classes.")
            elif task == "Classification" and y.value_counts().min() < 2: st.error("Each class needs at least two records for train/test split.")
            else:
                cats = X.select_dtypes(exclude=np.number).columns.tolist(); nums = X.select_dtypes(include=np.number).columns.tolist()
                transformers = []
                if nums: transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums))
                if cats: transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats))
                prep = ColumnTransformer(transformers)
                try:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task == "Classification" else None)
                except Exception as e: st.error(f"Could not split dataset: {e}"); st.stop()
                results, fitted = [], {}
                progress = st.progress(0)
                for i, name in enumerate(selected_models):
                    try:
                        pipe = Pipeline([("prep", prep), ("model", model_options[name])]); pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
                        if task == "Classification":
                            score = accuracy_score(yte, pred); results.append({"Model": name, "Accuracy": score, "F1 (weighted)": f1_score(yte, pred, average="weighted")})
                        else:
                            score = r2_score(yte, pred); results.append({"Model": name, "MAE": mean_absolute_error(yte, pred), "RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "R²": score})
                        fitted[name] = (pipe, score)
                    except Exception as e: st.warning(f"{name} failed: {e}")
                    progress.progress((i + 1) / len(selected_models))
                if results:
                    result_df = pd.DataFrame(results).sort_values("Accuracy" if task == "Classification" else "R²", ascending=False)
                    st.subheader("Model comparison"); st.dataframe(result_df, use_container_width=True)
                    best_name = result_df.iloc[0]["Model"]; best_pipe, best_score = fitted[best_name]
                    st.success(f"Best model: {best_name} · {'Accuracy' if task == 'Classification' else 'R²'} = {best_score:.3f}")
                    st.session_state.ip_model = best_pipe; st.session_state.ip_model_name = best_name
                    st.download_button("Download comparison (CSV)", result_df.to_csv(index=False).encode("utf-8"), "insightpilot_model_comparison.csv", "text/csv")
                    st.divider()
                    st.subheader("🔬 Model Evaluation & Explainability")
                    eval_tab, explain_tab = st.tabs(["Evaluation", "Feature importance"])
                    with eval_tab:
                        final_pred = best_pipe.predict(Xte)
                        if task == "Classification":
                            from sklearn.metrics import confusion_matrix, classification_report
                            labels = list(pd.Series(yte).unique())
                            cm = confusion_matrix(yte, final_pred, labels=labels)
                            st.write("Confusion matrix (rows = actual, columns = predicted)")
                            st.dataframe(pd.DataFrame(cm, index=[f"Actual: {x}" for x in labels],
                                                      columns=[f"Pred: {x}" for x in labels]), use_container_width=True)
                            st.text("Classification report")
                            st.code(classification_report(yte, final_pred, zero_division=0))
                        else:
                            eval_df = pd.DataFrame({"Actual": yte, "Predicted": final_pred})
                            st.write("Actual vs predicted values")
                            st.dataframe(eval_df.head(100), use_container_width=True)
                            fig, ax = plt.subplots()
                            ax.scatter(yte, final_pred, alpha=0.65)
                            low = min(float(np.min(yte)), float(np.min(final_pred)))
                            high = max(float(np.max(yte)), float(np.max(final_pred)))
                            ax.plot([low, high], [low, high], linestyle="--")
                            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs Predicted")
                            st.pyplot(fig); plt.close(fig)
                            residuals = np.asarray(yte, dtype=float) - np.asarray(final_pred, dtype=float)
                            fig, ax = plt.subplots()
                            ax.scatter(final_pred, residuals, alpha=0.65); ax.axhline(0, linestyle="--")
                            ax.set_xlabel("Predicted"); ax.set_ylabel("Residual"); ax.set_title("Residual plot")
                            st.pyplot(fig); plt.close(fig)
                    with explain_tab:
                        model_obj = best_pipe.named_steps["model"]
                        importance = getattr(model_obj, "feature_importances_", None)
                        coefficients = getattr(model_obj, "coef_", None)
                        try:
                            names = best_pipe.named_steps["prep"].get_feature_names_out()
                            if importance is not None:
                                imp_df = pd.DataFrame({"Feature": names, "Importance": importance}).sort_values("Importance", ascending=False).head(20)
                                st.write("Top 20 influential features (tree-based model)")
                                st.bar_chart(imp_df.set_index("Feature"))
                                st.dataframe(imp_df, use_container_width=True)
                            elif coefficients is not None:
                                coef = np.asarray(coefficients)
                                vals = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
                                imp_df = pd.DataFrame({"Feature": names, "Absolute coefficient": vals}).sort_values("Absolute coefficient", ascending=False).head(20)
                                st.write("Top 20 features by absolute coefficient (linear models)")
                                st.bar_chart(imp_df.set_index("Feature"))
                                st.dataframe(imp_df, use_container_width=True)
                            else:
                                st.info("This model does not expose built-in feature importance or coefficients. Try a tree-based or linear model.")
                        except Exception as ex:
                            st.info(f"Feature explanation unavailable for this model: {ex}")


# ---------- Reports ----------
elif page == "Report & Export":
    st.header("📄 Report & Export Center")
    st.write("Export the active dataset and a concise data profile.")
    csv=df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download active dataset (CSV)",csv,file_name="insightpilot_cleaned_data.csv",mime="text/csv")
    report={"dataset":st.session_state.get("ip_filename","dataset"),"rows":int(df.shape[0]),"columns":int(df.shape[1]),"missing_cells":int(df.isna().sum().sum()),"duplicate_rows":int(df.duplicated().sum()),"column_types":df.dtypes.astype(str).to_dict(),"missing_by_column":df.isna().sum().to_dict()}
    st.json(report)
    st.download_button("⬇️ Download profile (JSON)",json.dumps(report,indent=2,default=str),file_name="insightpilot_profile.json",mime="application/json")
    summary=df.describe(include="all").T.to_csv().encode("utf-8")
    st.download_button("⬇️ Download summary statistics (CSV)",summary,file_name="insightpilot_summary.csv",mime="text/csv")
    st.divider()
    st.subheader("🧾 Generate a complete HTML data report")
    st.caption("Creates a portable report containing dataset overview, column quality, descriptive statistics and recommendations.")
    missing_pct=(df.isna().mean()*100).round(2)
    quality=pd.DataFrame({
        "Column":df.columns.astype(str),
        "Data type":df.dtypes.astype(str).values,
        "Missing":df.isna().sum().values,
        "Missing (%)":missing_pct.values,
        "Unique values":df.nunique(dropna=True).values
    })
    numeric=df.select_dtypes(include=np.number)
    stats=numeric.describe().T.round(3) if not numeric.empty else pd.DataFrame()
    recommendations=[]
    for col in df.columns:
        miss=int(df[col].isna().sum())
        if miss: recommendations.append(f"{col}: {miss} missing values ({miss/len(df)*100:.1f}%). Consider appropriate imputation or row review.")
        if df[col].nunique(dropna=True)<=1: recommendations.append(f"{col}: constant column; consider removing if not meaningful.")
    dup=int(df.duplicated().sum())
    if dup: recommendations.append(f"Dataset: {dup} duplicate rows; review before modelling.")
    if not recommendations: recommendations=["No basic missing-value, constant-column, or duplicate-row issues detected."]
    html_report=f"""<!doctype html><html><head><meta charset="utf-8"><title>InsightPilot Data Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:35px auto;padding:0 20px;color:#222}}
h1,h2{{color:#17365d}}table{{border-collapse:collapse;width:100%;margin:12px 0 25px;font-size:13px}}
th,td{{border:1px solid #ddd;padding:7px;text-align:left}}th{{background:#edf2f8}}
.metric{{display:inline-block;background:#f3f6fa;padding:14px;margin:5px;border-radius:8px}}
li{{margin:8px 0}}.note{{color:#666;font-size:12px}}</style></head><body>
<h1>InsightPilot — Dataset Analysis Report</h1><p>Generated from the active dataset.</p>
<div class="metric"><b>Rows</b><br>{len(df):,}</div><div class="metric"><b>Columns</b><br>{df.shape[1]}</div>
<div class="metric"><b>Missing cells</b><br>{int(df.isna().sum().sum()):,}</div><div class="metric"><b>Duplicate rows</b><br>{dup:,}</div>
<h2>Column quality profile</h2>{quality.to_html(index=False,escape=True)}
<h2>Numeric descriptive statistics</h2>{stats.to_html(escape=True) if not stats.empty else '<p>No numeric columns.</p>'}
<h2>Recommendations</h2><ul>{''.join('<li>'+str(x).replace('&','&amp;').replace('<','&lt;')+'</li>' for x in recommendations)}</ul>
<p class="note">InsightPilot report — descriptive checks are not a substitute for domain validation.</p></body></html>"""
    st.download_button("📥 Download complete HTML report",html_report.encode("utf-8"),
                      file_name="InsightPilot_Data_Report.html",mime="text/html",type="primary")


st.sidebar.divider()
st.sidebar.caption(f"Active data: {len(df):,} rows × {df.shape[1]} columns")
st.sidebar.caption("InsightPilot · Integrated Platform")
