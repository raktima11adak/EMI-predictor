from pathlib import Path
import sys
import json
import streamlit as st
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from pipeline import (DATA_PATH, ARTIFACT_DIR, TARGET_CLASS, TARGET_REG,
                      engineer_features, load_data, train_all)

st.set_page_config(page_title='EMIPredict AI', page_icon='💳', layout='wide')
st.title('EMIPredict AI')
st.caption('Financial risk assessment for EMI applications | Educational decision support')

if 'emi_data' not in st.session_state:
    st.session_state.emi_data = load_data(DATA_PATH)
df = st.session_state.emi_data
page = st.sidebar.radio('Workspace', ['Risk assessment', 'Data explorer', 'Data management', 'Model results'])


def load_model():
    path = ARTIFACT_DIR / 'emipredict_models.joblib'
    return joblib.load(path) if path.exists() else None


if page == 'Risk assessment':
    st.subheader('Assess an EMI application')
    st.info('Predictions are based on the supplied sample dataset. Review the model results and sample size before using outputs. This is not a lending decision or financial advice.')
    if load_model() is None:
        st.warning('Models have not been trained yet. Use the button below to train them on the selected CSV.')
        if st.button('Train models now', type='primary'):
            with st.spinner('Training six model candidates and saving the best candidates...'):
                try:
                    summary = train_all(DATA_PATH)
                    st.success('Training complete. Open Model results to review metrics.')
                    st.json({'best_classifier': summary['best_classification_model'],
                             'best_regressor': summary['best_regression_model']})
                    st.rerun()
                except Exception as exc:
                    st.error(f'Training failed: {exc}')
    model_bundle = load_model()
    if model_bundle:
        with st.form('application'):
            st.write('Enter applicant details')
            base = df.drop(columns=[TARGET_CLASS, TARGET_REG], errors='ignore')
            categorical = list(base.select_dtypes(include=['object', 'string', 'category']).columns)
            numeric = list(base.select_dtypes(include=['number']).columns)
            values = {}
            left, right = st.columns(2)
            for i, col in enumerate(numeric):
                median = float(base[col].median()) if base[col].notna().any() else 0.0
                minimum = float(base[col].min()) if base[col].notna().any() else 0.0
                maximum = float(base[col].max()) if base[col].notna().any() else max(minimum + 1, 1)
                values[col] = (left if i % 2 == 0 else right).number_input(col.replace('_', ' ').title(),
                                     min_value=0.0, value=max(0.0, median), step=1.0,
                                     help=f'Sample range: {minimum:g} to {maximum:g}')
            for i, col in enumerate(categorical):
                opts = sorted(str(v) for v in base[col].dropna().unique())
                if opts:
                    values[col] = (left if i % 2 == 0 else right).selectbox(col.replace('_', ' ').title(), opts)
            submitted = st.form_submit_button('Assess application', type='primary')
        if submitted:
            row = pd.DataFrame([values])
            features = engineer_features(row)
            try:
                label = model_bundle['classifier'].predict(features)[0]
                amount = max(0.0, float(model_bundle['regressor'].predict(features)[0]))
                probs = model_bundle['classifier'].predict_proba(features)[0]
                labels = model_bundle['classifier'].named_steps['model'].classes_
                c1, c2 = st.columns(2)
                c1.metric('Predicted eligibility', str(label))
                c2.metric('Maximum monthly EMI estimate', f'₹{amount:,.0f}')
                st.write('Class probability estimates')
                st.bar_chart(pd.Series(probs, index=labels, name='Probability'))
            except Exception as exc:
                st.error(f'Could not score this application: {exc}')

elif page == 'Data explorer':
    st.subheader('Explore the supplied data')
    a, b, c = st.columns(3)
    a.metric('Records', f'{len(df):,}')
    b.metric('Features', str(len(df.columns) - 2))
    c.metric('Missing cells', f'{int(df.isna().sum().sum()):,}')
    st.write('Eligibility distribution')
    st.bar_chart(df[TARGET_CLASS].value_counts(dropna=False))
    st.write('Applications by EMI scenario')
    st.bar_chart(df['emi_scenario'].value_counts(dropna=False))
    st.write('Salary and requested amount')
    st.scatter_chart(df, x='monthly_salary', y='requested_amount', color='emi_eligibility')
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)
    st.download_button('Download current CSV', df.to_csv(index=False).encode('utf-8'), 'emi_database.csv', 'text/csv')

elif page == 'Data management':
    st.subheader('Financial profile records')
    st.caption('Changes are held in this app session and can be downloaded. Original source data remains unchanged.')
    st.dataframe(df, use_container_width=True, height=420)
    with st.expander('Add a record'):
        with st.form('add_record'):
            row = {}
            for col in df.columns:
                if col in [TARGET_CLASS]:
                    row[col] = st.selectbox(col, sorted(df[col].dropna().astype(str).unique()))
                elif col in df.select_dtypes(include='number').columns:
                    row[col] = st.number_input(col, value=float(df[col].median()) if df[col].notna().any() else 0.0)
                else:
                    opts = sorted(df[col].dropna().astype(str).unique())
                    row[col] = st.selectbox(col, opts) if opts else st.text_input(col)
            if st.form_submit_button('Add record'):
                st.session_state.emi_data = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                st.success('Record added to this session.')
                st.rerun()
    with st.expander('Edit or delete by row number'):
        row_id = st.number_input('Row number (starting at 0)', min_value=0, max_value=max(0, len(df)-1), step=1)
        edited = st.data_editor(df.iloc[[int(row_id)]], use_container_width=True, key='record_editor')
        e1, e2 = st.columns(2)
        if e1.button('Save edited row'):
            st.session_state.emi_data.iloc[int(row_id)] = edited.iloc[0]
            st.success('Updated in this session.')
            st.rerun()
        if e2.button('Delete row', type='secondary'):
            st.session_state.emi_data = df.drop(df.index[int(row_id)]).reset_index(drop=True)
            st.success('Deleted from this session.')
            st.rerun()
    st.download_button('Export managed data as CSV', df.to_csv(index=False).encode('utf-8'), 'emi_database_managed.csv', 'text/csv')

else:
    st.subheader('Model comparison')
    result_path = ROOT / 'reports' / 'training_summary.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        st.caption(f"Evaluation used {result['evaluation_rows']} held-out profiles; training used {result['training_rows']} profiles.")
        if result.get('sample_mode'):
            st.warning('Training was capped at a 20,000-row sample for speed.')
        st.markdown('#### Classification')
        st.dataframe(pd.DataFrame(result['classification']), use_container_width=True)
        st.markdown('#### Regression')
        st.dataframe(pd.DataFrame(result['regression']), use_container_width=True)
        st.markdown('#### Selection')
        st.write(f"Classifier: **{result['best_classification_model']}**")
        st.write(f"Regressor: **{result['best_regression_model']}**")
        if result.get('mlflow_status'):
            st.info(result['mlflow_status'])
        if result.get('quality'):
            st.json(result['quality'])
    else:
        st.info('Train models from Risk assessment to generate the comparison report and MLflow runs.')
        if st.button('Train models'):
            with st.spinner('Training model candidates...'):
                try:
                    train_all(DATA_PATH)
                    st.success('Training complete.')
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
