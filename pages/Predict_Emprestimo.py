import streamlit as st
import requests
import pandas as pd
import json

def predict_emprestimo(df):
    url = 'https://additional-fish-anagntto-1147e51d.koyeb.app/empresa/predict'
    headers = {'Content-type': 'application/json'}
    data = json.dumps(df.to_dict(orient='records'))
    response = requests.post(url, data=data, headers=headers)
    prediction = response.json()[0]['prediction']
    return prediction

st.set_page_config(
    layout='wide',
    page_title='Previsão de Inadimplência'
)

st.title('Previsão de Inadimplência')

# Coleta de dados
col1, col2, col3, col4 = st.columns(4)
with col1:
    person_age = st.number_input('Insira a Idade do Cliente')
with col2:
    person_income = st.number_input('Insira a Renda do Cliente')
with col3:
    person_emp_length = st.number_input('Insira o tempo de emprego do Cliente')
with col4:
    loan_amnt = st.number_input('Insira o valor do empréstimo')

col5, col6, col7, col8 = st.columns(4)
with col5:
    loan_int_rate = st.number_input('Insira a Taxa de Juros')
with col6:
    loan_percent_income = st.number_input('Insira a relação empréstimo/renda')
with col7:
    cb_person_cred_hist_length = st.number_input('Insira o histórico de Crédito')
with col8:
    person_home_ownership = st.selectbox('Insira a posse da casa', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])

col9, col10, col11 = st.columns(3)
with col9:
    loan_intent = st.selectbox('Insira a Finalidade do Empréstimo',
                               ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
with col10:
    loan_grade = st.selectbox('Insira o Grau do Risco do Empréstimo',
                              ['D', 'B', 'C', 'A', 'E', 'F', 'G'])
with col11:
    cb_person_default_on_file = st.selectbox('Insira o registro de inadimplência',
                                             ['Y', 'N'])

# Organiza os dados
dict_data = {
    'person_age': float(person_age),
    'person_income': float(person_income),
    'person_emp_length': float(person_emp_length),
    'loan_amnt': float(loan_amnt),
    'loan_int_rate': float(loan_int_rate),
    'loan_percent_income': float(loan_percent_income),
    'cb_person_cred_hist_length': float(cb_person_cred_hist_length),
    'person_home_ownership': person_home_ownership,
    'loan_intent': loan_intent,
    'loan_grade': loan_grade,
    'cb_person_default_on_file': cb_person_default_on_file
}

# Converte para DataFrame
df = pd.DataFrame([dict_data])

# Botão de previsão
if st.button('Fazer Previsão'):
    with st.spinner('Nosso Modelo de Inteligência Artificial está analisando os dados....'):
        try:
            previsao = predict_emprestimo(df)
            if previsao != 0:
                st.markdown("<h4 style='color:red;'>Nosso modelo recomenda NÃO conceder crédito.</h4>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='color:green;'>Nosso modelo recomenda conceder crédito.</h4>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ocorreu um erro na previsão: {e}")
