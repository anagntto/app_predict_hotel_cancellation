import streamlit as st

st.set_page_config(
    page_title='Análise de Inadimplencia'
)

def main():
    st.title('Bem Vindo à Previsão de Inadimplência')

    st.write(
        '''
        Está é uma aplicação de previsão de inadimplência desenvolvida com Streamlit.
        Aqui você poderá inserir os dados do cliente e obter uma previsão sobre se é recomendado
        ou não disponibilizar crédito para esse cliente.
        '''
    )

if __name__ == '__main__':
    main()
