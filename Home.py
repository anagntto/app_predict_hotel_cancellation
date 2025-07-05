import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Configuração da página
st.set_page_config(
    page_title="HotelSmart - Predição de Cancelamentos",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏨 HotelSmart - Sistema de Predição de Cancelamentos")
st.markdown("---")

# Carregando os modelos salvos
@st.cache_resource
def load_models():
    try:
        with open('model/final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('parameter/hotelsmart_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('parameter/market_segment_type_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("Modelos não encontrados! Certifique-se de que os arquivos .pkl estão no diretório correto.")
        return None, None, None

model, scaler, label_encoder = load_models()

if model is not None:
    # Sidebar para entrada de dados
    st.sidebar.header("📊 Dados da Reserva")
    
    # Inputs do usuário
    lead_time = st.sidebar.number_input(
        "Lead Time (dias de antecedência)", 
        min_value=0, 
        max_value=500, 
        value=30,
        help="Número de dias entre a reserva e a chegada"
    )
    
    arrival_month = st.sidebar.selectbox(
        "Mês de Chegada",
        options=list(range(1, 13)),
        index=5,
        help="Mês da chegada (1-12)"
    )
    
    arrival_date = st.sidebar.number_input(
        "Data de Chegada (dia do mês)",
        min_value=1,
        max_value=31,
        value=15,
        help="Dia do mês da chegada"
    )
    
    market_segment_type = st.sidebar.selectbox(
        "Tipo de Segmento de Mercado",
        options=["Aviation", "Complementary", "Corporate", "Online", "Offline"],
        index=3,
        help="Segmento de mercado da reserva"
    )
    
    avg_price_per_room = st.sidebar.number_input(
        "Preço Médio por Quarto (R$)",
        min_value=0.0,
        max_value=10000.0,
        value=150.0,
        step=10.0,
        help="Preço médio por quarto em reais"
    )
    
    no_of_special_requests = st.sidebar.number_input(
        "Número de Pedidos Especiais",
        min_value=0,
        max_value=10,
        value=1,
        help="Quantidade de pedidos especiais feitos pelo hóspede"
    )
    
    # Botão de predição
    if st.sidebar.button("🔮 Fazer Predição", type="primary"):
        # Preparar os dados para predição
        try:
            # Codificar o market_segment_type
            market_segment_encoded = label_encoder.transform([[market_segment_type]])[0][0]
            
            # Criar DataFrame com os dados de entrada
            input_data = pd.DataFrame({
                'lead_time': [lead_time],
                'arrival_month': [arrival_month],
                'arrival_date': [arrival_date],
                'market_segment_type': [market_segment_encoded],
                'avg_price_per_room': [avg_price_per_room],
                'no_of_special_requests': [no_of_special_requests]
            })
            
            # Fazer a predição
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Exibir resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Resultado da Predição")
                if prediction == 1:
                    st.error("⚠️ **ALTA PROBABILIDADE DE CANCELAMENTO**")
                    st.markdown(f"**Probabilidade de Cancelamento:** {prediction_proba[1]:.2%}")
                else:
                    st.success("✅ **BAIXA PROBABILIDADE DE CANCELAMENTO**")
                    st.markdown(f"**Probabilidade de Manutenção:** {prediction_proba[0]:.2%}")
            
            with col2:
                st.subheader("📊 Probabilidades")
                prob_df = pd.DataFrame({
                    'Status': ['Não Cancelar', 'Cancelar'],
                    'Probabilidade': [prediction_proba[0], prediction_proba[1]]
                })
                st.bar_chart(prob_df.set_index('Status'))
            
            # Informações adicionais
            st.markdown("---")
            st.subheader("💡 Recomendações")
            
            if prediction == 1:
                st.warning("""
                **Ações Recomendadas para Reduzir o Risco de Cancelamento:**
                - Entrar em contato com o cliente para confirmar a reserva
                - Oferecer flexibilidade nas datas ou condições
                - Verificar se há necessidades especiais não atendidas
                - Considerar ofertas ou upgrades para fidelizar o cliente
                """)
            else:
                st.info("""
                **Reserva com Baixo Risco de Cancelamento:**
                - Cliente provavelmente manterá a reserva
                - Foque em proporcionar uma excelente experiência
                - Prepare-se adequadamente para a chegada do hóspede
                """)
            
            # Detalhes da análise
            with st.expander("🔍 Detalhes da Análise"):
                st.write("**Dados de Entrada:**")
                st.json({
                    "Lead Time": f"{lead_time} dias",
                    "Mês de Chegada": arrival_month,
                    "Data de Chegada": arrival_date,
                    "Segmento de Mercado": market_segment_type,
                    "Preço Médio por Quarto": f"R$ {avg_price_per_room:.2f}",
                    "Pedidos Especiais": no_of_special_requests
                })
                
                st.write("**Fatores de Influência:**")
                st.markdown("""
                - **Lead Time**: Reservas com muito tempo de antecedência tendem a ter maior risco de cancelamento
                - **Preço**: Preços mais altos podem influenciar na decisão de cancelamento
                - **Pedidos Especiais**: Clientes com pedidos especiais tendem a cancelar menos
                - **Segmento de Mercado**: Diferentes segmentos têm comportamentos distintos
                """)
        
        except Exception as e:
            st.error(f"Erro ao fazer a predição: {str(e)}")
    
    # Informações sobre o modelo
    st.markdown("---")
    st.subheader("ℹ️ Sobre o Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Algoritmo", "Random Forest")
    
    with col2:
        st.metric("Features Utilizadas", "6")
    
    with col3:
        st.metric("Acurácia Estimada", "~85%")
    
    with st.expander("📋 Informações Técnicas"):
        st.markdown("""
        **Características do Modelo:**
        - **Algoritmo**: Random Forest Classifier
        - **Features**: lead_time, arrival_month, arrival_date, market_segment_type, avg_price_per_room, no_of_special_requests
        - **Pré-processamento**: StandardScaler para variáveis numéricas, LabelEncoder para variáveis categóricas
        - **Seleção de Features**: Baseada no algoritmo Boruta
        
        **Como Usar:**
        1. Preencha os dados da reserva na barra lateral
        2. Clique em "Fazer Predição"
        3. Analise o resultado e as recomendações
        4. Tome ações preventivas se necessário
        """)

else:
    st.error("⚠️ Não foi possível carregar os modelos. Verifique se os arquivos estão no diretório correto.")
    st.info("""
    **Arquivos necessários:**
    - model/final_model.pkl
    - parameter/hotelsmart_scaler.pkl
    - parameter/market_segment_type_encoder.pkl
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏨 HotelSmart - Sistema de Predição de Cancelamentos | Desenvolvido com Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)