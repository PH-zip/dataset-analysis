import streamlit as st

# Configurações da aba principal
st.set_page_config(
    page_title="Aplicação Multipáginas",
    page_icon=":book:",
    layout="wide"
)

# Menu de navegação
st.sidebar.title("Navegação")
menu = st.sidebar.radio("Selecione uma página:", ["Home", "Página 1", "Página 2"])

# Redireciona para a página selecionada
if menu == "Home":
    from Home import main as home
    home()
elif menu == "Página 1":
    from Pagina1 import main as pagina1
    pagina1()
elif menu == "Página 2":
    from Pagina2 import main as pagina2
    pagina2()
