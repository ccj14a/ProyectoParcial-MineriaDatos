import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Configuración de la interfaz en modo ancho
st.set_page_config(page_title="Parcial - Minería de Datos", layout="wide", page_icon="📊")

# Control de versión en cabecera (Requisito del examen)
st.sidebar.caption("📌 Versión de la Aplicación: v1.0.0 (Estable)")
st.sidebar.markdown("---")

# ==========================================
# MENÚ PRINCIPAL MEDIANTE PESTAÑAS (TABS)
# ==========================================
tab_app, tab_manual, tab_informe = st.tabs([
    "🚀 1. Aplicación Ejecutable", 
    "📖 2. Manual de Usuario", 
    "📄 3. Informe de la Aplicación"
])

# ==========================================
# PESTAÑA 1: LA APLICACIÓN INTERACTIVA
# ==========================================
with tab_app:
    st.title("📊 Sistema de Vectorización y Visualización de Texto")
    st.markdown("Carga tus datos, selecciona la técnica matemática y exporta tu nuevo dataset estructurado.")
    
    # --- INGRESO DE DATOS ---
    st.subheader("📥 1. Ingreso de Datos")
    origen_datos = st.radio("Selecciona cómo deseas ingresar los datos de texto:", ["Texto Manual (Escribir líneas)", "Cargar Archivo Local (CSV/Excel)"], horizontal=True)
    
    textos_a_procesar = []
    
    if origen_datos == "Texto Manual (Escribir líneas)":
        texto_libre = st.text_area(
            "Escribe tus oraciones de prueba (una frase por línea):",
            "La minería de datos es una disciplina matemática.\nEl procesamiento de texto requiere vectorización.\nMinería de datos y procesamiento de texto."
        )
        textos_a_procesar = [linea.strip() for linea in texto_libre.split("\n") if linea.strip()]
    else:
        archivo_cargado = st.file_uploader("Sube un archivo estructurado con texto", type=["csv", "xlsx"])
        if archivo_cargado is not None:
            try:
                if archivo_cargado.name.endswith('.csv'):
                    df_input = pd.read_csv(archivo_cargado)
                else:
                    df_input = pd.read_excel(archivo_cargado)
                
                columna_texto = st.selectbox("Selecciona la columna exacta que contiene el texto:", df_input.columns)
                textos_a_procesar = df_input[columna_texto].dropna().astype(str).tolist()
                st.success(f"✅ Se cargaron exitosamente {len(textos_a_procesar)} filas de texto.")
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")

    # --- APLICACIÓN DEL MODELO MATEMÁTICO ---
    if len(textos_a_procesar) > 0:
        st.markdown("---")
        st.subheader("⚙️ 2. Configuración y Aplicación del Modelo")
        
        tipo_vectorizacion = st.selectbox(
            "Selecciona el algoritmo de vectorización matemática a aplicar:", 
            ["Count Vectorizer (Bag of Words)", "TF-IDF (Term Frequency - Inverse Document Frequency)"]
        )
        
        # Renderizado del Modelo Matemático en tiempo real (Requisito)
        st.markdown("#### 🧮 Ecuación del Modelo Seleccionado")
        if tipo_vectorizacion == "Count Vectorizer (Bag of Words)":
            st.info("💡 **Count Vectorizer:** Construye un espacio vectorial basado únicamente en la frecuencia absoluta de aparición de cada token.")
            st.latex(r"Vector\_D_i = [f(w_1), f(w_2), \dots, f(w_n)]")
            st.caption("Donde $f(w_j)$ representa la frecuencia o conteo del término $j$ dentro del documento $i$.")
            vectorizador = CountVectorizer(lowercase=True)
        else:
            st.info("💡 **TF-IDF:** Pondera la importancia de las palabras reduciendo el peso de los términos hiper-frecuentes en el corpus.")
            st.latex(r"TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)")
            st.latex(r"TF(t, d) = \frac{f(t, d)}{\sum_{t'} f(t', d)} \quad \vert \quad IDF(t, D) = \log \left( \frac{|D|}{1 + |\{d \in D : t \in d\}|} \right)")
            st.caption("Donde $TF$ es la frecuencia normalizada del término y $IDF$ evalúa la dispersión global del término en todos los documentos.")
            vectorizador = TfidfVectorizer(lowercase=True)

        # --- VISUALIZACIÓN DE SALIDAS ---
        st.markdown("---")
        st.subheader("🖥️ 3. Visualización de Salidas y Descarga")
        
        with st.spinner("Ejecutando transformaciones matriciales..."):
            # Aplicación de Scikit-Learn
            matriz_num = vectorizador.fit_transform(textos_a_procesar)
            vocabulario = vectorizador.get_feature_names_out()
            
            # Construcción del nuevo dataset estructurado
            df_resultado = pd.DataFrame(data=matriz_num.toarray(), columns=vocabulario)
            df_resultado.insert(0, "Texto_Original", textos_a_procesar)
            
            sub_tab1, sub_tab2 = st.tabs(["📋 Nuevo Dataset Vectorizado", "📈 Análisis de Frecuencias"])
            
            with sub_tab1:
                st.markdown("##### Vista Previa de la Matriz Generada (Documento x Término)")
                st.dataframe(df_resultado, use_container_width=True)
                
                # Conversión y Botón de grabado/descarga del nuevo dataset (Requisito)
                csv_bytes = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Guardar y Descargar Nuevo Dataset (CSV)",
                    data=csv_bytes,
                    file_name="nuevo_dataset_vectorizado.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            with sub_tab2:
                st.markdown("##### Palabras más influyentes dentro del Corpus")
                pesos_acumulados = df_resultado.drop(columns=["Texto_Original"]).sum().sort_values(ascending=False)
                if not pesos_acumulados.empty:
                    st.bar_chart(pesos_acumulados.head(20))
                else:
                    st.warning("Sin datos suficientes para proyectar gráficas.")
    else:
        st.warning("⚠️ Esperando el ingreso de datos válidos para inicializar los motores matemáticos.")

# ==========================================
# PESTAÑA 2: MANUAL DE USUARIO (Requisito 2)
# ==========================================
with tab_manual:
    st.header("📘 Manual de Usuario")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏢 Carátula de Operación")
        st.markdown("""
        * **Curso:** Minería de Datos
        * **Evaluación:** Examen Parcial
        * **Integrantes:** Grupo de 2 Alumnos
        * **Estado del Arte:** Despliegue Local Exitoso
        """)
    with col2:
        st.subheader("📌 Grabado de Versión")
        st.info("**Versión Actual:** v1.0.0  \n**Última Modificación:** Mayo 2026  \n**Cambios:** Integración total de interfaz gráfica, soporte para múltiples tipos de vectorización y renderizado dinámico de fórmulas.")

    st.markdown("### 🚀 Procedimiento de Puesta en Marcha")
    st.code("""
# 1. Instalar las librerías necesarias mediante pip
pip install streamlit scikit-learn pandas openpyxl

# 2. Situarse en la carpeta raíz del proyecto y arrancar la aplicación
streamlit run src/app.py
    """, language="bash")
    
    st.markdown("""
    ### 🛑 Restricciones del Sistema
    1. **Formato de Archivos:** Para cargas masivas, solo se admiten archivos estructurados con extensión `.csv` o `.xlsx`.
    2. **Valores Nulos:** Las filas que no contengan texto (vacías) serán automáticamente omitidas por el preprocesador para resguardar la consistencia matemática.
    
    ### 🎯 Guía Operativa Paso a Paso
    * **Paso 1 (Ingreso de Datos):** Diríjase a la sección '1. Ingreso de Datos'. Elija la modalidad manual para tipear oraciones de prueba rápidas o cargue un dataset completo utilizando el explorador de archivos. Si sube un archivo, seleccione la columna exacta a vectorizar en el desplegable.
    * **Paso 2 (Aplicar Vectorización):** Seleccione el enfoque algorítmico deseado en el módulo 2. El sistema proyectará de forma inmediata la ecuación formal bajo la cual opera.
    * **Paso 3 (Obtener Respuestas y Grabado):** En el módulo 3, inspeccione visualmente la matriz numérica generada. Haga clic en el botón **'Guardar y Descargar Nuevo Dataset (CSV)'** para persistir los resultados localmente en su almacenamiento.
    """)

# ==========================================
# PESTAÑA 3: INFORME DE LA APLICACIÓN (Requisito 3)
# ==========================================
with tab_informe:
    st.header("📄 Informe Técnico de la Aplicación")
    st.markdown("---")
    
    st.subheader("📝 Descripción de la Aplicación")
    st.markdown("""
    Esta solución de software ha sido concebida para abordar una problemática medular en la **Minería de Datos**: la conversión de texto no estructurado (lenguaje natural humano) en matrices de características numéricas aptas para el consumo de modelos predictivos y de agrupamiento (Clustering, Clasificación, Regresión).
    La aplicación rompe el esquema de "caja negra", combinando la flexibilidad de cómputo de la librería `scikit-learn` con un entorno pedagógico interactivo que enseña las bases matemáticas del procesamiento de texto.
    """)
    
    st.subheader("🏗️ Esquema y Arquitectura del Sistema")
    st.markdown("""
    A continuación, se detalla el flujo secuencial de los datos a lo largo de los diferentes componentes de software:

    1. **Ingreso de texto:** el usuario escribe frases o carga un archivo CSV/Excel.
    2. **Preprocesamiento:** la aplicación selecciona y normaliza los textos válidos.
    3. **Vectorización:** `scikit-learn` transforma las palabras mediante Count Vectorizer o TF-IDF.
    4. **Visualización:** `pandas` presenta la matriz generada y Streamlit grafica las frecuencias.
    5. **Exportación:** el resultado se descarga como un nuevo archivo CSV.

    **Tecnologías empleadas:** Python, Streamlit, pandas, scikit-learn y openpyxl.
    """)
