import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorizar_textos(textos, vectorizador):
    """Genera la matriz numerica y detecta entradas sin vocabulario util."""
    try:
        matriz = vectorizador.fit_transform(textos)
        return matriz, vectorizador.get_feature_names_out()
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            return None, None
        raise


OPCIONES_NGRAMAS = {
    "Unigramas (1,1): palabras individuales": (1, 1),
    "Bigramas (1,2): palabras y pares consecutivos": (1, 2),
    "Trigramas (1,3): palabras, pares y tríos consecutivos": (1, 3),
}


# Configuración de la interfaz en modo ancho
st.set_page_config(page_title="Parcial - Minería de Datos", layout="wide", page_icon="📊")

# Control de versión en cabecera (Requisito del examen)
st.sidebar.caption("📌 Versión de la Aplicación: v1.2.0 (Estable)")
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
            "Minería de datos minería de datos permite analizar texto.\nLa minería de datos transforma texto en información.\nTF-IDF resalta términos relevantes del corpus.\nLos n-gramas capturan minería de datos como contexto."
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
        st.info(
            "🔎 **Preprocesamiento aplicado:** los textos se convierten a minúsculas; "
            "en archivos se omiten valores nulos; la puntuación se ignora según el "
            "tokenizador; y se consideran tokens alfanuméricos de dos o más caracteres."
        )
        
        tipo_vectorizacion = st.selectbox(
            "Selecciona el algoritmo de vectorización matemática a aplicar:", 
            [
                "Count Vectorizer (Bag of Words)",
                "Vectorización Binaria (Presencia/Ausencia)",
                "TF-IDF (Term Frequency - Inverse Document Frequency)",
                "N-gramas",
            ]
        )
        
        # Renderizado del Modelo Matemático en tiempo real (Requisito)
        st.markdown("#### 🧮 Ecuación del Modelo Seleccionado")
        if tipo_vectorizacion == "Count Vectorizer (Bag of Words)":
            st.info("💡 **Count Vectorizer:** Construye un espacio vectorial basado únicamente en la frecuencia absoluta de aparición de cada token.")
            st.latex(r"Vector\_D_i = [f(w_1), f(w_2), \dots, f(w_n)]")
            st.caption("Donde $f(w_j)$ representa la frecuencia o conteo del término $j$ dentro del documento $i$.")
            vectorizador = CountVectorizer(lowercase=True)
        elif tipo_vectorizacion == "Vectorización Binaria (Presencia/Ausencia)":
            st.info("💡 **Vectorización Binaria:** Indica únicamente si cada término aparece en el documento, sin contar cuántas veces se repite.")
            st.latex(r"x_{ij} = \begin{cases} 1, & \text{si el término } j \text{ aparece en el documento } i \\ 0, & \text{si no aparece} \end{cases}")
            st.caption("Cada columna representa un término; el valor solo expresa presencia (1) o ausencia (0).")
            vectorizador = CountVectorizer(lowercase=True, binary=True)
        elif tipo_vectorizacion == "N-gramas":
            st.info("💡 **N-gramas:** Representan secuencias consecutivas de palabras para capturar contexto textual.")
            opcion_ngrama = st.selectbox(
                "Selecciona el rango de N-gramas a generar:",
                list(OPCIONES_NGRAMAS.keys()),
            )
            rango_ngrama = OPCIONES_NGRAMAS[opcion_ngrama]
            st.markdown(
                """
                **Ejemplo con el texto:** `La minería de datos`

                - Unigramas: `["la", "minería", "de", "datos"]`
                - Bigramas: `["la minería", "minería de", "de datos"]`
                - Trigramas: `["la minería de", "minería de datos"]`
                """
            )
            st.latex(r"x_{ij} = f(g_j, d_i)")
            st.caption(
                "Donde $g_j$ es el n-grama j, $d_i$ es el documento i y $f$ es la "
                "frecuencia de ese n-grama. En scikit-learn, (1,2) conserva unigramas "
                "y agrega bigramas; (1,3) conserva unigramas y agrega bigramas y trigramas."
            )
            vectorizador = CountVectorizer(lowercase=True, ngram_range=rango_ngrama)
        else:
            st.info("💡 **TF-IDF:** Pondera cada término con la variante exacta utilizada por `TfidfVectorizer(lowercase=True)` de scikit-learn.")
            st.latex(r"tf(t,d) = f(t,d)")
            st.latex(r"idf(t) = \log\left(\frac{1 + N}{1 + df(t)}\right) + 1")
            st.latex(r"\widetilde{x}_{t,d} = tf(t,d) \times idf(t)")
            st.latex(r"x_d = \frac{\widetilde{x}_d}{\sqrt{\sum_j \widetilde{x}_{j,d}^{\,2}}}")
            st.caption("Donde $N$ es el número de documentos y $df(t)$ es la cantidad de documentos que contienen el término. El último paso aplica normalización L2.")
            vectorizador = TfidfVectorizer(lowercase=True)

        # --- VISUALIZACIÓN DE SALIDAS ---
        st.markdown("---")
        st.subheader("🖥️ 3. Visualización de Salidas y Descarga")
        
        with st.spinner("Ejecutando transformaciones matriciales..."):
            matriz_num, vocabulario = vectorizar_textos(textos_a_procesar, vectorizador)

        if matriz_num is None:
            if tipo_vectorizacion == "N-gramas":
                st.error("No se encontraron n-gramas válidos para vectorizar.")
            else:
                st.error(
                    "No se encontraron palabras válidas para vectorizar. Ingresa términos "
                    "de al menos dos caracteres o revisa el contenido del archivo."
                )
        else:
            # Construcción del nuevo dataset estructurado con términos o N-gramas.
            df_resultado = pd.DataFrame(data=matriz_num.toarray(), columns=vocabulario)
            df_resultado.insert(0, "Texto_Original", textos_a_procesar)
            
            sub_tab1, sub_tab2 = st.tabs(["📋 Nuevo Dataset Vectorizado", "📈 Análisis de Frecuencias"])
            
            with sub_tab1:
                st.markdown("##### Vista Previa de la Matriz Generada (Documento x Término)")
                st.dataframe(df_resultado, width="stretch")
                
                # Conversión y Botón de grabado/descarga del nuevo dataset (Requisito)
                csv_bytes = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Guardar y Descargar Nuevo Dataset (CSV)",
                    data=csv_bytes,
                    file_name="nuevo_dataset_vectorizado.csv",
                    mime="text/csv",
                    width="stretch"
                )
                
            with sub_tab2:
                st.markdown("##### Términos o N-gramas más influyentes dentro del Corpus")
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
        * **Integrantes:** Jeferson Coronado Cortez y Silva Burga Bryan
        * **Docente:** Completar nombre antes de la entrega
        * **Fecha de entrega:** Completar fecha oficial
        * **Estado:** Aplicación funcional para demostración académica
        """)
    with col2:
        st.subheader("📌 Grabado de Versión")
        st.info("**Versión Actual:** v1.2.0  \n**Última Modificación:** Mayo 2026  \n**Cambios:** Se añadió análisis de N-gramas configurable, se precisó el modelo TF-IDF y se gestionan entradas sin vocabulario válido.")

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
    3. **Tokens Válidos:** Los textos se convierten a minúsculas, la puntuación se ignora durante la tokenización y se consideran términos alfanuméricos de dos o más caracteres. Si no queda vocabulario válido, se informa al usuario sin detener la aplicación.

    ### 🔗 N-gramas
    Los N-gramas permiten representar contexto mediante secuencias consecutivas: los **unigramas** son palabras individuales, los **bigramas** son pares y los **trigramas** son grupos de tres palabras. Para un documento $d_i$, el valor de cada columna se calcula como $x_{ij}=f(g_j,d_i)$, donde $g_j$ es el N-grama y $f$ su frecuencia.

    Para el texto `La minería de datos`, se obtienen unigramas como `minería`, bigramas como `minería de` y trigramas como `minería de datos`. Las opciones `(1,2)` y `(1,3)` incluyen también los N-gramas de longitudes inferiores para conservar información de palabras individuales.
    
    ### 🎯 Guía Operativa Paso a Paso
    * **Paso 1 (Ingreso de Datos):** Diríjase a la sección '1. Ingreso de Datos'. Elija la modalidad manual para tipear oraciones de prueba rápidas o cargue un dataset completo utilizando el explorador de archivos. Si sube un archivo, seleccione la columna exacta a vectorizar en el desplegable.
    * **Paso 2 (Aplicar Vectorización):** Seleccione Count Vectorizer para frecuencias, Vectorización Binaria para presencia/ausencia, TF-IDF para ponderación normalizada o N-gramas para analizar secuencias. Si elige N-gramas, seleccione unigramas, bigramas o trigramas. El sistema proyectará la ecuación formal bajo la cual opera.
    * **Paso 3 (Obtener Respuestas y Grabado):** En el módulo 3, inspeccione visualmente la matriz numérica generada. Haga clic en el botón **'Guardar y Descargar Nuevo Dataset (CSV)'** para persistir los resultados localmente en su almacenamiento.

    ### 🎓 Secuencia Sugerida para la Exposición
    El dataset `data/ejemplo_textos.csv` contiene repeticiones y frases compartidas para evidenciar las diferencias entre modelos:

    1. Con **Count Vectorizer**, observe que `minería` y `datos` acumulan frecuencia.
    2. Con **Vectorización Binaria**, verifique que una repetición se representa solo como presencia (`1`).
    3. Con **TF-IDF**, explique que los términos frecuentes en muchos documentos pierden peso relativo.
    4. Con **N-gramas (1,2)**, identifique `minería de`; luego use **(1,3)** para identificar `minería de datos` como columna del nuevo dataset.
    5. Descargue el CSV para demostrar el guardado del dataset vectorizado.
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
    La aplicación rompe el esquema de "caja negra", combinando la flexibilidad de cómputo de la librería `scikit-learn` con un entorno pedagógico interactivo que enseña las bases matemáticas de Count Vectorizer, vectorización binaria, TF-IDF y N-gramas.
    """)
    
    st.subheader("🏗️ Esquema y Arquitectura del Sistema")
    st.markdown("""
    A continuación, se detalla el flujo secuencial de los datos a lo largo de los diferentes componentes de software:

    1. **Ingreso de texto:** el usuario escribe frases o carga un archivo CSV/Excel.
    2. **Preprocesamiento:** la aplicación omite valores nulos, transforma a minúsculas y tokeniza términos alfanuméricos de dos o más caracteres, ignorando puntuación.
    3. **Vectorización:** `scikit-learn` transforma las palabras mediante Count Vectorizer, Vectorización Binaria, TF-IDF o N-gramas, mostrando el modelo matemático correspondiente.
    4. **Visualización:** `pandas` presenta la matriz generada y Streamlit grafica las frecuencias.
    5. **Exportación:** el resultado se descarga como un nuevo archivo CSV.

    Si los textos no generan términos válidos, la interfaz presenta una advertencia comprensible en lugar de interrumpir la ejecución.

    En la modalidad **N-gramas**, el usuario selecciona rangos `(1,1)`, `(1,2)` o `(1,3)`. El modelo genera columnas para secuencias consecutivas y calcula $x_{ij}=f(g_j,d_i)$; así, los bigramas y trigramas permiten observar expresiones con contexto como `minería de datos`.

    El proyecto incorpora un dataset de demostración preparado para comparar frecuencias absolutas, presencia/ausencia, relevancia TF-IDF y secuencias N-grama dentro de un mismo flujo visual y exportable.

    **Tecnologías empleadas:** Python, Streamlit, pandas, scikit-learn y openpyxl.
    """)
