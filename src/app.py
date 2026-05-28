import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ==========================================
# ⚙️ CAPA DE LÓGICA Y PROCESAMIENTO (MODULAR)
# ==========================================


def cargar_archivo(archivo):
    """Lee el archivo cargado de forma robusta con fallback de motores para evitar errores de parseo."""
    try:
        if archivo.name.endswith(".csv"):
            try:
                # Intento 1: Detección automática de separador
                return pd.read_csv(archivo, sep=None, engine="python")
            except Exception:
                # Intento 2: Fallback estándar si el motor de python falla por comas internas
                archivo.seek(0)
                return pd.read_csv(archivo, sep=",", on_bad_lines="skip")
        else:
            return pd.read_excel(archivo)
    except Exception as e:
        st.error(f"❌ Error crítico al leer el archivo: {e}")
        return None


def limpiar_y_extraer_textos(df, columna):
    """Extrae textos, elimina nulos y filtra valores por defecto de datasets reales."""
    if columna not in df.columns:
        return []

    # Obtener lista de strings limpios
    raw_textos = df[columna].dropna().astype(str).tolist()

    # Filtro inteligente para datasets de Kaggle (evita ruido matemático de 'No Positive' / 'No Negative')
    textos_limpios = [
        linea.strip()
        for linea in raw_textos
        if linea.strip().lower() not in ["no negative", "no positive", ""]
    ]
    return textos_limpios


def vectorizar_textos(textos, vectorizador):
    """Genera la matriz numérica y gestiona excepciones de vocabulario de forma segura."""
    try:
        matriz = vectorizador.fit_transform(textos)
        return matriz, vectorizador.get_feature_names_out()
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            return None, None
        raise


# Diccionario Global de Configuración
OPCIONES_NGRAMAS = {
    "Unigramas (1,1): palabras individuales": (1, 1),
    "Bigramas (1,2): palabras y pares consecutivos": (1, 2),
    "Trigramas (1,3): palabras, pares y tríos consecutivos": (1, 3),
}

# ==========================================
# 🖥️ CAPA DE INTERFAZ GRÁFICA (STREAMLIT)
# ==========================================

st.set_page_config(
    page_title="Parcial - Minería de Datos", layout="wide", page_icon="📊"
)

# Inicializar el estado de la sesión para el control de la Landing Page
if "ingresar" not in st.session_state:
    st.session_state["ingresar"] = False

# =========================================================
# 🏠 CAPA DE BIENVENIDA (LANDING PAGE - UI/UX EXPERT)
# =========================================================
if not st.session_state["ingresar"]:
    # Espaciado superior estético
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Contenedor principal alineado al centro
    col_cen, col_der, col_izq = st.columns([1, 8, 1])
    with col_der:
        st.markdown(
            """
            <div style='text-align: center;'>
                <h1 style='font-size: 3.5rem; color: #FF4B4B; margin-bottom: 0.5rem;'>🎯 LexiQuant Analytics</h1>
                <h3 style='font-size: 1.5rem; font-weight: 300; margin-bottom: 2.5rem;'>
                    Plataforma de Minería de Textos, Extracción de Características y Modelado Espacial
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tarjetas de presentación de funcionalidades mediante columnas con iconos nativos
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(
                """
                <div style='background-color: #262730; padding: 1.8rem; border-radius: 10px; border-left: 5px solid #FF4B4B; height: 100%;'>
                    <h4 style='margin-top:0;'>📥 Ingesta Flexible</h4>
                    <p style='font-size: 0.9rem; color: #A3A8B4;'>Soporte avanzado para procesamiento manual de cadenas de texto o carga masiva de archivos planos estructurados en formatos CSV y hojas de cálculo Excel.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tc2:
            st.markdown(
                """
                <div style='background-color: #262730; padding: 1.8rem; border-radius: 10px; border-left: 5px solid #00C49F; height: 100%;'>
                    <h4 style='margin-top:0;'>🧮 Motores Matemáticos</h4>
                    <p style='font-size: 0.9rem; color: #A3A8B4;'>Mapeo dinámico del espacio vectorial utilizando modelos formales de Bag of Words, Vectorización Binaria, Relevancia Estadística TF-IDF y análisis de N-gramas.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tc3:
            st.markdown(
                """
                <div style='background-color: #262730; padding: 1.8rem; border-radius: 10px; border-left: 5px solid #0077B6; height: 100%;'>
                    <h4 style='margin-top:0;'>📈 Visualización Interactiva</h4>
                    <p style='font-size: 0.9rem; color: #A3A8B4;'>Dashboard descriptivo inmediato con diagnóstico exploratorio de longitud de textos, histogramas de distribución y descarga optimizada del nuevo dataset.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Botón de ingreso central y estilizado de gran tamaño
        # Botón de ingreso central y estilizado de gran tamaño
        b1, b2, b3 = st.columns([2, 2, 2])
        with b2:
            # 🔥 CORRECCIÓN DE SCROLL NATIIVA: Al presionar el botón, cambiamos el estado
            # e inmediatamente llamamos a st.rerun() antes de pintar cualquier componente.
            # Esto destruye la posición previa del scroll del navegador.
            if st.button(
                "🚀 INICIAR ANALÍTICA DE TEXTO",
                use_container_width=True,
                type="primary",
            ):
                st.session_state["ingresar"] = True
                st.rerun()

        # Créditos institucionales al pie de la landing page
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='text-align: center; border-top: 1px solid #4A4A4A; padding-top: 1.5rem;'>
                <p style='font-size: 0.85rem; color: #7F8C8D;'>Examen Parcial • Curso de Minería de Datos • Mayo 2026</p>
                <p style='font-size: 0.9rem; font-weight: bold; color: #BDC3C7;'>Desarrollado por: Jeferson Coronado Cortez & Silva Burga Bryan</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# ⚙️ ENTORNO DEL APLICATIVO PRINCIPAL (TABS)
# =========================================================
else:
    # Sidebar - Control de versión y Metadatos
    st.sidebar.caption("📌 Versión de la Aplicación: v1.4.0 (Dashboard Avanzado)")
    st.sidebar.markdown("**Desarrolladores:**")
    st.sidebar.markdown("- Jeferson Coronado Cortez\n- Silva Burga Bryan")
    st.sidebar.markdown("---")

    # Botón en el Sidebar para permitir regresar a la pantalla de bienvenida si se desea
    if st.sidebar.button("🏠 Volver al Inicio", use_container_width=True):
        st.session_state["ingresar"] = False
        st.rerun()

    tab_app, tab_manual, tab_informe = st.tabs(
        [
            "🚀 1. Aplicación Ejecutable",
            "📖 2. Manual de Usuario",
            "📄 3. Informe de la Aplicación",
        ]
    )

    # ------------------------------------------
    # PESTAÑA 1: LA APLICACIÓN INTERACTIVA
    # ------------------------------------------
    with tab_app:
        st.title("📊 Sistema de Vectorización y Visualización de Texto")
        # 🔥 ANCLA DE SCROLL: Fuerza al navegador a subir al inicio al renderizar esta pestaña
        st.components.v1.html(
            "<script>window.parent.document.querySelector('section.main').scrollTo(0,0);</script>",
            height=0,
        )
        

        st.text_input(
            "Foco_Scroll",
            label_visibility="collapsed",
            key="foco_scroll_top",
            disabled=True,
        )

        st.markdown(
            "Carga tus datos, selecciona la técnica matemática y exporta tu nuevo dataset estructurado."
        )

        st.subheader("📥 1. Ingreso y Selección de Datos")
        origen_datos = st.radio(
            "Selecciona el método de entrada:",
            ["Texto Manual (Escribir líneas)", "Cargar Archivo Local (CSV/Excel)"],
            horizontal=True,
        )

        textos_a_procesar = []

        if origen_datos == "Texto Manual (Escribir líneas)":
            texto_libre = st.text_area(
                "Escribe tus oraciones de prueba (una frase por línea):",
                "Minería de datos minería de datos permite analizar texto.\nLa minería de datos transforma texto en información.\nTF-IDF resalta términos relevantes del corpus.\nLos n-gramas capturan minería de datos como contexto.",
            )
            textos_a_procesar = [
                linea.strip() for linea in texto_libre.split("\n") if linea.strip()
            ]

        else:
            # Carga de archivos con optimización de renderizado
            archivo_cargado = st.file_uploader(
                "Sube tu dataset real (Formatos aceptados: .csv, .xlsx)",
                type=["csv", "xlsx"],
            )

            if archivo_cargado is not None:
                # DETECCIÓN DE CAMBIO DE ARCHIVO: Si el nombre cambia, destruimos la memoria vieja inmediatamente
                if (
                    "nombre_archivo" in st.session_state
                    and st.session_state["nombre_archivo"] != archivo_cargado.name
                ):
                    if "df_actual" in st.session_state:
                        del st.session_state["df_actual"]

                # Almacenar en caché del estado de Streamlit para no re-leer el archivo en cada clic
                if "df_actual" not in st.session_state:
                    st.session_state["df_actual"] = cargar_archivo(archivo_cargado)
                    st.session_state["nombre_archivo"] = archivo_cargado.name

                df = st.session_state["df_actual"]

                if df is not None:
                    # Layout de dos columnas: Izquierda para elegir, Derecha para previsualizar el archivo original
                    col_sel, col_prev = st.columns([1, 2])

                    with col_sel:
                        st.markdown("##### 🔍 Selección de Target")
                        # Forzamos un key único dinámico basado en el nombre del archivo para resetear el selector
                        columna_texto = st.selectbox(
                            "Selecciona la columna con el texto que deseas vectorizar:",
                            options=df.columns,
                            key=f"col_{archivo_cargado.name}",
                        )
                        textos_a_procesar = limpiar_y_extraer_textos(df, columna_texto)

                    with col_prev:
                        st.markdown(
                            "##### 📄 Vista Previa del Archivo Original (Primeras 3 filas)"
                        )
                        st.dataframe(df.head(3), use_container_width=True)

                    if textos_a_procesar:
                        st.success(
                            f"✅ Filtro aplicado: {len(textos_a_procesar)} filas válidas listas para procesamiento matemático."
                        )

                        # =========================================================
                        # 📈 MÓDULO: ANALÍTICA EXPLORATORIA (EDA DASHBOARD)
                        # =========================================================
                        with st.expander(
                            "📊 Ver Dashboard Estadístico del Dataset (EDA)",
                            expanded=False,
                        ):
                            st.markdown("### 🔬 Diagnóstico Exploratorio de los Textos")

                            # Conversión rápida a serie para métricas de longitud
                            series_textos = pd.Series(textos_a_procesar)
                            longitudes_caracteres = series_textos.str.len()
                            conteo_palabras = series_textos.str.split().str.len()

                            # Kpis en tarjetas
                            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                            kpi1.metric("Filas Totales", f"{len(df):,}")
                            kpi2.metric(
                                "Filas Procesables", f"{len(textos_a_procesar):,}"
                            )
                            kpi3.metric(
                                "Promedio de Palabras",
                                f"{int(conteo_palabras.mean())} palabras",
                            )
                            kpi4.metric(
                                "Máx. Caracteres", f"{longitudes_caracteres.max():,}"
                            )

                            # Gráficas analíticas descriptivas
                            g1, g2 = st.columns(2)
                            with g1:
                                st.markdown(
                                    "**Distribución del tamaño de los textos (Caracteres)**"
                                )
                                # Agrupar las longitudes en rangos para usar el gráfico nativo estable
                                counts, bins = pd.cut(
                                    longitudes_caracteres, bins=10, retbins=True
                                )
                                df_hist_1 = (
                                    counts.value_counts().sort_index().to_frame()
                                )
                                df_hist_1.index = df_hist_1.index.astype(str)
                                st.bar_chart(df_hist_1)

                            with g2:
                                st.markdown(
                                    "**Distribución del número de palabras por fila**"
                                )
                                counts_p, bins_p = pd.cut(
                                    conteo_palabras, bins=10, retbins=True
                                )
                                df_hist_2 = (
                                    counts_p.value_counts().sort_index().to_frame()
                                )
                                df_hist_2.index = df_hist_2.index.astype(str)
                                st.bar_chart(df_hist_2)
                    else:
                        st.error(
                            "❌ La columna seleccionada no contiene texto válido o solo contiene celdas vacías."
                        )

        # --- APLICACIÓN DEL MODELO MATEMÁTICO ---
        if len(textos_a_procesar) > 0:
            st.markdown("---")
            st.subheader("⚙️ 2. Configuración y Aplicación del Modelo")
            st.info(
                "🔎 **Preprocesamiento automático:** Los textos se convierten a minúsculas, se remueve la puntuación, "
                "se descartan nulos y se extraen palabras (tokens) alfanuméricas de 2 o más caracteres."
            )

            tipo_vectorizacion = st.selectbox(
                "Selecciona el algoritmo de vectorización matemática a aplicar:",
                [
                    "Count Vectorizer (Bag of Words)",
                    "Vectorización Binaria (Presencia/Ausencia)",
                    "TF-IDF (Term Frequency - Inverse Document Frequency)",
                    "N-gramas",
                ],
            )

            st.markdown("#### 🧮 Ecuación del Modelo Seleccionado")

            # PODA DE VOCABULARIO CRÍTICA: Limitamos a las 1500 palabras más importantes para proteger la memoria RAM
            MAX_FEATURES = 1500

            # Inyección dinámica de hiperparámetros y fórmulas
            if tipo_vectorizacion == "Count Vectorizer (Bag of Words)":
                st.info(
                    "💡 **Count Vectorizer:** Construye un espacio vectorial basado únicamente en la frecuencia absoluta de aparición de cada token."
                )
                st.latex(r"Vector\_D_i = [f(w_1), f(w_2), \dots, f(w_n)]")
                st.caption(
                    "Donde $f(w_j)$ representa la frecuencia o conteo del término $j$ dentro del documento $i$."
                )
                vectorizador = CountVectorizer(
                    lowercase=True, max_features=MAX_FEATURES
                )

            elif tipo_vectorizacion == "Vectorización Binaria (Presencia/Ausencia)":
                st.info(
                    "💡 **Vectorización Binaria:** Indica únicamente si cada término aparece en el documento, sin importar su frecuencia de repetición."
                )
                st.latex(
                    r"x_{ij} = \begin{cases} 1, & \text{si el término } j \text{ aparece en el documento } i \\ 0, & \text{si no aparece} \end{cases}"
                )
                st.caption(
                    "Cada columna representa un término; el valor mapea presencia (1) o ausencia (0)."
                )
                vectorizador = CountVectorizer(
                    lowercase=True, binary=True, max_features=MAX_FEATURES
                )

            elif tipo_vectorizacion == "N-gramas":
                st.info(
                    "💡 **N-gramas:** Representan secuencias consecutivas de palabras para capturar contexto semántico."
                )
                opcion_ngrama = st.selectbox(
                    "Selecciona el rango de N-gramas a generar:",
                    list(OPCIONES_NGRAMAS.keys()),
                )
                rango_ngrama = OPCIONES_NGRAMAS[opcion_ngrama]

                st.latex(r"x_{ij} = f(g_j, d_i)")
                st.caption(
                    "Donde $g_j$ es el n-grama $j$, $d_i$ es el documento $i$ y $f$ es la frecuencia de la secuencia."
                )
                vectorizador = CountVectorizer(
                    lowercase=True, ngram_range=rango_ngrama, max_features=MAX_FEATURES
                )

            else:
                st.info(
                    "💡 **TF-IDF:** Pondera la importancia de un término evaluando su frecuencia local contra su rareza global en el corpus."
                )
                st.latex(r"tf(t,d) = f(t,d)")
                st.latex(r"idf(t) = \log\left(\frac{1 + N}{1 + df(t)}\right) + 1")
                st.latex(
                    r"\widetilde{x}_{t,d} = tf(t,d) \times idf(t) \quad \longrightarrow \quad x_d = \frac{\widetilde{x}_d}{\sqrt{\sum_j \widetilde{x}_{j,d}^{\,2}}}"
                )
                st.caption(
                    "Aplica la variante exacta de scikit-learn con suavizado de IDF y normalización euclidiana L2."
                )
                vectorizador = TfidfVectorizer(
                    lowercase=True, max_features=MAX_FEATURES
                )

            # --- VISUALIZACIÓN DE SALIDAS ---
            st.markdown("---")
            st.subheader("🖥️ 3. Visualización de Salidas y Descarga")

            with st.spinner(
                "Ejecutando transformaciones matriciales en el espacio vectorial..."
            ):
                matriz_num, vocabulario = vectorizar_textos(
                    textos_a_procesar, vectorizador
                )

            if matriz_num is None:
                st.error(
                    "❌ Error de Vocabulario: Los textos procesados no contienen suficientes caracteres válidos para construir una matriz."
                )
            else:
                # 🔥 SOLUCIÓN DEFINITIVA: Mantenemos el DataFrame en formato disperso (Sparse)
                # Eliminamos por completo .to_dense() para que NO consuma esos 3GB de RAM independientes
                with st.spinner(
                    "Estructurando matriz en bloques de memoria optimizados..."
                ):
                    df_resultado = pd.DataFrame.sparse.from_spmatrix(
                        matriz_num, columns=vocabulario
                    )

                # Ajustamos dinámicamente la longitud de los textos por si el tokenizador descartó filas vacías
                textos_ajustados = textos_a_procesar[: len(df_resultado)]
                df_resultado.insert(0, "Texto_Original", textos_ajustados)

                sub_tab1, sub_tab2 = st.tabs(
                    ["📋 Nuevo Dataset Vectorizado", "📈 Análisis de Frecuencias"]
                )

                with sub_tab1:
                    st.markdown(
                        "##### Vista Previa de la Matriz Generada (Documento $\\times$ Término)"
                    )
                    # 🔥 SOLUCIÓN AL TYPEERROR:
                    # Convertimos a denso ÚNICAMENTE el bloque de las 500 filas para que Streamlit pueda pintarlo
                    # El resto del DataFrame principal sigue comprimido en memoria resguardando la RAM.
                    df_vista_previa = df_resultado.head(500).copy()
                    for col in df_vista_previa.columns:
                        if hasattr(df_vista_previa[col], "sparse"):
                            df_vista_previa[col] = df_vista_previa[
                                col
                            ].sparse.to_dense()

                    st.dataframe(df_vista_previa, use_container_width=True)
                    st.caption(
                        f"💡 Nota: Por rendimiento de la interfaz web, se visualizan las primeras 500 filas de las {len(df_resultado)} procesadas."
                    )

                    # 📥 OPTIMIZACIÓN DE DESCARGA: Convertimos a CSV de forma directa y segura
                    @st.cache_data
                    def optimizar_conversion_csv(df_datos):
                        return df_datos.to_csv(index=False).encode("utf-8")

                    with st.spinner("Preparando archivo de descarga masiva..."):
                        csv_bytes = optimizar_conversion_csv(df_resultado)

                    st.download_button(
                        label="💾 Guardar y Descargar Nuevo Dataset (CSV)",
                        data=csv_bytes,
                        file_name="nuevo_dataset_vectorizado.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with sub_tab2:
                    st.markdown("##### Distribución de Peso / Frecuencia en el Corpus")

                    # 🔥 SOLUCIÓN AL SEGUNDO TYPEERROR:
                    # Eliminamos la columna de texto original, sumamos los pesos, extraemos el top 20
                    # y convertimos la serie final a flotantes nativos densos usando .astype(float)
                    pesos_acumulados = (
                        df_resultado.drop(columns=["Texto_Original"])
                        .sum()
                        .sort_values(ascending=False)
                        .head(20)
                    )

                    if not pesos_acumulados.empty:
                        # Convertimos explícitamente a denso para que la librería gráfica no explote
                        pesos_acumulados_densos = pesos_acumulados.astype(float)
                        st.bar_chart(pesos_acumulados_densos)
                    else:
                        st.warning(
                            "Sin datos de peso estadístico suficientes para proyectar gráficas."
                        )
        else:
            st.warning(
                "⚠️ Esperando el ingreso de datos válidos para inicializar los motores matemáticos."
            )

    # ------------------------------------------
    # PESTAÑA 2: MANUAL DE USUARIO
    # ------------------------------------------
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
            * **Estado:** Aplicación de nivel profesional optimizada para Kaggle
            """)
        with col2:
            st.subheader("📌 Grabado de Versión")
            st.info(
                "**Versión Actual:** v1.4.0  \n**Última Modificación:** Mayo 2026  \n**Cambios:** Se agregó una Landing Page interactiva, panel de Diagnóstico Exploratorio (EDA), KPIs estadísticos en tiempo real, histogramas de distribución y control robusto de fallas de parseo."
            )

        st.markdown("### 🚀 Procedimiento de Puesta en Marcha")
        st.code(
            "pip install streamlit scikit-learn pandas openpyxl\nstreamlit run src/app.py --server.maxUploadSize 1024",
            language="bash",
        )

        st.markdown("""
        ### 🛑 Restricciones del Sistema
        1. **Formatos:** Soporta estrictamente archivos estructurados `.csv` y `.xlsx`.
        2. **Filtro Automático:** Ignora de manera automática cadenas vacías y respuestas nulas del tipo 'No Positive' o 'No Negative'.
        3. **Poda de Dimensionalidad:** Limita el espacio vectorial automáticamente a las 1,500 características de mayor peso formal para resguardar la estabilidad de la memoria RAM.
        
        ### 🎯 Guía Operativa con Carga Masiva
        * **Paso 1:** Suba el archivo en la sección 1. Automáticamente se desplegará una **Vista Previa** de la tabla original a la derecha. Use el selector desplegable de la izquierda para indicarle al sistema cuál columna contiene el texto libre que quiere analizar. Puedes expandir el panel **EDA** para ver estadísticas de distribución del texto.
        * **Paso 2:** Seleccione la técnica matemática. Verifique la ecuación en KaTeX.
        * **Paso 3:** Descargue el nuevo dataset estructurado listo para minería de datos secundaria.
        """)

    # ------------------------------------------
    # PESTAÑA 3: INFORME DE LA APLICACIÓN
    # ------------------------------------------
    with tab_informe:
        st.header("📄 Informe Técnico de la Aplicación")
        st.markdown("---")

        st.subheader("🏗️ Esquema y Arquitectura del Sistema")
        st.markdown("""
        La arquitectura implementa un desacoplamiento lógico basado en el patrón **Capa de Datos - Capa de Presentación**:
        
        1. **Capa de Entrada y Caché (I/O):** Lee los buffers binarios de los archivos subidos. Utiliza la persistencia `st.session_state` con destrucción activa de estados cruzados para permitir la alternancia fluida entre diferentes archivos en disco de forma dinámica.
        2. **Capa de Diagnóstico Analítico (EDA):** Extrae de manera nativa la distribución estadística del corpus (conteo de palabras y caracteres) para proporcionar un contexto del volumen informacional antes de inicializar la fase matricial.
        3. **Capa de Limpieza Integrada:** Un bloque de normalización descarta valores atípicos, nulos y strings de control de Kaggle de manera automatizada antes de pasarlos a la matriz.
        4. **Capa de Cómputo Vectorial:** Invoca motores de cálculo matricial paralelos limitados a 1,500 componentes principales para evitar el desbordamiento de memoria por dimensionalidad densa.
        5. **Capa de Almacenamiento Estructurado:** Transforma objetos dispersos (*Scipy Sparse Matrix*) a DataFrames bidimensionales de `pandas` acoplando longitudes de vectores de forma segura.
        """)
