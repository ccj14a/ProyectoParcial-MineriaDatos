import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st

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


def limpiar_texto_linea(linea):
    """Normaliza una cadena de texto individual eliminando ruidos comunes."""
    linea_str = str(linea).strip()
    if linea_str.lower() in ["no negative", "no positive", "nan", "none", ""]:
        return ""
    return linea_str


def limpiar_y_extraer_textos(df, columna=None, procesar_todo=False):
    """Extrae textos, elimina nulos y filtra valores por defecto de datasets reales.

    Permite procesar una columna o consolidar todo el dataset de forma
    horizontal.
    """
    if procesar_todo:
        # Seleccionar solo columnas de tipo texto/objeto para no meter ruido de IDs o métricas puramente numéricas
        columnas_texto = df.select_dtypes(include=["object", "string"]).columns.tolist()

        if not columnas_texto:
            return []

        textos_limpios = []
        for _, fila in df[columnas_texto].iterrows():
            # Combinar todas las columnas de texto de la fila actual separadas por espacio
            componentes = [limpiar_texto_linea(fila[col]) for col in columnas_texto]
            frase_consolidada = " ".join(
                [c for c in componentes if c]
            )  # Filtrar vacíos
            textos_limpios.append(frase_consolidada)

        return textos_limpios
    else:
        if columna not in df.columns:
            return []
        raw_textos = df[columna].fillna("").astype(str).tolist()
        return [limpiar_texto_linea(t) for t in raw_textos if limpiar_texto_linea(t)]


def vectorizer_textos(textos, vectorizador):
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
    st.markdown("<br><br>", unsafe_allow_html=True)

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

        b1, b2, b3 = st.columns([2, 2, 2])
        with b2:
            if st.button(
                "🚀 INICIAR ANALÍTICA DE TEXTO",
                use_container_width=True,
                type="primary",
            ):
                st.session_state["ingresar"] = True
                st.rerun()

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
    st.sidebar.caption("📌 Versión de la Aplicación: v1.5.0 (Global Vectorizer)")
    st.sidebar.markdown("**Desarrolladores:**")
    st.sidebar.markdown("- Jeferson Coronado Cortez\n- Silva Burga Bryan")
    st.sidebar.markdown("---")

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

    with tab_app:
        st.title("📊 Sistema de Vectorización y Visualización de Texto")

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
                # 🔄 DETECCIÓN DE CAMBIO REAL: Verificamos el nombre O si el contenido interno cambió
                # Guardamos el hash o valor del archivo para saber si es un archivo editado
                contenido_hash = hash(archivo_cargado.getvalue())
                
                cambio_nombre = "nombre_archivo" in st.session_state and st.session_state["nombre_archivo"] != archivo_cargado.name
                cambio_contenido = "hash_archivo" in st.session_state and st.session_state["hash_archivo"] != contenido_hash

                if cambio_nombre or cambio_contenido:
                    # Si algo cambió, destruimos inmediatamente la memoria vieja para forzar la re-lectura
                    if "df_actual" in st.session_state:
                        del st.session_state["df_actual"]

                # Almacenar en caché el nuevo estado si no existe
                if "df_actual" not in st.session_state:
                    st.session_state["df_actual"] = cargar_archivo(archivo_cargado)
                    st.session_state["nombre_archivo"] = archivo_cargado.name
                    st.session_state["hash_archivo"] = contenido_hash # Guardamos el identificador único del contenido

                df = st.session_state["df_actual"]

                if df is not None:
                    col_sel, col_prev = st.columns([1, 2])

                    with col_sel:
                        st.markdown("##### 🔍 Enfoque de Procesamiento")
                        enfoque = st.radio(
                            "¿Cómo deseas aplicar la vectorización?",
                            [
                                "Columna Única (Target Específico)",
                                "Todo el Dataset (Combinar columnas textuales)",
                            ],
                            key=f"enfoque_{archivo_cargado.name}",
                        )

                        if enfoque == "Columna Única (Target Específico)":
                            columna_texto = st.selectbox(
                                "Selecciona la columna target:",
                                options=df.columns,
                                key=f"col_{archivo_cargado.name}",
                            )
                            textos_a_procesar = limpiar_y_extraer_textos(
                                df, columna=columna_texto, procesar_todo=False
                            )
                        else:
                            st.caption(
                                "ℹ️ El sistema consolidará automáticamente todas las columnas categóricas y de texto en un solo corpus por fila."
                            )
                            textos_a_procesar = limpiar_y_extraer_textos(
                                df, procesar_todo=True
                            )

                    with col_prev:
                        st.markdown(
                            "##### 📄 Vista Previa del Archivo Original (Primeras 3 filas)"
                        )
                        st.dataframe(df.head(3), use_container_width=True)

                    if textos_a_procesar and any(t.strip() for t in textos_a_procesar):
                        st.success(
                            f"✅ Filtro aplicado: {len(textos_a_procesar)} filas listas para procesamiento matemático."
                        )

                        with st.expander(
                            "📊 Ver Dashboard Estadístico del Dataset (EDA)",
                            expanded=False,
                        ):
                            st.markdown("### 🔬 Diagnóstico Exploratorio de los Textos")

                            series_textos = pd.Series(textos_a_procesar)
                            longitudes_caracteres = series_textos.str.len()
                            conteo_palabras = series_textos.str.split().str.len()

                            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                            kpi1.metric("Filas Totales", f"{len(df):,}")
                            kpi2.metric(
                                "Filas Procesables",
                                f"{len(textos_a_procesar):,}",
                            )
                            kpi3.metric(
                                "Promedio de Palabras",
                                f"{int(conteo_palabras.mean())} palabras",
                            )
                            kpi4.metric(
                                "Máx. Caracteres",
                                f"{longitudes_caracteres.max():,}",
                            )

                            g1, g2 = st.columns(2)
                            with g1:
                                st.markdown(
                                    "**Distribución del tamaño de los textos (Caracteres)**"
                                )
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
                            "❌ La selección actual no contiene texto válido o el dataset no posee columnas de tipo cadena."
                        )

        # --- APLICACIÓN DEL MODELO MATEMÁTICO ---
        if len(textos_a_procesar) > 0 and any(t.strip() for t in textos_a_procesar):
            st.markdown("---")
            st.subheader("⚙️ 2. Configuración y Aplicación del Modelo")
            st.info(
                "🔎 **Preprocesamiento automático:** Los textos se convierten a minúsculas, se remueve la puntuación, "
                "se descartan nulos globales y se extraen tokens alfanuméricos de 2 o más caracteres."
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
            MAX_FEATURES = 1500

            if tipo_vectorizacion == "Count Vectorizer (Bag of Words)":
                st.info(
                    "💡 **Count Vectorizer:** Construye un espacio vectorial basado únicamente en la frecuencia absoluta de aparición de cada token."
                )
                st.latex(r"Vector\_D_i = [f(w_1), f(w_2), \dots, f(w_n)]")
                st.caption(
                    "Donde $f(w_j)$ representa la frecuencia del término $j$ dentro del documento $i$."
                )
                vectorizador = CountVectorizer(
                    lowercase=True, max_features=MAX_FEATURES
                )

            elif tipo_vectorizacion == "Vectorización Binaria (Presencia/Ausencia)":
                st.info(
                    "💡 **Vectorización Binaria:** Indica únicamente si cada término aparece en el documento, sin importar su frecuencia."
                )
                st.latex(
                    r"x_{ij} = \begin{cases} 1, & \text{si el término } j \text{ aparece en el documento } i \\ 0, & \text{si no aparece} \end{cases}"
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
                vectorizador = CountVectorizer(
                    lowercase=True,
                    ngram_range=rango_ngrama,
                    max_features=MAX_FEATURES,
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
                vectorizador = TfidfVectorizer(
                    lowercase=True, max_features=MAX_FEATURES
                )

            # --- VISUALIZACIÓN DE SALIDAS ---
            st.markdown("---")
            st.subheader("🖥️ 3. Visualización de Salidas y Descarga")

            with st.spinner(
                "Ejecutando transformaciones matriciales en el espacio vectorial..."
            ):
                matriz_num, vocabulario = vectorizer_textos(
                    textos_a_procesar, vectorizador
                )

            if matriz_num is None:
                st.error(
                    "❌ Error de Vocabulario: Los textos procesados no contienen suficientes tokens válidos."
                )
            else:
                with st.spinner(
                    "Estructurando matriz en bloques de memoria optimizados..."
                ):
                    df_resultado = pd.DataFrame.sparse.from_spmatrix(
                        matriz_num, columns=vocabulario
                    )

                textos_ajustados = textos_a_procesar[: len(df_resultado)]
                df_resultado.insert(0, "Texto_Consolidado_O_Target", textos_ajustados)

                sub_tab1, sub_tab2 = st.tabs(
                    ["📋 Nuevo Dataset Vectorizado", "📈 Análisis de Frecuencias"]
                )

                with sub_tab1:
                    st.markdown(
                        "##### Vista Previa de la Matriz Generada (Documento $\\times$ Término)"
                    )
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

                    @st.cache_data
                    def optimizar_conversion_csv(df_datos):
                        return df_datos.to_csv(index=False).encode("utf-8")

                    with st.spinner("Preparando archivo de descarga masiva..."):
                        csv_bytes = optimizar_conversion_csv(df_resultado)

                    st.download_button(
                        label="💾 Guardar y Descargar Nuevo Dataset (CSV)",
                        data=csv_bytes,
                        file_name="dataset_vectorizado_completo.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with sub_tab2:
                    st.markdown("##### Distribución de Peso / Frecuencia en el Corpus")
                    pesos_acumulados = (
                        df_resultado.drop(columns=["Texto_Consolidado_O_Target"])
                        .sum()
                        .sort_values(ascending=False)
                        .head(20)
                    )

                    if not pesos_acumulados.empty:
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
                "**Versión Actual:** v1.5.0  \n**Última Modificación:** Mayo 2026  \n**Cambios:** Se agregó la funcionalidad de procesamiento global multicolumna (Todo el Dataset), optimización de merges textuales horizontales y persistencia matricial en formato Sparse."
            )

    # ------------------------------------------
    # PESTAÑA 3: INFORME DE LA APLICACIÓN
    # ------------------------------------------
    with tab_informe:
        st.header("📄 Informe Técnico de la Aplicación")
        st.markdown("---")
        st.subheader("🏗️ Esquema y Arquitectura del Sistema")
        st.markdown("""
        La arquitectura implementa un desacoplamiento lógico basado en el patrón **Capa de Datos - Capa de Presentación**:
        
        1. **Capa de Entrada y Caché (I/O):** Permite aislar la lectura del archivo e inyectar el método de combinación lineal horizontal de strings si se desea procesar el dataset entero.
        2. **Capa de Diagnóstico Analítico (EDA):** Extrae la distribución estadística estructural de la recopilación consolidada o de la columna objetivo.
        3. **Capa de Cómputo Vectorial:** Aplica transformaciones de minería de texto usando la representación optimizada de Scipy para transferir dataframes sin sobrecargar la RAM del servidor.
        """)
