
import streamlit as st
import subprocess
import os
from PIL import Image
import pandas as pd

st.set_page_config(layout="wide")

st.title("Análisis Predictivo de Cáncer de Mama")
st.write("""
### Una aplicación interactiva para visualizar los resultados del análisis de clustering.
Esta herramienta permite ejecutar el pipeline de análisis de datos y explorar los perfiles de pacientes y las visualizaciones generadas.
""")

# Directorio de reportes
REPORTS_DIR = "reports/figures"

if st.button("Ejecutar Análisis de Datos y Modelado"):
    st.write("Ejecutando el pipeline... Esto puede tardar unos momentos.")
    
    # Usar subprocess para correr el main.py
    process = subprocess.run(["python", "main.py"], capture_output=True, text=True)
    
    st.subheader("Log de la Ejecución del Pipeline")
    st.text_area("Salida", process.stdout + process.stderr, height=300)
    
    if process.returncode == 0:
        st.success("¡Pipeline ejecutado exitosamente!")
    else:
        st.error("Ocurrió un error durante la ejecución del pipeline.")


st.header("Resultados del Análisis de Clustering")

# Comprobar si los resultados existen para mostrarlos
if os.path.exists(REPORTS_DIR) and len(os.listdir(REPORTS_DIR)) > 0:
    st.write("A continuación se muestran los gráficos generados a partir del último análisis ejecutado.")

    # Cargar y mostrar perfiles de cluster (si se guardan como CSV)
    # Por ahora, mostraremos las imágenes generadas.

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualización de Clusters (PCA)")
        if os.path.exists(f"{REPORTS_DIR}/pca_clusters.png"):
            image = Image.open(f"{REPORTS_DIR}/pca_clusters.png")
            st.image(image, caption='Clusters de pacientes en 2D (PCA)', use_column_width=True)
        else:
            st.warning("Gráfico de PCA no encontrado.")

    with col2:
        st.subheader("Distribución de Supervivencia")
        if os.path.exists(f"{REPORTS_DIR}/survivalmonths_distribution.png"):
            image = Image.open(f"{REPORTS_DIR}/survivalmonths_distribution.png")
            st.image(image, caption='Meses de Supervivencia por Cluster', use_column_width=True)
        else:
            st.warning("Gráfico de supervivencia no encontrado.")

    st.subheader("Distribución de Tipos de Cáncer y Respuesta al Tratamiento")
    col3, col4 = st.columns(2)
    
    with col3:
        if os.path.exists(f"{REPORTS_DIR}/cancertype_distribution_stacked.png"):
            image = Image.open(f"{REPORTS_DIR}/cancertype_distribution_stacked.png")
            st.image(image, caption='Tipos de Cáncer por Cluster (%)', use_column_width=True)
        else:
            st.warning("Gráfico de tipo de cáncer no encontrado.")
            
    with col4:
        if os.path.exists(f"{REPORTS_DIR}/treatmentresponse_distribution_stacked.png"):
            image = Image.open(f"{REPORTS_DIR}/treatmentresponse_distribution_stacked.png")
            st.image(image, caption='Respuesta al Tratamiento por Cluster (%)', use_column_width=True)
        else:
            st.warning("Gráfico de respuesta al tratamiento no encontrado.")

else:
    st.info("Aún no se ha ejecutado el análisis. Haz clic en el botón de arriba para generar los resultados.")

