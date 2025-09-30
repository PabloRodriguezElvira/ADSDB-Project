# ADSDB-Project
Multi-Modal Data Management process for MDS-ADSDB Course.

# Cómo ejecutar
1. Poner en marcha el cliente de MinIO a través de Docker: docker compose up -d
2. Desde la carpeta raíz, ejecutar el manager de MinIO: python -m src.common.minio_manager.py
3. Ejecutar los ingestions de la misma manera: python -m src.data_management.data_ingestion.{X}.py


# Requisitos
- Instalar librerias de Python con pip install -r requirements
- Instalar Docker (y docker-compose si no viene ya instalado)
- Descargar la Api Key de Kaggle y guardarla en ~.kaggle/kaggle.json
- Darle permisos a la Api Key de Kaggle con chmod 600 ~/.kaggle/kaggle.json
