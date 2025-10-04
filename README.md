# ADSDB-Project
Multi-Modal Data Management process for MDS-ADSDB Course.

# Cómo ejecutar
1. Poner en marcha el Docker (en mi caso es sudo systemctl start docker)
2. Poner en marcha el cliente de MinIO a través de Docker: docker compose up -d
3. Desde la carpeta raíz, ejecutar el manager de MinIO: python -m src.common.minio_manager.py. De momento, esto crea la estructura de buckets de la landing zone.
4. Ejecutar los ingestions de la misma manera: python -m src.data_management.data_ingestion.{X}.py. De momento, solo está la de texto (json).



# Requisitos
- Instalar librerias de Python con pip install -r requirements.txt
- Instalar Docker (y docker-compose si no viene ya instalado)

# API Key de Kaggle
- Descargar la Api Key de Kaggle y guardarla en ~.kaggle/kaggle.json. La API Key se obtiene en los ajustes de la cuenta en la página web.
- Darle permisos a la Api Key de Kaggle con chmod 600 ~/.kaggle/kaggle.json

# API Key de Pexels
- Para obtener la API Key se accede a https://www.pexels.com/api.
- Una vez obtenida, en Windows, se ha de añadir desde Variables de Entorno del Sistema. En Linux, se puede editar el ~/.bashrc o ~/.zshrc y 
añadir la línea export PEXELS_API_KEY={api_key}.
