# ADSDB-Project
Multi-Modal Data Management process for the MDS-ADSDB Course.

# How to run the pipeline
1. Start docker (in Linux is `sudo systemctl start docker`, in Windows just open Docker Desktop)
2. Go to the root folder (`ADSDB-Project/`) and launch the MinIO client configuration using the `docker-compose.yaml` file: `sudo docker compose up -d`
3. Open the notebook `orchestrator.ipynb`, go to the Module Selection section and follow the instructions to execute the different modules.

# Requirements
- Install Python libraries: `pip install -r requirements.txt`
- Install Docker (or Docker Desktop for Windows)

# Folder structure
In the `queries` folder we can find different images and videos used for the queries made in the multi-modal tasks.
In the `src` we can find all the project code organised in folders:
- `common`: Contains common code used in several other scripts.
- `data_management`: Contains all the scripts used to implement all the zones, from data ingestion to exploitation zone.
- `multi_modal_tasks`: Contains the three multi-modal tasks (`same_modality_task.py`, `multi_modality_task.py` and `generative_task.py`)

The `orchestrator.ipynb` is also inside `src` folder.

