# ADSDB-Project
Multi-Modal Data Management process for the MDS-ADSDB Course.

# How to run the pipeline
1. Start docker (in Linux is `sudo systemctl start docker`, in Windows just open Docker Desktop)
2. Go to the root folder (`ADSDB-Project/`) and launch the MinIO client configuration using the `docker-compose.yaml` file: `sudo docker compose up -d`
3. Open the notebook `orchestrator.ipynb`, go to the Module Selection section and follow the instructions to execute the different modules.

# Requirements
- Install Python libraries: `pip install -r requirements.txt`
- Install Docker (or Docker Desktop for Windows)

