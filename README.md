# Air Quality Prediction

This repository contains information about Air Quality Prediction using Machine Learning approach.

## Dataset Information

- Dataset downloaded from: [here](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021/data)
- Dataset description: \
This dataset contains the Air Quality Index (AQI) or Indeks Standar Pencemaran Udara (ISPU) measured from 5 air quality monitoring stations (SPKU) in DKI Jakarta from January 2021 to December 2021.

- Data Definition:

| **Variable** | **Type** | **Description**                                                                                                                                                               |
|--------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tanggal`    | string   | The date when the AQI measurement was recorded.                                                                                                                               |
| `stasiun`    | string   | The name or identifier of the monitoring station where the measurement was taken.                                                                                             |
| `pm25`       | integer  | The concentration of particulate matter with a diameter of 2.5 micrometers or less (PM2.5), measured in micrograms per cubic meter (µg/m³).                                   |
| `pm10`       | integer  | The concentration of particulate matter with a diameter of 10 micrometers or less (PM10), measured in micrograms per cubic meter (µg/m³).                                     |
| `so2`        | integer  | The concentration of sulfur dioxide (SO2), measured in parts per million (ppm).                                                                                               |
| `co`         | integer  | The concentration of carbon monoxide (CO), measured in parts per million (ppm).                                                                                               |
| `o3`         | integer  | The concentration of ozone (O3), measured in parts per million (ppm).                                                                                                         |
| `no2`        | integer  | The concentration of nitrogen dioxide (NO2), measured in parts per million (ppm).                                                                                             |
| `max`        | integer  | The maximum value recorded among the pollutants for that particular date and station. This value represents the highest concentration among PM25, PM10, SO2, CO, O3, and NO2. |
| `critical`   | string   | The pollutant that had the highest concentration for that date and station.                                                                                                   |
| `category`   | string   | The air quality category based on the 'max' value that describes the air quality level.                                                                                       |

## Predict API (FastAPI)
### Endpoint

`POST` `/predict`

### Description

This API endpoint accepts pollutant values as input and returns the predicted air quality. 

### Request

**Not all variables used in the prediction process**. Below are the example.

```json
{
  "stasiun": "DKI1 (Bunderan HI)",
  "pm10": 38,
  "pm25": 53,
  "so2": 29,
  "co": 6,
  "o3": 31,
  "no2": 13
}
```

### Response

```json
{
  "res": "BAIK",
  "error_msg": ""
}
```

## Deploy Model in Local
### Cloning the repository
- Use `git clone` to clone this repository, so you can run the code in local environment.
```bash
git clone https://github.com/indraps30/mlp-aq2026
```

### Prerequisites
- Make sure Python and PIP installed in your system (this project use Python 3.12.3 and PIP 26.0.1).
- Create and activate the virtual environment, don't forget to update the package manager.
```bash
python3 -m venv .venv_aq
source .venv_aq/bin/activate
pip install --upgrade pip
```
- Install the required packages from `requirements.txt`.
```bash
pip install -r requirements.txt
```

### Running FastAPI app locally
1. Open a terminal or command prompt.
2. Start the FastAPI app by running the following command:
```bash
python src/api.py
```

This will start the FastAPI server on [http://localhost:8080](http://localhost:8080)

### Running Streamlit app locally
1. Open a new terminal or command prompt.
2. Start the Streamlit app by running the following command:
```bash
streamlit run src/ui.py
```

This will start the Streamlit app on [http://localhost:8501](http://localhost:8501)

Now, both FastAPI server and Streamlit app are running locally. You can interact with the Streamlit app to test and make prediction using the FastAPI and the trained model.

## Retraining Model
You can re-train the model by following these steps:
1. Create new folders `data/interim`, `data/processed`, and `logs`.
2. Make sure to activate the virtual environment and you have installed the required packages.
3. Execute these Python scripts sequentially:
```bash
python src/data_pipeline.py
python src/preprocessing.py
python src/modeling.py
```

Those scripts will re-load the raw dataset, pre-processed it, and re-train the models.

### Docker Build and Run
Or, if you want to use Docker engine, you can simply build a Docker image of this project and get it run with the following commands:
```bash
# make sure you are currently on the project folder.
sudo docker compose up --build
```

PS: tested on WSL2

---
**Last Update: 05-03-2026**