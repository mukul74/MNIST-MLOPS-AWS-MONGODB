![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![MNIST](https://img.shields.io/badge/MNIST-blue?style=for-the-badge)


# MNIST Classification with FastAPI

A lightweight FastAPI-based service for loading, training, and predicting on the MNIST dataset and an attempt to use End2End MLOPS.

## Features

- **Data Storage:** Store the MNIST dataset in MongoDB.
- **Model Training:** Train a simple machine learning model for digit classification.
- **Prediction:** Predict on random samples and compare against ground truth.

## Prerequisites

Ensure you have the following installed before running the project:

- **MongoDB** (For storing the dataset)
- **Docker** (Optional, for containerized deployment)
- **Python** (For running FastAPI and model training)

## Installation & Running Locally

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
  git clone https://github.com/mukul74/MNIST_Classification.git
```

### 2. Navigate to the Project Directory

```bash
  cd MNIST_Classification
```

### 3. Create a Virtual Environment and Install Dependencies

```bash
  python -m venv venv
  source venv/bin/activate  # For Linux/Mac
  venv\Scripts\activate    # For Windows
  pip install -r requirements.txt
```
### 3.1 Start MongoDB Server
```bash
sudo systemctl start mongod
sudo systemctl start docker
```

### 3.2 Verify if these servers are active
```bash
systemctl status mongod
systemctl status docker
```

### 4. Run the FastAPI Application

```bash
  uvicorn src.main:app --reload
```

## Dockerizing the app

Follow the steps to dockerize the app

### 1. Create docker-compose.yml, .dockerignore and Dockerfile
    

### 2. Clean the previous images and containers
```bash
    docker compose down -v  # Removes all volumes and containers
    docker system prune -af  # Clears unused images and cache
```

### 3. Docker Compose
```bash
    docker-compose up --build
```

This will start the FastAPI server, which can be accessed at `http://127.0.0.1:8000`.

## API Endpoints

Once the server is running, you can interact with the API:

- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **Redoc:** `http://127.0.0.1:8000/redoc`

## Steps to install pre-commit

```bash
# Project setup
pip install pre-commit
pre-commit install
```
## Steps to push on Github

```bash
    git add @modified files
    git commit -m "relevant msg"
    git push -u origin main
```

## Author

- [Mukul Agarwal](https://github.com/mukul74)

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

## Upcoming
1. Adding test cases
2. Front End
3. Tracking of ML models and Data


## License

This project is open-source and available under the [MIT License](LICENSE).

