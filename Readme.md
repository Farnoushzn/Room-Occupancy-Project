This project predicts room occupancy based on sensor data.

# Room Occupancy ML Project

This project predicts room occupancy based on sensor data. The workflow includes:

- Training an XGBoost model using a structured pipeline.
- Serving predictions via a REST API.
- Containerizing components with Docker.
- Automating workflows using CI/CD.

## Folder Structure
- `data/`: Contains datasets used for training and testing.
- `notebooks/`: Jupyter notebooks for prototyping.
- `scripts/`: Python scripts for data processing and model training.
- `models/`: Trained models saved for inference.
- `api/`: API service for making predictions.
- `docker/`: Dockerfiles for containerization.
- `tests/`: Tests to ensure the project works as expected.

## Instructions
1. Set up the environment by installing dependencies (`requirements.txt`).
2. Run `DataExploration.ipynb` to preprocess and save the processed data.
3. Run `ModelTraining.ipynb` to train, optimize, and evaluate the model.
4. The saved model can be used for further predictions or deployment.

## Dependencies
All dependencies can be found in the `requirements.txt` file.

## Setup Instructions

### Setting Up Git on macOS

If you don't have Git installed, you can use Homebrew to install it by running the following command in your macOS Terminal:

```sh
brew install git  # Run this command in your mac terminal to install Git
```


### Setting Up the Conda Environment

To set up the Conda environment, follow these steps:

**Create a Conda Environment**:

   Run the following command to create a new Conda environment named `room_occupancy`:

   ```sh
   conda create -n room_occupancy python=3.8
   conda activate room_occupancy
   conda install pandas scikit-learn xgboost flask joblib jupyter
   ```

## Docker Containerization
The project includes Docker containerization to make it easier to deploy and run the application in a standardized environment.

### Dockerfile Overview
Two Dockerfiles is provided in the Docker folder to define all the dependencies and setup required to run the application in a container. The main components of the Dockerfile include:

#### Dockerfile for Model Training
- A Dockerfile named **`Dockerfile.training`** is provided to containerize the model training process. It handles:
  - Installing dependencies
  - Copying the training scripts, datasets, models
  - Running the training script to train the model
  - building the Docker Image

  - **Build and Run Model Training**:

  ```sh
  docker build -f docker/Dockerfile.training -t room-occupancy-training .
  docker run --name room-occupancy-training-container room-occupancy-training
  ```

#### Dockerfile for API Service
- A Dockerfile named **`Dockerfile.api`** is provided to create and run the API service. It handles:

1. Base Image: The image starts from python:3.8-slim as the base.
2. Install Dependencies: Installs all required Python packages using requirements.txt.
3. Copy Application Code: Copies the application code (API/) into the container.
4. Expose Port: Exposes port 5000 to handle requests from outside.
5. Run the Application: Runs the Flask server for serving predictions.
6. Building the Docker Image


To create a Docker image for the application, navigate to the root of the project directory and run:
```sh
docker build -f Docker/Dockerfile -t room-occupancy-api .

```
## Running the Docker Container
Once the Docker image is built, you can run it using:
```sh
docker run -d -p 5000:5000 --name room-occupancy-container room-occupancy-api

```
## Accessing the API
The API provides endpoints for checking server status and making occupancy predictions. Below are examples of how to access these endpoints.

### Home Endpoint
To confirm the server is running, go to: [http://127.0.0.1:5000/]

### Prediction Endpoint
You can make a prediction request using Python script in folder scripts/prediction_req.py or by `curl` to verify prediction accuracy and API response structure.

```sh
curl -X POST -H "Content-Type: application/json" -d '{"feature1": value, "feature2": value}' http://127.0.0.1:5000/predict
```

## stop and remove container if necessary
```sh
docker stop room-occupancy-container
docker rm room-occupancy-container
```

# Git Setup and Pushing the Project to GitHub
1. Initialize the Git Repository: Navigate to your project directory and initialize Git:
```sh
git init
```
2. Add Files to Staging: Add all the project files to the staging area:
```sh
git add .
```
3. Commit Changes: Create a commit to save your changes:
```sh
git commit -m "Initial commit with project files"
```
4. Connect to GitHub Repository: Create a new repository on GitHub. Then, link the local repository to the GitHub remote repository:
```sh
git remote add origin https://github.com/Farnoushzn/Room-Occupancy-Project.git
```

5. Push to GitHub: Push your local commits to the GitHub repository:
```sh
git branch -M main
git push -u origin main
```

## Contact
Maintainer: Farnoush Zohourian
Email: f.zohourian@gmail.com