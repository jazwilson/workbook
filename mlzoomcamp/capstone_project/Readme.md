
# Capstone Project: Measle vaccine rates

In 2019, the [Center for Disease Control and Prevention reported](https://www.cdc.gov/measles/cases-outbreaks.html) the highest incidence of measles cases in the US since 1992. The majority of individual cases were observed in the non-vaccinated population. To investigate the risk posed to children attending school in the US, The Wall Street Journal documented the vaccination rates for measles/mumps/rubella or ‘mmr’. Here, we use this dataset to see whether the type of schooling institution and the state in which the school is located plays any bearing on the mmr vaccination rate.

Data from: [TidyTuesday](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-25/readme.md)

## Environment Setup

Environment managed using `pipenv`.
To install pipenv: 

1. Run `pip install pipenv` 
2. Run `pipenv install --dev` to install all dependencies 
3. Launch the shell using `pipenv shell`

## Model training

Open [notebook.ipynb](https://github.com/jazwilson/workbook/blob/main/mlzoomcamp/capstone_project/notebook.ipynb)for EDA and model tuning

To train the final model use: [train.py](https://github.com/jazwilson/workbook/blob/main/mlzoomcamp/capstone_project/train.py) by running 

```bash
pipenv shell
python train.py
```

The data `data.csv` and model `view-model` have been committed to the project.
I provide [test.py](https://github.com/jazwilson/workbook/blob/main/mlzoomcamp/capstone_project/test.py) to test the docker and local kubernetes deployments described below.
 
## Docker Images and deployment

To install deployment dependencies, see instructions provided by the following links:
 - **MacOSX:** [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
 - **Linux and Windows:** [docker-compose](https://docs.docker.com/compose/install/), [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/), and [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)

To install and deploy the Docker instances navigate to the folder and run 

```bash
docker build -t vaccination-gateway:v001 -f vaccination-gateway.dockerfile .
docker build -t vaccination-model:v001 -f vaccination-model.dockerfile .

docker-compose up
```

To test run `python test.py`.

## Kubernetes Setup

1. Run `kind create cluster`
2. Load the docker images to kind 

```bash
kind load docker-image vaccination-gateway:v001
kind load docker-image vaccination-model:v001
```

3. Deploy pods and services
```bash
kubectl apply -f kube_config/model-deployment.yaml
kubectl apply -f kube_config/model-service.yaml
kubectl apply -f kube_config/gateway-deployment.yaml
kubectl apply -f kube_config/gateway-service.yaml
```

4. To test locally set port to coincide with the tester
```bash
kubectl port-forward service/gateway 9696:80
```
and test using `python test.py`

5. Remove the cluster using `kind delete cluster`.
