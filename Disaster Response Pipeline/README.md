# DSND-Disaster-Response-Pipelines
During disasters, there are many communications and messages regarding the needs of people affected by the disaster. Keyword search can be used to find out the exact needs, but it is time consuming when organizations have least capacity to handle, and it can be misleading. For example, keyword "water" does not necessarily indicate the need of water, alternatively, keyword "thirsty" could actually indicate the need of drinking water.

In this project, we build supervised machine learning models to accurately categorize the messages during disasters.

## Installation
Run the following command to clone to local machine

```
git clone https://github.com/yukiteb/DSND-Disaster-Response-Pipelines.git
```

## File structure
The files are structured as follows:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py  # python script to process data and save to dababase
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py  # python script to train a model and save the model
|- classifier.pkl  # saved model 

- README.md
```

## How to run

Clean the data and save as SQLite database named "DisasterResponse.db"
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

Train and save the model as "classifier.pkl"
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

Run the Flask App
```
python run.py
```
Once the app is up and running, type the following in your browser to run on local machine.
```
localhost:3001
```
## Example of the Web app
If you run the Web app, you will see the overview of the training data as below.
![Overview of Training Data](https://github.com/yukiteb/Data-Science-Nanodegree/blob/master/Disaster%20Response%20Pipeline/overview_training_data.PNG)

If you type "I am very thirsty" in the message and hit "Classify Message", this is the result. 

![Example Result](https://github.com/yukiteb/Data-Science-Nanodegree/blob/master/Disaster%20Response%20Pipeline/example_result.PNG)

