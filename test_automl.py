import logging
from autogluon.tabular import TabularDataset, TabularPredictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# TO DO receive parameter from calling function
calling_param = ''

logging.info(f"STARTING with parameter {calling_param}")

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
logging.info(train_data.head())

label = 'class'
logging.info("Summary of class variable: \n", train_data[label].describe())


save_path = 'agModels-predictClass'  # specifies folder to store trained models

predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
logging.info(test_data_nolab.head())

predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data_nolab)
logging.info("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

logging.info(perf)
logging.info("DONE")

# TO DO
# Save run output of each logging.info statement to file in Azure storage
# File format: YYYYMMDD_HHMMSS_container_run.txt where time stamp is process start time
# return reponse 200 to calling function