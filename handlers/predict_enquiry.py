import os
import json
import logging
from services import classifiers as classifier
from services import processors as processorService
from config import Config
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

config=Config()
logger = Config.get_logger()
data_processor = processorService.data_processor(logger)
data_provider=processorService.data_provider(logger)
classify = classifier.classifier(config,logger,data_processor,data_provider)

def lambda_handler(event, context):
    try:
        logger.info("Prediction request received.")
        classifier_type=config.get_classifier_type()
        data_source = config.get_data_source()
        model_name = config.get_binary_classifier_model_name() # load('binary_classifier_rf_model_NEW.joblib')
        data_count = config.get_data_count()
        logger.info(f"Predictors Settings: \n" 
                            f"Data source: {data_source},\n "
                            f"Data count: {data_count}, \n"
                            f"Model name: {model_name}, \n"
                            f"Classifier Type: {classifier_type.value}\n")  # Log classifier type as string

        output ={}

        # Step 1: get model
        if os.path.exists(model_name):
            model=load(model_name)  # load model
            logger.info(
                f"Model Found: {model_name}. This model will be used for prediction."
            )
        else:
             return dict(statusCode=200, body="We've received your request!!")

        # Step 2: get data        
        if data_source == data_source.JSON:
            testing_data_path = config.get_json_file_path("testing")
            sample_test_data = data_provider.from_json(data_count,testing_data_path)
        elif data_source == data_source.DATABASE:
            sample_test_data = data_provider.from_db(data_count,"testing_db" )#todo: enum purpose
        else:
            logger.info("Data source for prediction request not found. Data Source : %s", data_source)
            return dict(statusCode=200, body="We've received your request.!!")

        if not sample_test_data:
            logger.info("No data found for prediction.")
            return dict(statusCode=200, body="We've received your request!!")

        logger.info("Fetched testing data. Data count %s",len(sample_test_data))

       #Step 2: Define categorical and target variables
        categorical_columns = ['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']
        target_column = 'FF Sent' 
        # get encoded features and labels
        X_test, y_test = data_processor.preprocess_data(sample_test_data,categorical_columns,target_column)
        logger.info("Encoded prediction data (feature and labels.)")
        # Feed into Model predictions
        logger.info("Feeding test data into model for predictions. %s",len(X_test))
        # logger.info(X_test)
        # logger.info(y_test)
        y_pred = model.predict(X_test)
        output = {"predictions": y_pred.tolist()}

         # If ground truth labels are provided, evaluate model performance
        if y_test is not None:
            logger.info("Evaluating Model's Performance : you've provided output")
            accuracy = accuracy_score(y_test, y_pred)
            # df_ac_report = pd.DataFrame(accuracy)
            logger.info(f"Accuracy Report :\n {accuracy}")

            report = classification_report(y_test, y_pred,zero_division=0, output_dict=True)   
            df_report = pd.DataFrame(report).transpose()
            logger.info(f"Classification Report : \n {df_report.to_string()}")

            output["evaluation"] = {
                "accuracy": accuracy,
                "classification_report": report
            }
            logger.info("Successfully generated predictions and evaluation.")
            # logger.info(output)
        else:
            logger.info(output)
            logger.info("Successfully generated predictions.")
        return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(
            {
                "message": "Successfully Predicting Enquiry for given data set !! ",
                "data":  output 
            }
        ),
    }
    except Exception as ex:
        logger.error(ex)
        return dict(statusCode=500, body=str(ex))