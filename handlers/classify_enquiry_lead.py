import os
import json
import logging
from services import classifiers as classifier
from services import processors as processors
from config import Config  # Import the Config class

logger = logging.getLogger()
logger.setLevel(logging.INFO)
classify = classifier.classifier()
config=Config()
DATA_SOURCE= processors.DataSource
CLASSIFIER_TYPE= processors.ClassifierType


# This handler is responsible for Generating the supervised classifying model for Enquiry's Lead reach state
#  Whether the enquiry will have ffSent reached or not?? 
## Gives Accuracy % OR  returns Accuracy & Evaluation of models performance
def lambda_handler(event, context):
    try:
        logger.info("Classifier request received.")
        data_source = config.get_data_source()        
        data_count = config.get_data_count()
        classifier_type = config.get_classifier_type()
        under_sample_flag= config.get_under_sample_flag()
        logger.info(f"Classifier Settings: \n" 
                    f"Data source: {data_source}, "
                    f"Data count: {data_count}, "
                    f"Under Sample Majority Flag: {under_sample_flag}, "
                    f"Classifier Type: {classifier_type.value}.")  # Log classifier type as string
        output ={}

        if classifier_type== CLASSIFIER_TYPE.BINARY:
            output = classify.get_binary_prediction(data_count,data_source,under_sample_flag)
        
        return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(
            {
                "message": "Successfully Trained & Saved the Binary Classifier Model !! ",
                "data":  {"Accuracy" : f"{output :.2f}%"} #"data"
            }
        ),
    }
    except Exception as ex:
        logger.error(ex)
        return dict(statusCode=500, body=str(ex))