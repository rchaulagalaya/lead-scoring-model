import os
import json
from config import Config  
from services import processors as processors
from services import classifiers as classifier
from common import DataSource, ClassifierType
# Import the Config & services class
# Initialize logger
logger = Config.get_logger()
# Initialize config
config=Config()
# Initialize data processor and data provider
data_processor = processors.data_processor(logger)
data_provider = processors.data_provider(logger)
# Pass all required arguments to the classifier
classify = classifier.classifier(logger,config, data_processor, data_provider)
# classify = classifier.classifier(logger)



# This handler is responsible for Generating the supervised classifying model for Enquiry's Lead reach state
#  Whether the enquiry will have ffSent reached or not?? 
## Gives Accuracy % OR  returns Accuracy & Evaluation of models performance
def lambda_handler(event, context):
    try:
        # logger = Config.get_logger()
        logger.info("Classifier request received.")
        # logger.info(f"Classifier request received for event: {event}")
        data_source = config.get_data_source()        
        data_count = config.get_data_count()
        classifier_type = config.get_classifier_type()
        under_sample_flag= config.get_under_sample_flag()
        logger.info(f"Classifier Settings: \n " 
                         f"Data source: {data_source}, \n "
                        f"Data count: {data_count}, \n "
                        f"Under Sample Majority Flag: {under_sample_flag}, \n "
                        f"Classifier Type: {classifier_type.value}. \n")
        output ={}
        if classifier_type== ClassifierType.BINARY:
            output = classify.get_binary_prediction(data_count,data_source,under_sample_flag)
        logger.info("Classifier request completed.")        
        return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(
            {
                "message": "Successfully Trained & Saved the Binary Classifier Model !! ",
                "data":  {"Accuracy" : f"{output}"}
                # "data":  {"Accuracy" : f"{output :.2f}%"}
                # "data":  {"Accuracy": f"{float(output):.2f}%"}

            }
        ),
    }
    except Exception as ex:
        logger.error("An error occurred during classification.", exc_info=True)
        return dict(statusCode=500, body=str(ex))