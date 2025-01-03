import json
import logging
from services import classifiers as classifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)
classify = classifier.classifier()

def lambda_handler(event, context):
    try:
        data_count = 200000
        binary_prediction = True
        output ={}

        if binary_prediction:
            output = classify.get_binary_prediction(data_count)
        # else:
        #     output = classify.get_multi_label_prediction(data_count)

        return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(
            {
                "message": "Successfully Completed Yuhuu !! ",
                "data":  output #"data"
            }
        ),
    }
    except Exception as ex:
        logger.error(ex)
        return dict(statusCode=500, body=str(ex))