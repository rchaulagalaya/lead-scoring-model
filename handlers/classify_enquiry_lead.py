# from curses import typeahead
from datetime import datetime, timezone
import sys, os
import logging
from numpy import column_stack
import pandas as pd
import numpy as np
from services import mongo_service as mongoDbService
# corelation
import seaborn as sns
import matplotlib.pyplot as plt
import json
from dateutil.parser import parse
from bson import ObjectId

# set logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Mongo db
db = mongoDbService.dbConnection()

from dateutil.parser import parse
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def process_opportunity_status(histories):
    # Step 1: Get the opportunityStatusHistory field
    opportunity_status_history = histories
    # logger.info("opportunity_status_histories %s ", len(opportunity_status_history))

    # Log the number of entries
    # logger.info("Processing %d entries in opportunityStatusHistory", len(opportunity_status_history))

    most_recent_entry = None  # Default value if no valid data exists

    if opportunity_status_history:
        # Step 2: Filter valid entries with `updateDate`
        valid_entries = [item for item in opportunity_status_history if "updateDate" in item]

        # logger.info("Found %d valid entries with updateDate", len(valid_entries))
        # logger.info(valid_entries)

        if valid_entries:
            # Step 3: Parse and normalize all `updateDate` fields
            for entry in valid_entries:
                # logger.info("VALID history %s", entry)
                if isinstance(entry["updateDate"], str):
                    # Parse string to offset-aware datetime
                    entry["updateDate"] = parse(entry["updateDate"])
                elif isinstance(entry["updateDate"], datetime) and entry["updateDate"].tzinfo is None:
                    # Convert naive datetime to offset-aware (UTC)
                    entry["updateDate"] = entry["updateDate"].replace(tzinfo=timezone.utc)

               # Replace empty strings in 'oldStatus' with 'NA'
            for entry in valid_entries:
                if entry.get("oldStatus", "") == "":
                    entry["oldStatus"] = "NA"

            # Step 4: Sort entries by updateDate (descending)
            sorted_entries = sorted(
                valid_entries,
                key=lambda x: x["updateDate"],
                reverse=True
            )
            sorted_df = pd.DataFrame(sorted_entries)

            # Replace empty strings in 'oldStatus' with 'NA' after sorting
            # sorted_df['oldStatus'] = sorted_df['oldStatus'].replace('', 'NA')
            # print(sorted_df.to_string(index=False))
            # logger.info("SORTED StatusValid Entries %s ", sorted_entries)

            most_recent_entry = sorted_entries[0]  # Select the most recent one
        else:
            logger.info("No valid entries with updateDate found.")
    else:
        logger.info("No opportunity status history available.")

    # Step 5: Extract the most recent oldStatus value
    recent_old_status = (
        most_recent_entry["oldStatus"]
        if isinstance(most_recent_entry, dict) and "oldStatus" in most_recent_entry
        else "NA"
    )
    return recent_old_status

def process_data(records):
    processed = []
    for record in records:
        # ID
        id = record.get("_id", 0)
        # logger.info("ID : %s", id)

        # Loan Amount
        loan_amount = record.get("loanAmount", "NA") or "NA"
        # logger.info("loan Amount %s", loan_amount)

        # Lead Source
        lead_source = record.get("leadSource", "NA") or "NA"

        # Lead Segment
        lead_segment = record.get("leadSegment", "NA") or "NA"

        # Lead Category
        lead_category = record.get("leadCategory", "NA") or "NA"

        # FF Sent
        ff_sent = record.get("lcr", {}).get("ffSent", "NA")

        # Validate the format of ff_sent
        if ff_sent != "NA" and isinstance(ff_sent, str):
            # Optional: Format the date if needed
            try:
                from datetime import datetime
                ff_sent = datetime.fromisoformat(ff_sent.replace("Z", "+00:00")).isoformat()
            except ValueError as e:
                logger.error("Failed to parse ffSent date: %s", e)


        # Format result
        processed.append(
            {
                "ID": id,
                "Loan Amount": loan_amount,
                "Previous Opportunity Status": recent_old_status,
                "Current Opportunity Status": current_opportunity_status,
                "Broker": broker_name,
                "FFSent": ff_sent,
            }
        )
    # logger.info(processed)
    return processed

# RETURN 20 data records
def getFromDB(data_count):
    try:
        logger.info("----------Fetching data from Database.")

        # Connect to MongoDB
        client = db.get_mongo_connection()  # Mongo server
        database = client["policymatrix"]  # MongoDB database
        collection = database["plf_lead_events"]  # MongoDB collection

         # Define the filter, projection, and sorting
        filter_query = {
            "enquiryId": {"$exists": True},  #  enquiryId is present
            "leadCategory": {"$in": ["LDAS", "INCOMPLETE"]}  #  leadCategory
            # ,"_id": "67648cfb86322ca6e4b93dee"
        }
        projection = {
            "_id": 1,
            "loanAmount": 1,
            # "leadCategory": 1,
            # "leadSource": 1,
            # "leadSegment": 1,
            "lcr": 1  # Nested field
        }
        sort_order = [("createdAt", -1)]  # Sort by createdAt in descending order

        # Fetch required data with filter, projection, and sorting
        data = list(
            collection.find(
                filter_query,  # Where filters
                projection     # Select 
            )
            .sort(sort_order)  # Sort by createdAt
            .limit(data_count) 
        )

        print(data)
        return None
        for doc in data:
            logger.info(f"Document: {doc}")
            # Extract ffSent safely for display
            ff_sent = doc.get("lcr", {}).get("ffSent", None)

        logger.info("Got data from Db . TOTAL %s", len(data))

        # Close MongoDB connection
        db.close_mongo_connection(connection=client)
        return data  # Return filtered and limited DataFrame
    except Exception as ex:
        logger.error("Error while reading data from database: %s", str(ex))
        return None

def lambda_handler(event, context):
    try:
        logger.info("Started enquiry lead classification")

        # Step 1: Get data    // 50 records
        dbData = getFromDB(5)
        sth =  pd.DataFrame(dbData)
        print("TRUE DATA")
        print(sth.to_string(index=False))
        return 

        #Step 2:  Process / clean/format data
        processed_data = process_data(dbData)

        # Step 3: Convert to dataframe
        processed_df = pd.DataFrame(processed_data)
        # print(processed_df.to_string(index=False))  # this is our main data set


        # Step 4:  Handle Missing Values
        processed_df.replace("NA", np.nan, inplace=True)  # Replace "NA" strings with NaN
        processed_df.fillna("Unknown", inplace=True)  # Fill NaN with a placeholder
        print(processed_df.to_string(index=False))  # this is our main data set

        # Step 5: Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Loan Amount','Previous Opportunity Status', 'Current Opportunity Status', 'Broker']
        for col in categorical_columns:
            processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))

        # Step 6:  Prepare features and labels ( x & y)
        X = processed_df[['Loan Amount', 'Previous Opportunity Status','Current Opportunity Status','Broker']] # independent variables
        y =  label_encoder.fit_transform(processed_df['FFSent'].astype(str))
        # processed_df = processed_df[processed_df['FFSent'] != "Unknown"]

        print("Encoded Y Values ")
        print(y)

        logger.info("Encoded X Independent variables")
        print(X.to_string(index=False))  # Debug: print X

        # Step 7: Split data into train and test sets [ 80 % training,  20% test]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("X_train")
        print(X_train.to_string(index=False))

        # print("X_test")
        # print(X_test.to_string(index=False))

        print("y_train")
        print(y_train)

        print("y_test")
        print(y_test)


        # Step 8: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42) # training features
        model.fit(X_train, y_train)  #training labels

        #  Step 9: Make predictions
        y_pred = model.predict(X_test)
        print("y_pred")
        print(y_pred)


         # Step 10:  Transform the Predicted FFSent
         # Inverse transform the predicted labels to get the actual predicted values
        actual_predicted_values = label_encoder.inverse_transform(y_pred)
        print("Actual Predicted Values:")
        print(actual_predicted_values)

        # Step 11: Evaluate Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")


        return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(
            {
                "message": "Successfully Completed Yuhuu !! ",
                "data": "data"
            }
        ),
    }
    except Exception as ex:
        logger.error(ex)
        return dict(statusCode=500, body=str(ex))