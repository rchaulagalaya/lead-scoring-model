import logging
from services import mongo_service as mongoDbService
# set logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Mongo db.
db = mongoDbService.dbConnection()

import json
from datetime import datetime, timezone

class data_service:
     # Function to format the date
    def format_date(self,date_str):
        try:
            # Parse the date and format it to the desired string format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            return "NO DATE"

    def from_json(self, data_count):
        try:
            json_file_path = "sample-data/plf_lead_events_raw_prod_original.json"
            # json_file_path = "sample-data/plf_lead_events_raw_prod.json"
            logger.info("----------GET JSON data from %s", json_file_path)

            # Get the start of the current month
            now = datetime.now()
            start_of_month = datetime(now.year, now.month, 1)

            # Load JSON data from the file
            with open(json_file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                logger.info("Loaded JSON data count: %d", len(json_data))

                # Filter data
                filtered_data = []
                for doc in json_data:
                    try:
                        # Parse and compare createdAt dates
                        created_at = doc.get("createdAt", {}).get("$date", "NA")
                        if created_at == "NA":
                            continue  # Skip invalid dates
                        created_at_date = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        ).replace(tzinfo=None)

                        lead_category = doc.get("leadCategory", "NA")
                        lead_source = doc.get("leadSource", "NA")
                        lead_segment = doc.get("leadSegment", "NA")
                        enquiry_id = doc.get("enquiryId", "NA")
                        ff_sent = doc.get("lcr", {}).get("ffSent", {}).get("$date", "NA")
                        # Process ff_sent only if it is not NA
                        if ff_sent != "NA":
                            ff_sent_date = datetime.fromisoformat(ff_sent.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S.%f")

                        if (
                            enquiry_id is not None
                            and lead_category in ["LDAS", "INCOMPLETE"]
                            and lead_source not in [None, ""]
                            and lead_segment not in [None, ""]
                            and created_at_date < start_of_month
                            and "lcr" in doc
                        ):
                            filtered_data.append(doc)
                    except Exception as inner_error:
                        logger.warning("Error processing document: %s. Skipping it.", inner_error)

                logger.info("Filtered JSON data count: %d", len(filtered_data))

                 # Process and standardize the fetched data
                processed_data = []
                for f_data in filtered_data:
                     # Check if ffSent is a datetime object , yes then convert to string
                    fff_sent_date = f_data.get("lcr", {}).get("ffSent", {}).get("$date", "NA")#ff_sent = doc.get("lcr", {}).get("ffSent", "NA")
                    if isinstance(fff_sent_date, datetime):
                        fff_sent_date = datetime.fromisoformat(fff_sent_date.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S.%f")    #ff_sent.strftime("%Y-%m-%d %H:%M:%S")  # Convert to string

                    processed_data.append({
                        "_id": f_data.get("_id"),
                        "loanAmount": f_data.get("loanAmount", "NA"),
                        "leadCategory": f_data.get("leadCategory", "NA"),
                        "leadSource": f_data.get("leadSource", "NA"),
                        "leadSegment": f_data.get("leadSegment", "NA"),
                        "lcr.ffSent": fff_sent_date
                    })

                logger.info("Standardized JSON data count %s", len(processed_data))
                return processed_data

        except Exception as e:
            logger.error("An error occurred while reading the JSON data: %s", e)

    def from_db(self, data_count):
        try:
            logger.info("----------Fetching data from Database.")

            # Get the start of the current month
            now = datetime.now()
            start_of_month = datetime(now.year, now.month, 1)

            # Connect to MongoDB
            client = db.get_mongo_connection()
            database = client["policymatrix"]
            collection = database["plf_lead_events"]

             # Define the filter, projection, and sorting
            filter_query = {
                "enquiryId": {"$exists": True}  #  enquiryId is present
                ,"leadCategory": {"$in": ["LDAS", "INCOMPLETE"]} #  leadCategory
                ,"leadSource":{"$exists": True, "$ne": None}
                ,"leadCategory":{"$exists": True, "$ne": None}
                ,"leadSegment":{"$exists": True, "$ne": None}
               , "createdAt": {"$lt": start_of_month}  # Exclude data from the current month
               ,"lcr":  {"$exists": True}
                # , "_id": ObjectId("643402b5bf990396f0251d5b")
                # , "_id": {"$in": [ObjectId("643402b5bf990396f0251d5b"), ObjectId("64341c3bbf990396f0273034"),ObjectId("643434f3bf990396f0292d78"),ObjectId("643474ffbf990396f02e4d0b"),ObjectId("64347db7bf990396f02f00b3"),ObjectId("64348114bf990396f02f4901"),ObjectId("64348d2dbf990396f03043fd"),ObjectId("643497b2bf990396f0311e6e"),ObjectId("64349b2ebf990396f0316941"),ObjectId("643434f3bf990396f0292d78"),]}
            }
            projection = {
                "_id": 1,
                "loanAmount": 1,
                "leadCategory": 1,
                "leadSource": 1,
                "leadSegment": 1,
                "lcr": 1  # Nested field
            }
            sort_order = [("createdAt", -1)]

            # Fetch required data with filter, projection, and sorting
            data = list(
                collection.find(
                    filter_query,
                    projection
                )
                .sort(sort_order)
                .limit(data_count)
            )

            # Process and standardize the fetched data
            processed_data = []
            for doc in data:
                 # Check if ffSent is a datetime object , yes then convert to string
                ff_sent = doc.get("lcr", {}).get("ffSent", "NA")
                if isinstance(ff_sent, datetime):
                    ff_sent = ff_sent.strftime("%Y-%m-%d %H:%M:%S")  # Convert to string

                processed_data.append({
                    "_id": doc.get("_id"),
                    "loanAmount": doc.get("loanAmount", "NA"),
                    "leadCategory": doc.get("leadCategory", "NA"),
                    "leadSource": doc.get("leadSource", "NA"),
                    "leadSegment": doc.get("leadSegment", "NA"),
                    "lcr.ffSent": ff_sent
                })

            logger.info("Got data from Db . TOTAL %s", len(processed_data))
            # Close MongoDB connection
            db.close_mongo_connection(connection=client)
            return processed_data  # Return filtered and limited DataFrame
        except Exception as ex:
            logger.error("Error while reading data from database: %s", str(ex))
            return None
