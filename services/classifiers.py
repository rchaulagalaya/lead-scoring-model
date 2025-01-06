import os
import pandas as pd
# import logging
# from config import Config
from joblib import dump

# from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from services import mongo_service as mongoDbService
from services import processors as processorService


from common import DataSource
DATA_SOURCE =DataSource

class classifier:
    #ctor
    def __init__(self, logger,config,data_processor,data_provider):
        self.logger = logger  # Use the injected logger DI
        self.config = config
        self.get_data = data_provider #processorService.data_provider(logger)
        self.data_processor = data_processor # processorService.data_processor(logger)
        self.db = mongoDbService.dbConnection(logger,config)

    def under_sampling_majority_class(self, feature, label):
        try:
            self.logger.info("Under Sampling majority class.")
            # Combine X and y into a single DataFrame for sampling
            X= feature
            y =label
            data = pd.concat([X, y], axis=1)

            # Separate majority and minority classes
            data_majority = data[data['FF Sent'] == 1]
            data_minority = data[data['FF Sent'] == 0]

            # Undersample the majority class
            n_samples_majority = len(data_majority)
            data_majority_downsampled = resample(
                data_majority,
                replace=False,  # sample without replacement
                # n_samples=1000,  # desired number of majority class samples
                n_samples=n_samples_majority , #1000  # adjust to the available number of majority samples
                random_state=42
            )

            # Oversample the minority class
            data_minority_upsampled = resample(
                data_minority,
                replace=True,   # sample with replacement
                n_samples=900,  # desired number of minority class samples
                random_state=42
            )

            # Combine minority and majority classes
            balanced_data = pd.concat([data_majority_downsampled, data_minority_upsampled])

            # Separate back into features and labels
            X_balanced = balanced_data.drop('FF Sent', axis=1)
            y_balanced = balanced_data['FF Sent']
            self.logger.info("Returned balanced Majority & Minority class.")
            return X_balanced, y_balanced

        except Exception as ex:
            self.logger.error(ex)
            return str(ex)

        ## TRAINING ONLY YO
    def get_binary_prediction(self,data_count,data_source,under_sample_flag=False):
        try:
            self.logger.info("Started BINARY Classification Model Generator")

            # Step 1: Get data : get training data to train model on          
            if data_source == DATA_SOURCE.JSON:
                training_data_path = self.config.get_json_file_path("training")
                # self.logger.info("Getting training json data file path %s", training_data_path)
                self.logger.info(f"Getting training json data file path: {str(training_data_path)}")
                data = self.get_data.from_json(data_count,training_data_path)
            elif data_source == DATA_SOURCE.DATABASE:
                data = self.get_data.from_db(data_count,"training_db") # when creating model, lets use prod data base server training_db
            else:
                self.logger.error("Invalid data source: %s",data_source)
                return {"error": "Invalid data source specified."}
                return None

            #Step 2: Define categorical and target variables
            categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
            target_column='FF Sent'
            X,y = self.data_processor.preprocess_data(data,categorical_columns,target_column)

            #Under Sampling Majority Class : better training
            usf = under_sample_flag #bool(under_sample_flag)
            # print(usf)
            if usf ==True:
                self.logger.info("Under sampling majority class by balancing features and labels")
                feature, label = self.under_sampling_majority_class(X,y)
                X = feature
                y = label
            
             # Step 5: Split data into train and test sets [ 80 % training,  20% test] -- SELECTION / INPUT PREPARATION
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

            self.logger.info("Training Class Distribution")
            df_y_train = pd.DataFrame(y_train.value_counts())
            self.logger.info(f"\n {df_y_train}")

            self.logger.info("Test Class Distribution")
            df_y_test = pd.DataFrame(y_test.value_counts())
            self.logger.info(f"\n {df_y_test}")

            # Step 6: Ensure X_train and y_train are aligned
            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)

            # Step 7: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier --- Model Training:
            model_name = self.config.get_binary_classifier_model_name()
            if os.path.exists(model_name):
                self.logger.info(
                    f"Model already exists: {model_name}. It will be replaced with the new trained model."
                )
            else:
                self.logger.info(f"Creating & Training a model: {model_name}")

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # training features
            model.fit(X_train, y_train) # model training
            dump(model,model_name)# save model, desired model name
            self.logger.info("Model saved as %s",model_name)

            #  Step 8: Make predictions
            y_pred = model.predict(X_test)

            # Step 9: Generate the classification report
            self.logger.info("Model's Classification Report")
            report = classification_report(y_test, y_pred, target_names=['will_not_reach', 'will_reach'],output_dict=True)
            # Convert the dictionary to a DataFrame
            df_report = pd.DataFrame(report).transpose()
            self.logger.info(f"\n {df_report.to_string()}")

            # Step 10: Evaluate Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            self.logger.info(f"Model's' Accuracy: {accuracy * 100:.2f}%")

            # return accuracy *100
            return f"{accuracy * 100:.2f}%"
        except Exception as ex:
            self.logger.error(ex)
            return str(ex)