import os
import pandas as pd
import logging
from config import Config
from joblib import dump

# from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from services import mongo_service as mongoDbService
from services import processors as processorService

# set logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)
config=Config()
# Mongo db
db = mongoDbService.dbConnection()
get_data = processorService.data_provider()
data_processor = processorService.data_processor()
data_source = processorService.DataSource

# This class is reponsible for generating / training all types of classifiers : Binary Classifier / Multi-label Classifier..
# parameter :
#                data source / file, db 
 #               data count / number
 #               under sampling majority class ? 1/0    
##
class classifier:
    def under_sampling_majority_class(self, feature, label ):
        try:
            logger.info("Under Sampling majority class.")
            # Combine X and y into a single DataFrame for sampling
            X= feature
            y =label
            data = pd.concat([X, y], axis=1)

            # Separate majority and minority classes
            data_majority = data[data['FF Sent'] == 1]
            data_minority = data[data['FF Sent'] == 0]

            # Undersample the majority class
            data_majority_downsampled = resample(
                data_majority,
                replace=False,  # sample without replacement
                n_samples=1000,  # desired number of majority class samples
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
            logger.info("Returned balanced Majority & Minority class.")
            return X_balanced, y_balanced

        except Exception as ex:
            logger.error(ex)
            return str(ex)

    def get_binary_prediction(self,data_count,data_source,under_sample_flag=False):
        try:
            logger.info("Started BINARY Classification Model Generator")

            # Step 1: Get data : get training data to train model on 
            training_data_path = config.get_json_file_path("training")
            logger.info("Training json data file path %s", training_data_path)

            if data_source == data_source.JSON:
                training_data_path = config.get_json_file_path("training")
                data = get_data.from_json(data_count,training_data_path)
            elif data_source == data_source.DATABASE:
                data = get_data.from_db(data_count)
            else:
                return None

            #Step 2: Define categorical and target variables
            categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
            target_column='FF Sent'
            X,y = data_processor.preprocess_data(data,categorical_columns,target_column)

            #better training
            if under_sample_flag==True:
                feature, label = self.under_sampling_majority_class(X,y)
                X = feature
                y = label

             # Step 5: Split data into train and test sets [ 80 % training,  20% test] -- SELECTION / INPUT PREPARATION
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

            logger.info("Training Class Distribution:")
            logger.info(y_train.value_counts())
            logger.info("Test Class Distribution:")
            logger.info(y_test.value_counts())

            # Step 6: Ensure X_train and y_train are aligned
            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)

            # Step 7: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier --- Model Training:
            model_name = config.get_binary_classifier_model_name()
            if os.path.exists(model_name):
                logger.info(
                    f"Model already exists: {model_name}. It will be replaced with the new trained model."
                )
            else:
                logger.info(f"Creating & Training a model: {model_name}")

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # training features
            model.fit(X_train, y_train) # model training
            dump(model,model_name)# save model, desired model name
            logger.info("Model saved as %s",model_name)

            #  Step 8: Make predictions
            y_pred = model.predict(X_test)

            # Step 9: Generate the classification report
            logger.info("Generating Classification Report")
            report = classification_report(y_test, y_pred, target_names=['will_not_reach', 'will_reach'])
            logger.info(report)


            # Step 10: Evaluate Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            return accuracy *100
        except Exception as ex:
            logger.error(ex)
            return str(ex)

     # Clean / Format / Parse data "" to NA














            """ #Step 2:  Process / clean / format data
            processed_data = self.process_data(data)
            processed_df = pd.DataFrame(processed_data)
            logger.info("Total PROCESSED data %s",len(processed_df))
            # logger.info("Total PROCESSED data %s",processed_df.to_string(index=False))

            # Step 3: Encode categorical variables
            label_encoder = LabelEncoder()
            categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
            for col in categorical_columns:
                processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))
            
            # Step 4:  Prepare features and labels ( x & y) -- FEATURE ENGINEERING
            X = processed_df[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']] # predictors , input
            y =  (processed_df['FF Sent'] != "NA").astype(int) # if ff sent has value then convert to 1 else 0. we aint encoding FF send coz its binary classification   // output"""
