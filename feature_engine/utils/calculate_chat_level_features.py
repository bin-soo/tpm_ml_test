"""
file: calculate_chat_level_features.py
---
This file defines the ChatLevelFeaturesCalculator class using the modules defined in "features".
The intention behind this class is to use these modules and define any and all chat level features here. 

The steps needed to add a feature would be to:
- First define any building blocks that the feature would need in the appropriate "features" module (like word counter).
- Define a function within the class that uses these building blocks to build the feature and appends it 
  to the chat level dataframe as columns.
- Call the feature defining function in the driver function.
"""

# Importing modules from features
import pandas as pd
from features.basic_features import *
from features.sentiment_features import *

class ChatLevelFeaturesCalculator:
	def __init__(self, chat_data: pd.DataFrame) -> None:
		"""
			This function is used to initialize variables and objects that can be used by all functions of this class.

		PARAMETERS:
			@param chat_data (pd.DataFrame): This is a pandas dataframe of the chat level features read in from the input dataset.
		"""
		self.chat_data = chat_data
		
	def calculate_chat_level_features(self) -> pd.DataFrame:
		"""
			This is the main driver function for this class.

		RETURNS:
			(pd.DataFrame): The chat level dataset given to this class during initialization along with 
							new columns for each chat level feature.
		"""
		# Text-Based Basic Features
		self.text_based_features()
		
		# YOUR SENTIMENT FUNCTION WILL BE CALLED HERE
		self.sentiment_features()

		# Return the input dataset with the chat level features appended (as columns)
		return self.chat_data
		
	def text_based_features(self) -> None:
		"""
			This function is used to implement the common text based features.
		"""
		# Count Words
		self.chat_data["num_words"] = self.chat_data["message"].apply(count_words)
		
		# Count Characters
		self.chat_data["num_chars"] = self.chat_data["message"].apply(count_characters)
		
		# Count Messages		
		self.chat_data["num_messages"] = self.chat_data["message"].apply(count_messages)
		
	def sentiment_features(self) -> None:
		"""
			This function is where your sentiment function will be called.
		"""
		self.senti_model = sentiment_model()
		print("Sentiment Model Loaded!")
		df_scaled, df_combined = normalization(self.chat_data)
		self.chat_data = clustering(df_scaled, df_combined)
		# self.chat_data["sentiment"] = self.chat_data["message"].apply(lambda x: get_sentiment(self.senti_model, x))
		self.chat_data["sentiment"] = self.chat_data.apply(
        lambda row: get_sentiment(
            self.senti_model, 
            row["message"], 
            row["cluster"], 
        ),
        axis=1
    )
	# def get_sentiment_features(self, model, row):
	# 	chat_message = row["message"]
	# 	efficiency = row["efficiency"]
	# 	duration = row["duration"]
	# 	score = row["score"]
	# 	# Call the model for sentiment prediction
	# 	sentiment = predict_sentiment_with_features(model, chat_message, efficiency, duration, score)
	# 	return sentiment
