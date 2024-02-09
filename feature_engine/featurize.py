"""
file: featurize.py
---
This file is the main driver of the feature generating pipeline. 
It instantiates and calls the FeatureBuilder class which defines the logic used for feature creation.
"""

# Importing the Feature Generating Class
from feature_builder import FeatureBuilder
import pandas as pd

# Main Function
if __name__ == "__main__":
	

	# Instantiating the Feature Generating Class
	# Calling the "engine"/"driver" function of the FeatureBuilder class 
	# that creates the features, and writes them in output.
	# Defines one class for each dataset.

	# FULL DATASETS BELOW
	model_name = "test"
	# Juries
	# jury_feature_builder = FeatureBuilder(
	#  	input_file_path = "../feature_engine/data/raw_data/jury_conversations_with_outcome_var.csv",
	#  	output_file_path_chat_level = f"../feature_engine/output/jury_output_chat_level_{model_name}.csv",
	#  	output_file_path_conv_level = "../feature_engine/output/jury_output_conversation_level_{model_name}.csv"
	#  )

	# jury_feature_builder.featurize(col="message")

	# CSOP
	
	csop_feature_builder = FeatureBuilder(
		input_file_path = "../feature_engine/data/raw_data/csopII_conversations_withblanks.csv",
		output_file_path_chat_level = f"../feature_engine/output/csopII_output_chat_level_{model_name}.csv",
		output_file_path_conv_level = f"../feature_engine/output/csopII_output_conversation_level_{model_name}.csv"
	)

	csop_feature_builder.featurize(col="message")
 
	# csop = pd.read_csv("../feature_engine/data/raw_data/csopII_conversations_withblanks.csv")
	# csop = csop.head(100)
	# csop.to_csv("../feature_engine/data/raw_data/csopII_conversations_withblanks_100.csv", index=False)
