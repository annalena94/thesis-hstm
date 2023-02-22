import os
import numpy as np
import pandas as pd
import argparse

# This python script loads the preprocessed data contained in the dat folder files needed for analysis.
# Run this script from the src folder or set working directory accordingly in the run config.

def load_my_dataset(column):
	df = pd.read_csv('../dat/' + column + '.csv')
	docs = df[column].values
	responses = df['outcome_numeric'].values
	return docs, responses

def load_dataset_all():
	df = pd.read_csv('../dat/startup_data_final_all_text.csv')
	docs = df['text'].values
	responses = df['outcome_numeric'].values
	return docs, responses

def main(args):
	# data_file = args.data_file
	out = args.outfile
	dataset = args.data
	seed = 12345

	np.random.seed(seed)

	if dataset == 'my_dataset':
		doc, responses = load_my_dataset(args.column)
	elif dataset == 'my_dataset_all':
		doc, responses = load_dataset_all()

	df = pd.DataFrame(np.column_stack((doc.T, responses.T)) , columns=['text', 'label'])
	os.makedirs('../dat/csv_proc', exist_ok=True)

	if out == "":
		out = '../dat/csv_proc/' + dataset + '.csv'
	
	df.to_csv(out)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", action="store", default="")
	parser.add_argument("--outfile", action="store", default="")
	parser.add_argument("--column", action="store", default="from_idea_to_product")
	parser.add_argument("--data", action="store", default="amazon")
	args = parser.parse_args()

	main(args)