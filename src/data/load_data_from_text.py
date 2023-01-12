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

# Function for testing other data set
def load_amazon():
	df = pd.read_csv('../dat/amazon.csv')
	docs = df['text'].values.astype('U')
	responses = df['label'].values.astype('float')
	return docs, responses

def main(args):
	# data_file = args.data_file
	out = args.outfile
	dataset = args.data
	seed = 12345

	np.random.seed(seed)

	if dataset == 'my_dataset':
		doc, responses = load_my_dataset()

	df = pd.DataFrame(np.column_stack((doc.T, responses.T)) , columns=['text', 'label'])
	os.makedirs('../dat/csv_proc', exist_ok=True)

	if out == "":
		out = '../dat/csv_proc/' + dataset + '.csv'
	
	df.to_csv(out)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", action="store", default="")
	parser.add_argument("--outfile", action="store", default="")
	parser.add_argument("--data", action="store", default="amazon")
	args = parser.parse_args()

	main(args)