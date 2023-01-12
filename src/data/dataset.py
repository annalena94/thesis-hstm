from src.data.load_data_from_text import load_my_dataset, load_amazon
import argparse
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')

# This python script prepares the interview data for analysis. It contains functions to load the data, to generate the
# document-term-matrix, to normalize the data and to assign train/test splits.
# Run this script from the src folder or set working directory accordingly in the run config.

class LemmaTokenizer:

	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def __call__(self, doc):
		#return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if str.isalpha(t)]
		return [t for t in word_tokenize(doc) if str.isalpha(t)]


class TextResponseDataset(Dataset):
	CLASSIFICATION_SETTINGS = {'my_dataset', 'amazon'}

	def __init__(self, dataset_name, column, data_file, processed_data_file, **kwargs):
		super(Dataset, self).__init__()
		self.dataset_name = dataset_name
		self.data_file = data_file
		self.processed_data_file = processed_data_file
		self.column = column

		self.label_is_bool = False
		if self.dataset_name in TextResponseDataset.CLASSIFICATION_SETTINGS:
			self.label_is_bool = True

		self.eval_mode=False
			
		self.parse_args(**kwargs)

	def parse_args(self, **kwargs):
		self.pretrained_theta = kwargs.get('pretrained_theta', None)
		self.use_bigrams=bool(kwargs.get('use_bigrams', False))

	def load_data_from_raw(self):
		if self.dataset_name == 'my_dataset':
			docs, responses = load_my_dataset(self.column)
		if self.dataset_name == 'amazon':
			docs, responses = load_amazon();
		return docs, responses

	def load_processed_data(self):
		arrays = np.load(self.processed_data_file, allow_pickle=True)
		labels = arrays['labels']
		counts = arrays['counts']
		vocab = arrays['vocab']
		docs = arrays['docs']

		return counts, labels, vocab, docs

	def get_vocab_size(self):
		return self.vocab.shape[0]

	def process_dataset(self):
		if os.path.exists(self.processed_data_file):
			counts, responses, vocab, docs = self.load_processed_data()
		else:
			docs, responses = self.load_data_from_raw()
			stop = stopwords.words('english')

			#vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1), stop_words=stop, max_df=0.7, min_df=0.0007)
			vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1), stop_words=stop, max_df=0.7, min_df=0.0007)
			counts = vectorizer.fit_transform(docs).toarray()
			vocab = vectorizer.get_feature_names_out()

			if self.use_bigrams:
				exclude2 = {'doesn','don', 'but', 'not', 'wasn', 'wouldn', 'couldn', 'didn', 'isn'} #{'not'}
				stop2 = list(set(stop) - exclude2)
				bigram_vec = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop2, ngram_range=(2, 2), min_df=0.007)
				#bigram_vec = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop2, ngram_range=(2, 2), min_df=0.007)
				bigram_counts = bigram_vec.fit_transform(docs).toarray()
				counts = np.column_stack((counts, bigram_counts))
				bigrams = bigram_vec.get_feature_names_out()
				vocab = vocab+bigrams

			vocab = np.array(vocab)
			# saves the processed file to the /proc directory
			np.savez_compressed(self.processed_data_file, labels=responses, counts=counts, vocab=vocab, docs=docs)
			
		self.counts = counts
		self.vocab = vocab
		self.labels = responses
		self.docs = docs

	def set_to_eval_mode(self):
		self.eval_mode = True
		
	def preprocessing(self):
		term_total = self.counts.sum(axis=1)
		valid = (term_total > 1)
		self.labels = self.labels[valid]
		self.counts = self.counts[valid,:]
		self.docs = self.docs[valid]

		self.normalized_counts = self.counts / self.counts.sum(axis=1)[:,np.newaxis]

		if not self.label_is_bool:
			self.labels = (self.labels - self.labels.mean())/(self.labels.std())

	def assign_splits(self, train_indices, test_indices):
		self.train_set_counts = self.counts[train_indices, :]
		self.train_set_labels = self.labels[train_indices]
		self.test_set_counts = self.counts[test_indices, :]
		self.test_set_labels = self.labels[test_indices]
		self.train_set_docs = self.docs[train_indices]
		self.test_set_docs = self.docs[test_indices]

		self.train_set_normalized_counts = self.normalized_counts[train_indices, :]
		self.test_set_normalized_counts = self.normalized_counts[test_indices, :]

		if self.pretrained_theta is not None:
			self.train_set_pretrained_theta = self.pretrained_theta[train_indices, :]
			self.test_set_pretrained_theta = self.pretrained_theta[test_indices, :]
		else:
			self.train_set_pretrained_theta = None
			self.test_set_pretrained_theta = None

	def __getitem__(self, idx):
		if not self.eval_mode:
			datadict = {
					# Hint: this actually represents tf-idf when the TfidfVectorizer is used in process_dataset()
					'normalized_bow':torch.tensor(self.train_set_normalized_counts[idx, :], dtype=torch.float),
					'bow':torch.tensor(self.train_set_counts[idx, :], dtype=torch.long),
					'label':torch.tensor(self.train_set_labels[idx], dtype=torch.float)
				}
			if self.train_set_pretrained_theta is not None:
				datadict.update({'pretrained_theta':torch.tensor(self.train_set_pretrained_theta[idx, :], dtype=torch.float)})
		else:
			datadict = {
					'normalized_bow':torch.tensor(self.test_set_normalized_counts[idx, :], dtype=torch.float),
					'bow':torch.tensor(self.test_set_counts[idx, :], dtype=torch.long),
					'label':torch.tensor(self.test_set_labels[idx], dtype=torch.float)
				}
			if self.test_set_pretrained_theta is not None:
				datadict.update({'pretrained_theta':torch.tensor(self.test_set_pretrained_theta[idx, :], dtype=torch.float)})
		return datadict

	def __len__(self):
		return self.train_set_counts.shape[0]

	def get_full_size(self):
		return self.counts.shape[0]


def main():
	use_bigrams = True
	
	dataset = TextResponseDataset(data, data_file, column, use_bigrams=use_bigrams)
	dataset.process_dataset()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", action="store", default="my_dataset")
	parser.add_argument("--column", action='store', default="from_idea_to_product")
	parser.add_argument("--data_file", action='store', default="")

	args = parser.parse_args()
	data = args.data
	column = args.column
	data_file = args.data_file

	main()