import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_l1_loss(x, dim=0, C=0.01):
	"""Calculates the L1 loss between a target and a set of weights

	input: tensor of batch size x number of classes
	output: total loss
	"""
	l1_loss = nn.L1Loss(reduction='sum')
	size = x.size()[dim]
	target = torch.zeros(size, device=device)
	num_classes = x.size()[dim+1]

	loss = 0
	for i in range(num_classes):
		weights = x[:,i]
		loss += C*l1_loss(weights,target)
	return loss

class HeterogeneousSupervisedTopicModel(nn.Module):
	def __init__(self, num_topics, vocab_size, num_documents, t_hidden_size=300, enc_drop=0.0, theta_act='thanh', label_is_bool=False, beta_init=None, C_weights=5e-4, C_topics=5e-6, response_model='hstm-all'):
		super(HeterogeneousSupervisedTopicModel, self).__init__()

		## define hyperparameters
		self.num_topics = num_topics # number of topics in the model
		self.vocab_size = vocab_size # size of the vocabulary in the data
		self.num_documents = num_documents # number of documents in the data
		self.t_hidden_size = t_hidden_size # size of the hidden layers in the model
		self.theta_act = self.get_activation(theta_act) # activation function for the encoding layer
		self.enc_drop = enc_drop # dropout rate for the encoding layer
		self.t_drop = nn.Dropout(enc_drop)
		self.C_topics = C_topics # weight decay regularization strength for the BoW weights
		self.C_weights = C_weights # weight decay regularization strength for the topic proportions (beta)
		self.C_base_rates = C_weights
		self.response_model = response_model # model used for prediction

		torch.manual_seed(42)

		# logit_betas: logit of the beta distribution for the topics / topic proportions
		if beta_init is not None:
			self.logit_betas = nn.Parameter(torch.tensor(beta_init, dtype=torch.float))
		else:
			self.logit_betas = nn.Parameter(torch.randn(vocab_size, num_topics))

		self.gammas = nn.Parameter(torch.randn(vocab_size, num_topics)) # topic-specific gamma values / topic weights (gamma)
		self.base_rates = nn.Parameter(torch.randn(vocab_size, 1)) # base rates for the topics (b)
		self.bow_weights = nn.Linear(vocab_size, 1) # the weights for the bag-of-words representation (omega)
		self.topic_weights = nn.Linear(num_topics, 1) # the weights for the topics (alpha)

		# setup neural network for encoding the document-topic associations
		self.q_theta = nn.Sequential(

				# first layer takes document representation as input (represented by bag-of-words) and outputs a
				# 't_hidden_size' dimensional vector
				nn.Linear(vocab_size, t_hidden_size),

				#  batch-normalization plus additional linear layers
				self.theta_act,
				nn.BatchNorm1d(t_hidden_size),
				nn.Linear(t_hidden_size, t_hidden_size),
				self.theta_act,
				nn.BatchNorm1d(t_hidden_size)
			)

		# final layer
		self.mu_q_theta = nn.Linear(t_hidden_size, num_topics) # layer for computing mean (µ) of the encodings
		self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics) # layer for computing the log-standard deviation (σ) of the encodings

		# set response model as binary or continuous
		self.is_bool = label_is_bool

		self.smoothing = nn.Parameter(torch.randn(vocab_size, 1))

	def get_activation(self, act):
		if act == 'sigmoid':
			act = nn.Sigmoid()
		elif act == 'tanh':
			act = nn.Tanh()
		elif act == 'relu':
			act = nn.ReLU()
		elif act == 'softplus':
			act = nn.Softplus()
		elif act == 'rrelu':
			act = nn.RReLU()
		elif act == 'leakyrelu':
			act = nn.LeakyReLU()
		elif act == 'elu':
			act = nn.ELU()
		elif act == 'selu':
			act = nn.SELU()
		elif act == 'glu':
			act = nn.GLU()
		else:
			print('Defaulting to tanh activations...')
			act = nn.Tanh()
		return act 

	def reparameterize(self, mu, logvar):
		"""
		Returns a sample from a Gaussian distribution via reparameterization.
		"""
		if self.training:
			std = torch.exp(0.5 * logvar) 
			eps = torch.randn_like(std)
			return eps.mul_(std).add_(mu)
		else:
			return mu

	def encode(self, bows):
		"""
		Returns parameters of the variational distribution for \theta.
		:param bows: batch of bag-of-words...tensor of shape bsz x V
		:return:  mu_theta, log_sigma_theta
		"""

		# apply neural network q_theta to input 'bows' to produce encoding for document-topic associations
		q_theta = self.q_theta(bows)
		if self.enc_drop > 0:
			q_theta = self.t_drop(q_theta)

		# pass encoding through two linear layers to produce the mean and log-standard deviation of the encoding
		mu_theta = self.mu_q_theta(q_theta)
		logsigma_theta = self.logsigma_q_theta(q_theta)

		# compute the KL-divergence loss that measures the difference between the encoding distribution and a prior
		kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()

		return mu_theta, logsigma_theta, kl_theta

	def set_beta(self):
		self.betas = F.softmax(self.logit_betas, dim=0).transpose(1, 0) ## softmax over vocab dimension

	def get_beta(self):
		return self.betas

	def get_logit_beta(self):
		return self.logit_betas.t()

	def get_theta(self, normalized_bows):
		"""
		Predicts the bag-of-words representation of the documents
		---> implementation of formula ?? in Sridhar et al., 2022
		:param normalized_bows: bag-of-word representation of documents
		:return: theta: topic distribution for a batch of documents
		"""

		# calculate parameters of the variational distribution for the topic proportions 'theta' given the document representation
		mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)

		# sample topic proportion from the Gaussian distribution with mu_theta and logsigma_theta
		z = self.reparameterize(mu_theta, logsigma_theta)

		# normalize the topic proportions to a valid probability distribution with positive values that sum up to 1
		theta = F.softmax(z, dim=-1)

		# return topic distribution and Kullback-Leibler divergence
		return theta, kld_theta

	def decode(self, theta):
		"""
		Predicts the bag-of-words representation of the documents
		---> implementation of formula (2) in Sridhar et al., 2022
		:param theta: topic distribution for a batch of documents
		:return: preds: predicted bag-of-words representation
		"""

		# logits = pre-activation output of the linear-layer applied to inputs (unnormalized log-probabilities of a categorical distribution)
		# compute logits by taking the matrix product of the topic distributions and the transposed topic-word associations
		logits = torch.mm(theta, self.logit_betas.t())

		# add constant bias for each document-word association (i.e. baseline popularity of each word in the corpus)
		logits += self.base_rates.squeeze(1)

		# calculate probability distribution over the words in the vocabulary for each document
		res = F.softmax(logits, dim=-1)

		# transform probabilities into log probabilities with natural logarithm
		preds = torch.log(res+1e-6)
		return preds

	def predict_labels(self, theta, bows):
		"""
		Computes the predicted labels for a set of documents
		:param theta: topic distribution for the documents
		:param bows: bag-of-words
		:return: predicted labels for each document
		"""
		# gammas = self.apply_attention_smoothing(theta)
		gammas = self.gammas
		scaled_beta = self.logit_betas * gammas
		weights = torch.mm(theta, scaled_beta.t())

		if self.response_model == 'stm':
			expected_pred = self.topic_weights(theta).squeeze()
		elif self.response_model == 'stm+bow':
			expected_pred = self.topic_weights(theta).squeeze() + self.bow_weights(bows).squeeze()
		elif self.response_model == 'hstm':
			expected_pred = (bows * weights).sum(1)
		elif self.response_model == 'hstm+bow':
			expected_pred = (bows * weights).sum(1) + self.bow_weights(bows).squeeze()
		elif self.response_model == 'hstm+topics':
			expected_pred = (bows * weights).sum(1) + self.topic_weights(theta).squeeze()
		elif self.response_model == 'hstm-all' or self.response_model == 'hstm-all-2stage':
			expected_pred = (bows * weights).sum(1)\
							+ self.bow_weights(bows).squeeze() + self.topic_weights(theta).squeeze()
			# theta is the distribution of topics in the document and topic_weights is the impact on outcome depending
			# on the topic distribution
		return expected_pred

		
	def forward(self, bows, normalized_bows, labels, theta=None, do_prediction=True, penalty_bow=True, penalty_gamma=True):
		"""
		Computes the loss of the model on a batch of documents
		:param bows: batch of bag-of-words...tensor of shape bsz x V
		:param normalized_bows:
		:param labels:
		:param theta: topic distribution for each document
		:param do_prediction:
		:param penalty_bow:
		:param penalty_gamma:
		:return: reconstruction loss, other loss, kl divergence loss
		"""

		# pick loss function based on label-type
		if self.is_bool:
			loss = nn.BCEWithLogitsLoss() # binary cross-entropy with logits loss
		else:
			loss = nn.MSELoss() # mean squared error loss

		other_loss = torch.tensor([0.0], dtype=torch.float, device=device)

		if theta is None:
			# calculate KL divergence loss  if the topic distribution theta (for each document) is not given
			theta, kld_theta = self.get_theta(normalized_bows)

			# calculate prediced bag-of-words representation
			preds = self.decode(theta)

			# calculate reconstruction loss as the negative mean sum of the element-wise product of the predicted
			# document representation and the input document representation
			recon_loss = -(preds * bows).sum(1).mean()

			other_loss = get_l1_loss(self.base_rates, C=self.C_weights)
		else:
			recon_loss = torch.tensor([0.0], dtype=torch.float, device=device)
			kld_theta = torch.tensor([0.0], dtype=torch.float, device=device)

		if do_prediction:
			expected_label_pred = self.predict_labels(theta, normalized_bows)

			# add label prediction loss
			other_loss += loss(expected_label_pred, labels)

			# add L1 loss of the topic weight gammas
			if penalty_gamma:
				other_loss += get_l1_loss(self.gammas, C=self.C_topics)

			# add L1 penalty on the bow_weights
			if penalty_bow:
				other_loss += self.C_weights*(torch.norm(self.bow_weights.weight))

		return recon_loss, other_loss, kld_theta

