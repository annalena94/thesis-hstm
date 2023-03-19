import absl
from src.model import adjusted_hstm
from src.model.model_trainer import ModelTrainer
from src.evaluation.evaluator import Evaluator
from src.util import util
import os
import numpy as np
from src.data.dataset import TextResponseDataset
from torch.utils.data import DataLoader
from absl import flags
from absl import app


# This python script runs the training of the topic model for the specified data set.
# Run this script from the src folder or set working directory accordingly in the run config.

def main(argv):
    """
    this function allows the script to be run from the command line or with absl incl. parameters
    e.g. run_experiment.py --data=my_dataset --model=hstm-all --epochs=20 --num_topics=5
    """

    # configure path to save processed data
    processed_file = FLAGS.procfile

    # data set used
    base_dataset = FLAGS.data

    if processed_file == "":
        processed_file = '../dat/proc/' + base_dataset + '_proc.npz'

    # set number of topics for training
    num_topics = FLAGS.num_topics

    label_is_bool = True

    print("Running model", FLAGS.model, '..' * 20)

    beta = None
    text_dataset = TextResponseDataset(FLAGS.data, FLAGS.column, FLAGS.datafile, processed_file)
    # transform input data to document-term matrix
    text_dataset.process_dataset()

    # normalize counts
    text_dataset.preprocessing()

    # set total number of documents in data set
    total_docs = text_dataset.get_full_size()

    # if train_test_mode = true, run train/test experiment otherwise do cross-validation (default)
    if FLAGS.train_test_mode:
        train_set_indices = np.arange(0, FLAGS.train_size)
        test_set_indices = np.arange(FLAGS.train_size + 1, total_docs)
    else:
        # assigns documents to specified number of splits
        split_indices = util.cross_val_splits(total_docs, num_splits=FLAGS.num_folds)
        all_indices = np.arange(total_docs)
        test_set_indices = split_indices[FLAGS.split]
        train_set_indices = np.setdiff1d(all_indices, test_set_indices)

    text_dataset.assign_splits(train_set_indices, test_set_indices)

    train_params = {'batch_size': FLAGS.batch_size, 'shuffle': True, 'num_workers': 0}
    training_dataloader = DataLoader(text_dataset, **train_params)
    vocab_size = text_dataset.get_vocab_size()
    n_docs = len(text_dataset)

    model = adjusted_hstm.HeterogeneousSupervisedTopicModel(num_topics, vocab_size, n_docs,
                                                            label_is_bool=label_is_bool, beta_init=beta,
                                                            C_weights=FLAGS.C, C_topics=FLAGS.C_topics,
                                                            response_model=FLAGS.model)

    trainer = ModelTrainer(model,
                           model_name=FLAGS.model,
                           use_pretrained=False,
                           do_pretraining_stage=False,
                           do_finetuning=False,
                           save=FLAGS.save,
                           load=FLAGS.load,
                           model_file=FLAGS.model_file)

    trainer.train(training_dataloader, epochs=FLAGS.epochs, extra_epochs=FLAGS.extra_epochs)

    # Negative log likelihood
    test_nll = trainer.evaluate_heldout_nll(text_dataset.test_set_counts, theta=text_dataset.test_set_pretrained_theta)
    print("Held out neg. log likelihood:", test_nll)

    test_err, predictions = trainer.evaluate_heldout_prediction(text_dataset.test_set_counts,
                                                                text_dataset.test_set_labels,
                                                                theta=text_dataset.test_set_pretrained_theta)

    test_mse = perplexity = npmi = shuffle_loss = 0.0
    test_auc = test_accuracy = test_log_loss = test_precision = test_f1 = 0.0
    test_auc = test_err[0]  # auc = area under curve
    test_log_loss = test_err[1]
    test_accuracy = test_err[2]
    test_precision = test_err[3]
    test_f1 = test_err[4]

    print("AUC on test set:", test_auc, "Log loss:", test_log_loss)
    print("Accuracy:", test_accuracy, "Precision:", test_precision, "F1:", test_f1)

    # evaluation based on test set
    evaluator = Evaluator(model,
                          text_dataset.vocab,
                          text_dataset.test_set_counts,
                          text_dataset.test_set_labels,
                          text_dataset.test_set_docs,
                          model_name=FLAGS.model)

    # evaluator based on complete data set (train + test) generated topics --> for topic modeling evaluation metrics
    npmi_evaluator = Evaluator(model,
                               text_dataset.vocab,
                               text_dataset.counts,
                               text_dataset.labels,
                               text_dataset.docs,
                               model_name=FLAGS.model)

    # perplexity =  measure of how well the probability model predicts the sample
    perplexity = evaluator.get_perplexity()

    # normalized pointwise mutual information (NPMI) = measure of the association between two words that takes into
    # account the frequency of each word in the corpus
    npmi = npmi_evaluator.get_normalized_pmi_df()

    print("Perplexity:", perplexity)
    print("NPMI:", npmi)

    # Print top words of each topic (i.e. 'neutral' words) - topics_str contains a latex representation
    print('Top words of each topic (neutral words)')
    topics_str = evaluator.visualize_topics(format_pretty=True, num_words=7)

    # Both individual words and topics influence the outcome variable
    # Print most positively and negatively influencing words - bow_str contains a latex representation
    print('Most positively and negatively influencing words on outcome by topic')
    bow_str = evaluator.visualize_word_weights(num_words=7)

    # Print topics with positive and negative influence
    if FLAGS.model in {'hstm', 'hstm-all', 'hstm-nobeta'}:
        print('Topic words with positive influence (pro)')
        pos_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=True, format_pretty=True,
                                                               compare_to_bow=False, num_words=7)
        print('Topic words with negative influence (anti)')
        neg_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=False, format_pretty=True,
                                                               compare_to_bow=False, num_words=7)

        pos_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='pos')
        neg_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='neg')
        print("NPMI for positive topics and negative topics:", pos_npmi, neg_npmi)

    os.makedirs(FLAGS.outdir, exist_ok=True)

    # save model metrics to outdir
    np.save(os.path.join(FLAGS.outdir, FLAGS.model + '.result.' + 'split' + str(FLAGS.split) + '.setting=' + str(
        (FLAGS.C, FLAGS.C_topics))),
            np.array([test_mse, test_auc, test_log_loss, test_accuracy, test_precision, test_f1, perplexity, npmi,
                      shuffle_loss]))

    # save info for visualization with pyLDAvis
    visualization_dict = {'topic_term_dists': npmi_evaluator.topics,
                          'doc_topic_dists': npmi_evaluator.theta,
                          'documents': npmi_evaluator.texts,
                          'vocab': npmi_evaluator.vocab,
                          'term_frequency': text_dataset.counts,
                          'topic_weights': npmi_evaluator.topic_weights}

    np.save(os.path.join(FLAGS.outdir, FLAGS.model + '.visualization.' + 'split' + str(FLAGS.split) + '.setting=' + str(
        (FLAGS.C, FLAGS.C_topics))),
            np.array([visualization_dict]))

    # betas (#Topics x #VocabWords)
    # gammas (#VocabWords x #Topics)
    # logit-betas (#VocabWords x #Topics)
    # smoothing (#VocabWords x 1) --> Smoothing??
    # topic-weights (#Topics x 1)

    if FLAGS.print_latex:
        latex = evaluator.get_latex_for_topics(normalize=True, num_words=7, num_topics=num_topics)
        print('\n', latex)


def main_without_flags(data, num_topics, column, num_folds, split, batch_size, model_file, C, C_Topics,
                       max_features, max_df, epochs, extra_epochs, printDetailedResult=False, activation='tanh',
                       save_visualization=False):
    """
    this function allows the script to be run from another python script.

    the following hyperparameters can be set:
    - num_topics: number of topics
    - C:
    - C_Topics:
    - max_features: build a vocabulary that only considers the top max_features ordered by term frequency across the corpus
    - num_folds: number of folds in k-cross-validation
    - max_df: when building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words)
    """

    print("Running model for", column, '..' * 20)

    # data set used
    base_dataset = data

    # configure path to save processed data
    processed_file = '../dat/proc/' + base_dataset + '_' + column + '_' + str(max_features) + '_' + str(max_df) + '_proc.npz'

    # set number of topics for training
    num_topics = num_topics

    label_is_bool = True

    beta = None

    text_dataset = TextResponseDataset(data, column, "", processed_file)

    # transform input data to document-term matrix
    text_dataset.process_dataset(max_df, max_features)

    # normalize counts
    text_dataset.preprocessing()

    # set total number of documents in data set
    total_docs = text_dataset.get_full_size()

    # setup cross-validation
    split_indices = util.cross_val_splits(total_docs, num_splits=num_folds)
    all_indices = np.arange(total_docs)
    test_set_indices = split_indices[split]
    train_set_indices = np.setdiff1d(all_indices, test_set_indices)
    text_dataset.assign_splits(train_set_indices, test_set_indices)

    # setup training parameters
    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    training_dataloader = DataLoader(text_dataset, **train_params)
    vocab_size = text_dataset.get_vocab_size()
    n_docs = len(text_dataset)

    model = adjusted_hstm.HeterogeneousSupervisedTopicModel(num_topics, vocab_size, n_docs, theta_act=activation,
                                                            label_is_bool=label_is_bool, beta_init=beta,
                                                            C_weights=C, C_topics=C_Topics,
                                                            response_model="hstm-all")

    trainer = ModelTrainer(model,
                           model_name="hstm-all",
                           use_pretrained=False,
                           do_pretraining_stage=False,
                           do_finetuning=False,
                           save=False,
                           load=False,
                           model_file=model_file)

    trainer.train(training_dataloader, epochs=epochs, extra_epochs=extra_epochs)

    # Negative log likelihood
    test_nll = trainer.evaluate_heldout_nll(text_dataset.test_set_counts, theta=text_dataset.test_set_pretrained_theta)
    print("Held out neg. log likelihood:", test_nll)

    test_err, predictions = trainer.evaluate_heldout_prediction(text_dataset.test_set_counts,
                                                                text_dataset.test_set_labels,
                                                                theta=text_dataset.test_set_pretrained_theta)

    test_mse = perplexity = npmi = shuffle_loss = 0.0
    test_auc = test_accuracy = test_log_loss = test_precision = test_f1 = 0.0
    test_auc = test_err[0]  # auc = area under curve
    test_log_loss = test_err[1]
    test_accuracy = test_err[2]
    test_precision = test_err[3]
    test_f1 = test_err[4]

    print("AUC on test set:", test_auc, "Log loss:", test_log_loss)
    print("Accuracy:", test_accuracy, "Precision:", test_precision, "F1:", test_f1)

    # evaluation based on test set
    evaluator = Evaluator(model,
                          text_dataset.vocab,
                          text_dataset.test_set_counts,
                          text_dataset.test_set_labels,
                          text_dataset.test_set_docs,
                          model_name="hstm-all")

    # evaluation based on complete set (train + test)
    npmi_evaluator = Evaluator(model,
                               text_dataset.vocab,
                               text_dataset.counts,
                               text_dataset.labels,
                               text_dataset.docs,
                               model_name="hstm-all")

    # perplexity =  measure of how well the probability model predicts the sample
    perplexity = evaluator.get_perplexity()

    # normalized pointwise mutual information (NPMI) = measure of the association between two words that takes into
    # account the frequency of each word in the corpus
    npmi = npmi_evaluator.get_normalized_pmi_df()

    print("Perplexity:", perplexity)
    print("NPMI:", npmi)

    if (printDetailedResult):
        # Print top words of each topic (i.e. 'neutral' words) - topics_str contains a latex representation
        print('Top words of each topic (neutral words)')
        topics_str = evaluator.visualize_topics(format_pretty=True, num_words=7)

        # Both individual words and topics influence the outcome variable
        # Print most positively and negatively influencing words - bow_str contains a latex representation
        print('Most positively and negatively influencing words on outcome by topic')
        bow_str = evaluator.visualize_word_weights(num_words=7)

        # Print topics with positive and negative influence on outcome
        print('Topic words with positive influence (pro)')
        pos_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=True, format_pretty=True,
                                                               compare_to_bow=False, num_words=7)
        print('Topic words with negative influence (anti)')
        neg_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=False, format_pretty=True,
                                                               compare_to_bow=False, num_words=7)

        pos_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='pos')
        neg_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='neg')
        print("NPMI for positive topics and negative topics:", pos_npmi, neg_npmi)

    # save model metrics to outdir
    os.makedirs("../out/", exist_ok=True)
    np.save(os.path.join("../out/", 'hstm-all' + '.result.' + 'split' + str(split) + '.setting=' + str(
        (column, C, C_Topics, num_topics, max_df, max_features))),
            np.array([test_mse, test_auc, test_log_loss, test_accuracy, test_precision, test_f1, perplexity, npmi,
                      shuffle_loss]))

    # save info for visualization with pyLDAvis
    visualization_dict = {'topic_term_dists': npmi_evaluator.topics,
                          'doc_topic_dists': npmi_evaluator.theta,
                          'documents': npmi_evaluator.texts,
                          'vocab': npmi_evaluator.vocab,
                          'term_frequency': text_dataset.counts,
                          'topic_weights': npmi_evaluator.topic_weights}

    if save_visualization:
        np.save(os.path.join("../out/", 'hstm-all' + '.visualization.' + 'split' + str(split) + '.setting=' + str(
        (column, C, C_Topics, num_topics, max_df, max_features))),
            np.array([visualization_dict]))

    latex = evaluator.get_latex_for_topics(normalize=True, num_words=7, num_topics=num_topics)

    print(latex)


if __name__ == '__main__':
    FLAGS = absl.flags.FLAGS
    absl.flags.DEFINE_string('model', 'hstm-all', "type of response model.")
    absl.flags.DEFINE_string("datafile", "", "path to file if using raw data files.")
    absl.flags.DEFINE_string("column", "from_idea_to_product", "path")
    absl.flags.DEFINE_string("procfile", "", "path to file for processed data.")
    absl.flags.DEFINE_string("pretraining_file", "", "path to pretrained data.")
    absl.flags.DEFINE_string("data", "my_dataset", "name of text corpus.")
    absl.flags.DEFINE_string("outdir", "../out/", "directory to which to write outputs.")
    absl.flags.DEFINE_string("model_file", "../out/model/hstm-all.myDataset", "file where model is saved.")

    absl.flags.DEFINE_float("C", 1e-6, "l1 penalty for BoW weights and base rates.")
    absl.flags.DEFINE_float("C_topics", 1e-6, "l1 penalty for gammas.")

    absl.flags.DEFINE_integer("train_size", 10000, "number of samples to set aside for training split (only valid if "
                                                   "train/test setting is used)")
    absl.flags.DEFINE_integer("num_topics", 20, "number of topics to use.")
    absl.flags.DEFINE_integer("batch_size", 512, "batch size to use in training.")  # changed from 512
    absl.flags.DEFINE_integer("split", 0, "split to use as the test data in cross-fold validation.")
    absl.flags.DEFINE_integer("num_folds", 10,
                              "number of splits for cross-fold validation (i.e. K in K-fold CV).")
    absl.flags.DEFINE_integer("epochs", 10, "number of epochs for training.")
    absl.flags.DEFINE_integer("extra_epochs", 10, "number of extra epochs to train supervised model.")
    absl.flags.DEFINE_boolean("train_test_mode", False,
                              "flag to use to run a train/test experiment instead of cross validation (default).")
    absl.flags.DEFINE_boolean("pretrained", True, "flag to use pretrained LDA topics or not.")
    absl.flags.DEFINE_boolean("pretrained_prodlda", False, "flag to use pretrained ProdLDA topics or not.")
    absl.flags.DEFINE_boolean("do_pretraining_stage", False, "flag to run sgd steps for topic model only.")
    absl.flags.DEFINE_boolean("do_finetuning", False, "flag to run sgd steps for response model only.")
    absl.flags.DEFINE_boolean("save", False, "flag to save model.")
    absl.flags.DEFINE_boolean("load", False, "flag to load saved model.")
    absl.flags.DEFINE_boolean("print_latex", False, "flag to print latex for tables.")
    absl.flags.DEFINE_float("max_df", 0.8,
                            "ignore terms that have a df strictly higher than given threshold (i.e. corpus-specific stop words)")

    absl.app.run(main)
