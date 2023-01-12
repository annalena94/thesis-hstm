import absl
from src.model import adjusted_hstm, topic_model, supervised_lda as slda
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

    # configure path to look for pretrained data
    pretraining_file = FLAGS.pretraining_file

    # data set used
    base_dataset = FLAGS.data

    base_pretraining = '_pretraining.npz'
    if FLAGS.pretrained_prodlda:
        base_pretraining = '_prodlda_pretraining.npz'

    if pretraining_file == "":
        pretraining_file = '../dat/proc/' + base_dataset + base_pretraining
    if processed_file == "":
        processed_file = '../dat/proc/' + base_dataset + '_proc.npz'

    num_topics = FLAGS.num_topics

    label_is_bool = True

    print("Running model", FLAGS.model, '..' * 20)

    if FLAGS.pretrained or FLAGS.pretrained_prodlda or FLAGS.model == 'hstm-all-2stage':
        array = np.load(pretraining_file)
        beta = np.log(array['beta']).T
    else:
        beta = None

    if FLAGS.model == 'hstm-all-2stage':
        text_dataset = TextResponseDataset(FLAGS.data, FLAGS.column, FLAGS.datafile, processed_file, pretrained_theta=theta_pretrained)
    else:
        text_dataset = TextResponseDataset(FLAGS.data, FLAGS.column, FLAGS.datafile, processed_file)

    text_dataset.process_dataset()
    text_dataset.preprocessing()

    # total number of documents in data set
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
    # TODO AF: add other variables to the DataLoader through the text_dataset
    training_dataloader = DataLoader(text_dataset, **train_params)
    vocab_size = text_dataset.get_vocab_size()
    n_docs = len(text_dataset)

    if FLAGS.model == 'prodlda':
        model = topic_model.TopicModel(num_topics, vocab_size, n_docs, c=FLAGS.C, beta_init=beta)

    elif FLAGS.model == 'slda':
        model = slda.SupervisedLDA(num_topics,
                                   vocab_size,
                                   n_docs,
                                   label_is_bool=label_is_bool,
                                   beta_init=beta,
                                   predict_with_z=True)
    else:
        model = adjusted_hstm.HeterogeneousSupervisedTopicModel(num_topics, vocab_size, n_docs,
                                                                label_is_bool=label_is_bool, beta_init=beta,
                                                                C_weights=FLAGS.C, C_topics=FLAGS.C_topics,
                                                                response_model=FLAGS.model)

    trainer = ModelTrainer(model,
                           model_name=FLAGS.model,
                           use_pretrained=(FLAGS.pretrained or FLAGS.pretrained_prodlda),
                           do_pretraining_stage=FLAGS.do_pretraining_stage,
                           do_finetuning=FLAGS.do_finetuning,
                           save=FLAGS.save,
                           load=FLAGS.load,
                           model_file=FLAGS.model_file)

    trainer.train(training_dataloader, epochs=FLAGS.epochs, extra_epochs=FLAGS.extra_epochs)

    # Negative log likelihood
    test_nll = trainer.evaluate_heldout_nll(text_dataset.test_set_counts, theta=text_dataset.test_set_pretrained_theta)

    print("Held out neg. log likelihood:", test_nll)

    if FLAGS.model != 'prodlda':
        test_err, predictions = trainer.evaluate_heldout_prediction(text_dataset.test_set_counts,
                                                                    text_dataset.test_set_labels,
                                                                    theta=text_dataset.test_set_pretrained_theta)

        test_mse = perplexity = npmi = shuffle_loss = 0.0
        test_auc = test_accuracy = test_log_loss = 0.0

        test_auc = test_err[0] # auc = area under curve
        test_log_loss = test_err[1]
        test_accuracy = test_err[2]

        print("AUC on test set:", test_auc, "Log loss:", test_log_loss, "Accuracy:", test_accuracy)

    # evaluator based on test set
    evaluator = Evaluator(model,
                          text_dataset.vocab,
                          text_dataset.test_set_counts,
                          text_dataset.test_set_labels,
                          text_dataset.test_set_docs,
                          model_name=FLAGS.model)

    # evaluator based on complete set (train + test)
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

    # Print top words of each topic
    topics_str = evaluator.visualize_topics(format_pretty=True, num_words=7)

    # Print most positively and negatively influencing words
    if FLAGS.model in {'hstm-all', 'stm+bow', 'hstm-nobeta'}:
        bow_str = evaluator.visualize_word_weights(num_words=7)
    else:
        bow_str = "No BoW weights to report."

    # Print topics with positive and negative influence
    if FLAGS.model in {'hstm', 'hstm-all', 'hstm-nobeta'}:
        pos_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=True, format_pretty=True,
                                                               compare_to_bow=False, num_words=7)
        neg_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=False, format_pretty=True,
                                                               compare_to_bow=False, num_words=7)

        pos_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='pos')
        neg_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='neg')
        print("NPMI for positive topics and negative topics:", pos_npmi, neg_npmi)

    os.makedirs(FLAGS.outdir, exist_ok=True)

    # save model to outdir
    np.save(os.path.join(FLAGS.outdir, FLAGS.model + '.result.' + 'split' + str(FLAGS.split) + '.setting=' + str(
        (FLAGS.C, FLAGS.C_topics))),
            np.array([test_mse, test_auc, test_log_loss, test_accuracy, perplexity, npmi, shuffle_loss]))

    if FLAGS.print_latex:
        latex = evaluator.get_latex_for_topics(normalize=True, num_words=7, num_topics=5)
        print('\n', latex)

        print('\n', bow_str)

    # TODO AF: print results to readable document


def main_without_flags(data, num_topics, column, num_folds, split, batch_size, model_file, C, C_Topics,
                       epochs, extra_epochs, printDetailedResult=False):
    """
    this function allows the script to be run from another python script
    """

    # data set used
    base_dataset = data

    # configure path to save processed data
    processed_file = '../dat/proc/' + base_dataset + '_proc.npz'

    num_topics = num_topics

    label_is_bool = True

    beta = None

    text_dataset = TextResponseDataset(data, column, "", processed_file)

    text_dataset.process_dataset()
    text_dataset.preprocessing()

    # total number of documents in data set
    total_docs = text_dataset.get_full_size()


    split_indices = util.cross_val_splits(total_docs, num_splits=num_folds)
    all_indices = np.arange(total_docs)
    test_set_indices = split_indices[split]
    train_set_indices = np.setdiff1d(all_indices, test_set_indices)

    text_dataset.assign_splits(train_set_indices, test_set_indices)

    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
    training_dataloader = DataLoader(text_dataset, **train_params)
    vocab_size = text_dataset.get_vocab_size()
    n_docs = len(text_dataset)

    model = adjusted_hstm.HeterogeneousSupervisedTopicModel(num_topics, vocab_size, n_docs,
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

    # Negative log likelihood (of test set?)
    test_nll = trainer.evaluate_heldout_nll(text_dataset.test_set_counts, theta=text_dataset.test_set_pretrained_theta)

    print("Held out neg. log likelihood:", test_nll)

    test_err, predictions = trainer.evaluate_heldout_prediction(text_dataset.test_set_counts,
                                                                    text_dataset.test_set_labels,
                                                                    theta=text_dataset.test_set_pretrained_theta)

    test_mse = perplexity = npmi = shuffle_loss = 0.0
    test_auc = test_accuracy = test_log_loss = 0.0
    test_auc = test_err[0] # auc = area under curve
    test_log_loss = test_err[1]
    test_accuracy = test_err[2]

    print("AUC on test set:", test_auc, "Log loss:", test_log_loss, "Accuracy:", test_accuracy)

    # evaluator based on test set
    evaluator = Evaluator(model,
                          text_dataset.vocab,
                          text_dataset.test_set_counts,
                          text_dataset.test_set_labels,
                          text_dataset.test_set_docs,
                          model_name="hstm-all")

    # evaluator based on complete set (train + test)
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

    if(printDetailedResult):
        topics_str = evaluator.visualize_topics(format_pretty=True, num_words=7)

        # Print most positively and negatively influencing words
        bow_str = evaluator.visualize_word_weights(num_words=7)

        # Print topics with positive and negative influence
        pos_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=True, format_pretty=True, compare_to_bow=False, num_words=7)
        neg_topics_str = evaluator.visualize_supervised_topics(normalize=True, pos_topics=False, format_pretty=True, compare_to_bow=False, num_words=7)

        pos_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='pos')
        neg_npmi = npmi_evaluator.get_normalized_pmi_df(topics_to_use='neg')
        print("NPMI for positive topics and negative topics:", pos_npmi, neg_npmi)

    os.makedirs("../out/", exist_ok=True)
    np.save(os.path.join("../out/", 'hstm-all' + '.result.' + 'split' + str(split) + '.setting=' + str(
        (column, C, C_Topics, num_topics))),
            np.array([test_mse, test_auc, test_log_loss, test_accuracy, perplexity, npmi, shuffle_loss]))

    latex = evaluator.get_latex_for_topics(normalize=True, num_words=7, num_topics=num_topics)
    if(printDetailedResult):
        print('\n', latex)
        print('\n', bow_str)

if __name__ == '__main__':
    FLAGS = absl.flags.FLAGS
    absl.flags.DEFINE_string('model', 'hstm-all', "type of response model.")
    absl.flags.DEFINE_string("datafile", "", "path to file if using raw data files.")
    absl.flags.DEFINE_string("column", "from_idea_to_product", "path")
    absl.flags.DEFINE_string("procfile", "", "path to file for processed data.")
    absl.flags.DEFINE_string("pretraining_file", "", "path to pretrained data.")
    absl.flags.DEFINE_string("data", "my_dataset", "name of text corpus.")
    absl.flags.DEFINE_string("outdir", "../out/", "directory to which to write outputs.")
    # TODO: more sopisticated naming for model_file (doesn't work)
    absl.flags.DEFINE_string("model_file", "../out/model/hstm-all.myDataset", "file where model is saved.")

    absl.flags.DEFINE_float("C", 1e-6, "l1 penalty for BoW weights and base rates.")
    absl.flags.DEFINE_float("C_topics", 1e-6, "l1 penalty for gammas.")

    absl.flags.DEFINE_integer("train_size", 10000, "number of samples to set aside for training split (only valid if "
                                                   "train/test setting is used)")
    absl.flags.DEFINE_integer("num_topics", 20, "number of topics to use.")
    absl.flags.DEFINE_integer("batch_size", 512, "batch size to use in training.") # changed from 512
    absl.flags.DEFINE_integer("split", 0, "split to use as the test data in cross-fold validation.")
    absl.flags.DEFINE_integer("num_folds", 5, "number of splits for cross-fold validation (i.e. K in K-fold CV).") # changed from 10
    absl.flags.DEFINE_integer("epochs", 10, "number of epochs for training.")
    absl.flags.DEFINE_integer("extra_epochs", 10, "number of extra epochs to train supervised model.")

    absl.flags.DEFINE_boolean("train_test_mode", False,
                              "flag to use to run a train/test experiment instead of cross validation (default).")
    absl.flags.DEFINE_boolean("pretrained", False, "flag to use pretrained LDA topics or not.")
    absl.flags.DEFINE_boolean("pretrained_prodlda", False, "flag to use pretrained ProdLDA topics or not.")
    absl.flags.DEFINE_boolean("do_pretraining_stage", False, "flag to run sgd steps for topic model only.")
    absl.flags.DEFINE_boolean("do_finetuning", False, "flag to run sgd steps for response model only.")
    absl.flags.DEFINE_boolean("save", False, "flag to save model.")
    absl.flags.DEFINE_boolean("load", False, "flag to load saved model.")
    absl.flags.DEFINE_boolean("print_latex", True, "flag to print latex for tables.") # changed
    absl.flags.DEFINE_float("max_df", 0.8,
                            "ignore terms that have a df strictly higher than given threshold (i.e. corpus-specific stop words)")

    absl.app.run(main)
