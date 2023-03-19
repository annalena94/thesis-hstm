import pandas as pd
from absl import app
import numpy as np
import os
import itertools as it

from src.experiment import run_experiment


def main(args):
    # Declaration of metrics needed for processing
    metrics = {'mse': 0, 'auc': 1, 'll': 2, 'acc': 3, 'pre': 4, 'f1': 5, 'perp': 6, 'npmi': 7, 'shuffle': 8}
    column_names = {'from_idea_to_product': 'From Idea To Product', 'idea_backstory_motivation':
                    'Idea, Backstory & Motivation', 'marketing_strategies': 'Marketing Strategies',
                    'background_and_current_work': 'Background and Current Work',
                    'challenges_obstacles_mistakes': 'Challenges, Obstacles & Mistakes',
                    'main_lessons_and_advice': 'Main Lessons & Advice'}
    out = os.path.join('..', 'out')
    models = ['hstm-all']

    datasets = ['my_dataset']
    columns = {'from_idea_to_product', 'idea_backstory_motivation', 'marketing_strategies', 'challenges_obstacles_mistakes',
               'main_lessons_and_advice', 'background_and_current_work', 'marketing_strategies'}
    Cs = {1e-4, 5e-5, 1e-5, 5e-6, 1e-6}  # l1 penalty for BoW weights and base rates
    C_Topics = {1e-4, 5e-5, 1e-5, 5e-6, 1e-6}  # l1 penalty for gammas.
    Num_Topics = {10, 20, 30}
    Max_DF = {0.5, 0.6, 0.7, 0.8}
    Max_Features = {500, 1000, 2000}
    num_folds = 10
    n_metrics = len(metrics.keys())

    # Setting up data that holds results --> 2625 configurations per question category, 15750 overall
    # The exp_results dictionary contains six entries, i.e. for each interview question a corresponding result
    # Each dictionary entry contains itself a dictionary where num(keys) = combination of hyperparameters
    exp_results = {
        column: {'hstm-all': {(C, Ct, num, df, features): np.zeros((num_folds, n_metrics)) for C in Cs for Ct in C_Topics
                              for num in Num_Topics for df in Max_DF for features in Max_Features}}
        for column in columns}

    # Loading results from hyperparameter tuning
    for model in models:
        for column in columns:
            for e_idx in range(num_folds):
                for (C, Ct) in it.product(Cs, C_Topics):
                    for num in Num_Topics:
                        for max_df in Max_DF:
                            for max_feature in Max_Features:
                                filename = get_filename(out, model, e_idx, column, C, Ct, num, max_df, max_feature)
                                results = np.load(filename + '.npy')
                                exp_results[column][model][(C, Ct, num, max_df, max_feature)][e_idx] = results

    # Find hyperparameters that give best results for each column
    #model_settings_dict = {model: {data: None for data in datasets} for model in models}
    #for model in models:
     #   for column in columns:
      #      is_classification = True
       #     model_settings_dict[model][column] = get_best_results(metrics, exp_results, column, model,
        #                                                          is_classification=is_classification)

    # Visualize predictive results for a given data set
    # df = create_results_table(columns, metrics, exp_results, model_settings_dict)
    # print(df)

    # Repeat for results with NPMI
    model_settings_dict_with_npmi = {model: {data: None for data in datasets} for model in models}
    for model in models:
        for column in columns:
            is_classification = True
            model_settings_dict_with_npmi[model][column] = get_best_results(metrics, exp_results, column, model,
                                                                            is_classification=is_classification,
                                                                            withNPMI=True)

    df_with_npmi = create_results_table(columns, metrics, exp_results, model_settings_dict_with_npmi)
    print(df_with_npmi)

    # Save dataframe with the best hyperparameter combinations
    # np.save(os.path.join("../out/", "hyperparam_combinations.npy"), np.array([df]))
    np.save(os.path.join("../out/", "hyperparam_combinations_npmi.npy"), np.array([df_with_npmi]))

    # Run experiment again for best results to create visualizations
    data = "my_dataset"
    batch_size = 512
    model_file = "../out/model/hstm-all.myDataset"
    num_folds = 10
    split = 0
    epochs = 10
    extra_epochs = 10

    # for row in df.values:
    #    (c, c_topics, num_topics, max_df, max_features) = row[5]
     #   column = row[0]
      #  run_experiment.main_without_flags(data, num_topics, column, num_folds, split, batch_size,
       #                                   model_file, c, c_topics, max_features, max_df, epochs,
        #                                  extra_epochs, True, 'tanh', True)

    for row in df_with_npmi.values:
        (c, c_topics, num_topics, max_df, max_features) = row[5]
        column = row[0]
        run_experiment.main_without_flags(data, num_topics, column, num_folds, split, batch_size,
                                          model_file, c, c_topics, max_features, max_df, epochs,
                                          extra_epochs, True, 'tanh', True)


## -------- HELPER FUNCTIONS -------- ##
def get_filename(out, model, e_idx, column, C, C_Topics, num_topics, max_df, max_features):
    # hstm-all.result.split0.setting=('marketing_strategies', 1e-05, 5e-05, 10, 0.8, 2000).npy
    fname = model + '.result.split0' + '.setting=' + str((column, C, C_Topics, num_topics, max_df, max_features))

    return os.path.join(out, fname)


def create_results_table(columns, metrics, exp_results, model_settings_dict):
    data = []
    for column in columns:

        values = [column]
        setting = model_settings_dict['hstm-all'][column]

        average_acc = exp_results[column]['hstm-all'][setting].mean(axis=0)[3]
        values += [str(np.round(average_acc, 2))]

        average_f1 = exp_results[column]['hstm-all'][setting].mean(axis=0)[5]
        values += [str(np.round(average_f1, 2))]

        average_perplexity = exp_results[column]['hstm-all'][setting].mean(axis=0)[6]
        values += [str(np.round(average_perplexity, 2))]

        average_coherence = exp_results[column]['hstm-all'][setting].mean(axis=0)[7]
        values += [str(np.round(average_coherence, 2))]

        values += [setting]
        data.append(values)

    results_df = pd.DataFrame(data, columns=['Question Category', 'Accuracy', 'F1 Score', 'Perplexity', 'Coherence', 'Hyperparameter Setting'])
    return results_df


# This method gets the best result based on the weighted and normalized f1-score and perplexity-score of the topic model
def get_best_results(metrics, exp_results, column, model, is_classification=False, withNPMI=False):
    best_score = 1e7 if not is_classification else 0.
    best_config = None
    metric_idx1 = metrics['f1']
    metric_idx2 = metrics['perp']
    metric_idx3 = metrics['npmi']

    cross_validation_mean_results = {}

    # step 1: calculate average from k-cross-validation
    for config, res in exp_results[column][model].items():
        mean_results = res.mean(axis=0)
        mean_f1 = mean_results[metric_idx1]
        mean_perp = mean_results[metric_idx2]
        mean_npmi = mean_results[metric_idx3]
        cross_validation_mean_results[config] = np.array([mean_f1, mean_perp, mean_npmi])

    # step 2: calculate min and max of metrics to use for normalization
    f1_min = np.min(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 0])
    f1_max = np.max(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 0])
    perp_min = np.min(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 1])
    perp_max = np.max(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 1])
    npmi_min = np.min(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 2])
    npmi_max = np.max(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 2])

    # step 3: normalize f1 score and perplexity
    normalized_results = {}
    for config, res in cross_validation_mean_results.items():
        f1_normalized = (res[0] - f1_min)/(f1_max - f1_min)
        perp_normalized = (res[1] - perp_min)/(perp_max - perp_min)
        npmi_normalized = (res[2] - npmi_min)/(npmi_max - npmi_min)
        normalized_results[config] = np.array([f1_normalized, perp_normalized, npmi_normalized])

    # step 4: set weights
    f1_weight = 0.7
    perp_weight = 0.3

    # step 5: get the best configuration based on weighted metrics
    for config, res in normalized_results.items():
        # perplexity is subtracted as a lower score indicates a better model
        # f1-score is added as a higher score indicates a better model
        if withNPMI:
            current_score = f1_weight * res[0] - perp_weight/2 * res[1] + perp_weight/2 * res[2]
        else:
            current_score = f1_weight * res[0] - perp_weight * res[1]
        if current_score > best_score:
            best_score = current_score
            best_config = config

    # print average metrics of all models for column
    averageF1 = np.mean(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 0])
    averagePerp = np.mean(pd.DataFrame.from_dict(cross_validation_mean_results, orient='index').iloc[:, 1])
    print(column)
    print("Average F1: " + str(averageF1))
    print("Average Perplexity: " + str(averagePerp))
    return best_config

def get_average_coherence_by_topics(exp_results, column, model):
    results = exp_results[column][model]

    # step 1: group results by topic number

    # step 2: average per topic number

    # step 3: return array with tuples [(num_topics, avg_coherence)]

if __name__ == '__main__':
    app.run(main)
