import run_experiment


# This python script tunes the hyperparameters for the specified model and corresponding data set.
# Results for all combinations of hyperparameters are saved to the out-dir.
# Run this script from the src folder or set working directory accordingly in the run config.

def main():
    # Define static parameters
    data = "my_dataset"
    batch_size = 512

    # Define varying parameters
    model_file = "../out/model/hstm-all.myDataset"
    num_folds = 10
    split = 0
    epochs = 10
    extra_epochs = 10

    # Define Parameter Values
    columns = {'from_idea_to_product', 'idea_backstory_motivation',
               'marketing_strategies', 'background_and_current_work',
               'challenges_obstacles_mistakes', 'main_lessons_and_advice'}
    Cs = {1e-4, 5e-5, 1e-5, 5e-6, 1e-6}  # l1 penalty for BoW weights and base rates
    C_Topics = {1e-4, 5e-5, 1e-5, 5e-6, 1e-6}  # l1 penalty for gammas
    Num_Topics = {10, 20, 30}
    Max_DF = {0.5, 0.6, 0.7, 0.8}
    Max_Features = {500, 1000, 2000}

    # Iterate over different options
    for column_i in columns:
        for c in Cs:
            for c_topic in C_Topics:
                for num_topics in Num_Topics:
                    for max_feature in Max_Features:
                        for max_df in Max_DF:
                            print('Column: ' + column_i + ', C: ' + str(c)
                                  + ', C_Topics: ' + str(c_topic) + ', #Topics: ' + str(num_topics)
                                  + ', Max_Feature: ' + str(max_feature) + ', TF_IDF: ' + str(max_df))

                            run_experiment.main_without_flags(data, num_topics, column_i, num_folds, split, batch_size,
                                                              model_file, c, c_topic, max_feature, max_df, epochs,
                                                              extra_epochs)

                            print('..' * 50)


if __name__ == '__main__':
    main()