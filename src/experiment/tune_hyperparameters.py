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
    max_df = 0.8
    num_folds = 10
    split = 0
    epochs = 20
    extra_epochs = 10

    # Define Parameter Values
    columns = {'from_idea_to_product', 'idea_backstory_motivation',
               'marketing_strategies', 'background_and_current_work', 'challenges_obstacles_mistakes'}
    Cs = {1e-4, 5e-5, 1e-5, 5e-6, 1e-6 } # l1 penalty for BoW weights and base rates
    C_Topics = {1e-4, 5e-5, 1e-5, 5e-6, 1e-6 } # l1 penalty for gammas.
    Num_Topics = {3, 5, 10, 15, 20, 25, 30 }

    # Iterate over different options
    for column_i in columns:
        for c in Cs:
            for c_topic in C_Topics:
                for num_topics in Num_Topics:

                    print('Column: ' + column_i + ', C: ' + str(c)
                          + ', C_Topics: ' + str(c_topic) + ', #Topics: ' + str(num_topics ))

                    run_experiment.main_without_flags(data, num_topics, column_i, num_folds, split, batch_size, model_file,
                                          c, c_topic, epochs, extra_epochs)

                    print('..' * 50)


if __name__ == '__main__':
    main()