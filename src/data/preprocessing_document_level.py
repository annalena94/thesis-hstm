import pandas as pd
from absl import app

# This python script preprocesses the interview data scraped from https://www.failory.com/interviews
# The repository contains the data produced by this script in the dat folder.
# Run this script from the src folder or set working directory accordingly in the run config.


def main(args):
    df = pd.read_csv('../dat/startup_data_final.csv')

    # Step 1: drop columns
    df = df[['startup', 'background_and_current_work',
             'causes_of_failure', 'challenges_obstacles_mistakes',
             'current_state_goals_future', 'disadvantages', 'from_idea_to_product',
             'growth_of_startup_and_customers', 'idea_backstory_motivation',
             'main_lessons_and_advice', 'marketing_strategies',
             'revenue_expenses_losses', 'competition', 'more_infos_about_startup', 'recommended_resources',
             'outcome_numeric']]

    print(df)
    print(df.isnull().sum(axis=0))
    # Step 2: merge column values together
    df = df.reset_index()
    df_all = pd.DataFrame({'text': pd.Series(dtype='str'),
                       'outcome_numeric': pd.Series(dtype='int')})

    for index, row in df.iterrows():
        text_all = str(row['background_and_current_work']) + str(row['causes_of_failure']) + str(row['challenges_obstacles_mistakes']) + \
                   str(row['current_state_goals_future']) + str(row['disadvantages']) + str(row['from_idea_to_product']) +\
                   str(row['growth_of_startup_and_customers']) + str(row['idea_backstory_motivation']) + str(row['main_lessons_and_advice']) + \
                   str(row['marketing_strategies']) + str(row['revenue_expenses_losses'])

        outcome = row['outcome_numeric']
        data = {'text': [text_all], 'outcome_numeric': [outcome]}
        new_row = pd.DataFrame(data)
        df_all = pd.concat([df_all, new_row])

    df_all.to_csv('../dat/startup_data_final_all_text.csv', header=True)

if __name__ == '__main__':
    app.run(main)