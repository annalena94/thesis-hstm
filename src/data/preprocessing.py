import pandas as pd
from absl import app

# This python script preprocesses the interview data scraped from https://www.failory.com/interviews
# The repository contains the data produced by this script in the dat folder.
# Run this script from the src folder or set working directory accordingly in the run config.

def main(args):
    df = pd.read_csv('../dat/startup_data_final.csv')
    gender = pd.read_csv('../dat/gender.csv')
    columns = ['from_idea_to_product',
               'growth_of_startup_and_customers',
               'idea_backstory_motivation',
               'marketing_strategies',
               'background_and_current_work',
               'challenges_obstacles_mistakes']

    for column in columns:
        df_reduced = df[[column, 'outcome_numeric']]
        df_reduced.insert(2, "gender", gender['gender'])

        # Drop all rows with NaN values
        df_reduced = df_reduced.dropna()

        # Reset index after drop
        df_reduced = df_reduced.dropna().reset_index(drop=True)

        # Write to csv
        df_reduced.to_csv('../dat/' + column + '.csv', header=True)

if __name__ == '__main__':
    app.run(main)

