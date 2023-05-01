import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def main():
    N = 7 # Variable that defines how far back we should look at each team's record
    home_data = pd.read_csv('home_match_data.csv')
    away_data = pd.read_csv('away_match_data.csv')
    raw_data = pd.read_csv('raw_combined_data.csv')
    # home_data.columns = ['match date', 'match_id', 'country_id', 'league_id', 'season', 'home_team_api_id', 'away_team_api_id', 'home_team_name', 'away_team_name', 'winner', 'goal_difference']
    # home_data.shape = (25979, 11) = away_data.shape
    # Create column for win, tie, loss percentages and goal difference
    home_data['home_win_percent'] = 0
    home_data['home_loss_percent'] = 0
    home_data['home_tie_percent'] = 0
    home_data['home_goal_difference'] = 0
    away_data['away_win_percent'] = 0
    away_data['away_loss_percent'] = 0
    away_data['away_tie_percent'] = 0
    away_data['away_goal_difference'] = 0
    # Help From: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    for index, match in home_data.iterrows():
        home_team_id = match['home_team_api_id']
        match_date = match['match date']
        home_record = get_last_matches(raw_data, match_date, home_team_id, N)
        win_number = 0
        loss_number = 0
        tie_number = 0
        goal_difference = 0
        for i, m in home_record.iterrows():
            if (m['winner'] == 'Home'):
                win_number = win_number + 1
                goal_difference = goal_difference + m['goal_difference']
            if (m['winner'] == 'Away'):
                loss_number = loss_number + 1
                goal_difference = goal_difference - m['goal_difference']
            if (m['winner'] == 'Tie'):
                tie_number = tie_number + 1
        win_percent = win_number / N
        loss_percent = loss_number / N
        tie_percent = tie_number / N
        # Help From: https://www.askpython.com/python-modules/pandas/update-the-value-of-a-row-dataframe
        home_data.at[index, 'home_win_percent'] = win_percent
        home_data.at[index, 'home_loss_percent'] = loss_percent
        home_data.at[index, 'home_tie_percent']  = tie_percent
        home_data.at[index, 'home_goal_difference']  = goal_difference
        
    for index_away, match_away in away_data.iterrows():
        away_team_id= match_away['away_team_api_id']
        match_date = match_away['match date']
        away_win_number = 0
        away_loss_number = 0
        away_tie_number = 0
        away_goal_difference = 0
        away_record = get_last_matches(raw_data, match_date, away_team_id, N)
        for i, m in away_record.iterrows():
            if (m['winner'] == 'Home'):
                away_win_number = away_win_number + 1
                away_goal_difference = away_goal_difference + m['goal_difference']
            if (m['winner'] == 'Away'):
                away_loss_number = away_loss_number + 1
                away_goal_difference = away_goal_difference - m['goal_difference']
            if (m['winner'] == 'Tie'):
                away_tie_number = away_tie_number + 1
        away_win_percent = away_win_number / N
        away_loss_percent = away_loss_number / N
        away_tie_percent = away_tie_number / N
        away_data.at[index_away, 'away_win_percent'] = away_win_percent
        away_data.at[index_away, 'away_loss_percent'] = away_loss_percent
        away_data.at[index_away, 'away_tie_percent']  = away_tie_percent
        away_data.at[index_away, 'away_goal_difference']  = away_goal_difference
    # Time to join the matches to merge the home and away new columns etc
    # home_data.columns = 'match date', 'match_id', 'country_id', 'league_id', 'season',
    # 'home_team_api_id', 'away_team_api_id', 'home_team_name',
    #  'away_team_name', 'winner', 'goal_difference', 'home_win_percent',
    #  'home_loss_percent', 'home_tie_percent', 'home_goal_difference'
    # home_data.shape = (25979, 15) = away_data.shape
    # away_data.columns = 'match date', 'match_id', 'country_id', 'league_id', 'season',
    #  'home_team_api_id', 'away_team_api_id', 'home_team_name',
    #  'away_team_name', 'winner', 'goal_difference', 'away_win_percent',
    #  'away_loss_percent', 'away_tie_percent', 'away_goal_difference']
    # Help From: https://www.geeksforgeeks.org/merge-two-pandas-dataframes-by-matched-id-number/
    match_data = pd.merge(home_data, away_data, on='match_id')
    # Want to keep these columns:
    match_data = match_data[['winner_x', 'away_win_percent', 'away_loss_percent', 
                             'away_tie_percent', 'away_goal_difference', 'home_win_percent', 
                             'home_loss_percent', 'home_tie_percent', 'home_goal_difference',]]
    # Quick rename to have our label be distinct
    match_data = match_data.rename(columns={'winner_x': 'winner'})
    # TODO: Convert the winner into a numerical value for prediction -- loss = 0, win = 1?
    match_data['winner'] = np.where(match_data['winner'] == 'Home', 1, match_data['winner'])
    match_data['winner'] = np.where(match_data['winner'] == 'Tie', 0.5, match_data['winner'])
    match_data['winner'] = np.where(match_data['winner'] == 'Away', 0, match_data['winner'])
    # Finally, lets export this to a CSV file
    filename = 'data_n_of_' + str(N) + '.csv'
    match_data.to_csv(filename, index = False)


# Help From: https://www.kaggle.com/code/airback/match-outcome-prediction-in-football/notebook
def get_last_matches(matches, date, team, x):
    ''' Get the last x matches of a given team. '''
    #Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]                  
    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    #Return last matches
    return last_matches
    
if __name__ == "__main__":
    main()
