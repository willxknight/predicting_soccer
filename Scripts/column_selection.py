# From: https://www.kaggle.com/code/jocelynallen/starter-european-soccer-database-5bfba1ad-d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3

def main():
    # Import our database using global path
    path = '/Users/will_knight/Desktop/CS 334/Project/database.sqlite'
    database = sqlite3.connect(path)
    # store tables in dataframes
    match = pd.read_sql_query('SELECT * from Match', database)
    country = pd.read_sql_query('SELECT * from Country', database) 
    league = pd.read_sql_query('SELECT * from League', database) 
    team = pd.read_sql_query('SELECT * from Team', database) 
     # We now have each team from each game inside our dataframe
    database.close()
    # country.shape = (11, 2) -- id, name
    # match.shape = (25979, 115)
    # league.shape = (11, 3) -- id, country_id, name
    # team.shape (299, 5) -- team_api_id, team_fifa_api_id, team_fifa_api_id, team_short_name
    # From Team Dataset: Want Team's name, Can drop team_api_id, team_fifa_api_id, team_short_name
    # Dropping Columns Help From: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
    team = team.drop(['team_fifa_api_id', 'team_short_name'], axis = 1)
    # Match Dataset Columns: id, country_id, league_id, season, stage, date, home_team_goal, away_team_goal
    # players for both teams, home_team_api_id, away_team_api_id, goal
    match = match[['id', 'country_id', 'league_id', 'season', 'date', 'home_team_goal', 'away_team_goal', 'home_team_api_id', 'away_team_api_id', 'goal']]
    # match.columns = ['id', 'country_id', 'league_id', 'season', 'date', 'home_team_goal', 'away_team_goal', 'home_team_api_id', 'away_team_api_id', 'goal']
    # Need to merge this with the other databases
    # First, lets pull in home_team_api_id which we can merge with team_api_id -- first rename
    match = match.rename(columns={'id': 'match_id'})
    team_home = team.rename(columns={'team_api_id': 'home_team_api_id', 'team_long_name': 'home_team_name'})
    team_away = team.rename(columns={'team_api_id': 'away_team_api_id', 'team_long_name': 'away_team_name'})
    # Now merge on the same column name
    match = pd.merge(match, team_home, on="home_team_api_id")
    match = pd.merge(match, team_away, on="away_team_api_id")
    # Now, we're going to create a column which includes who won the game, or the result of the game
    # Help From: https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
    match['winner'] = np.where(match['home_team_goal'] > match['away_team_goal'], 'Home', 'Away')
    # Now go and convert all the matches where they tied into ties as well w/o changing others
    match['winner'] = np.where(match['home_team_goal'] == match['away_team_goal'], 'Tie', match['winner'])
    # Now lets make a column that records the goal difference between the two teams
    match['goal_difference'] = np.where(match['winner'] == 'Home', match['home_team_goal'] - match['away_team_goal'], 0)
    match['goal_difference'] = np.where(match['winner'] == 'Away', match['away_team_goal'] - match['home_team_goal'], match['goal_difference'])
    match['goal_difference'] = np.where(match['winner'] == 'Tie', 0, match['goal_difference'])
    # We can now drop the goal columns as the winner is the target, not predicting how many goals were scored
    match = match.drop(['id_x', 'id_y', 'home_team_goal', 'away_team_goal', 'goal'], axis = 1)
    # We still need columns for team ID, as this will be used to group together the teams so we can extract recent performance
    # match.columns = ['match_id', 'country_id', 'league_id', 'season', 'date', 'home_team_api_id', 'away_team_api_id', 'home_team_name', 'away_team_name', 'winner', 'goal_difference']
    # match.shape = (25979, 11)
    # We now have a working database with the goal difference and winner included, lets export to a CSV file
    # Help From: https://towardsdatascience.com/how-to-export-pandas-dataframe-to-csv-2038e43d9c03
    match.to_csv('raw_combined_data.csv', index = False)

if __name__ == "__main__":
    main()

