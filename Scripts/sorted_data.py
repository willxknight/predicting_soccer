# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
# import sqlite3

def main(): 
    
    # import data into df
    match_data = pd.read_csv('raw_combined_data.csv')
    
    # basic summary stats
    print(match_data.describe())
    # summary stats on object types
    # notes: home team wins more frequently, unique home & away team counts 
    #       are the same
    print(match_data.describe(include = object))

    # ------------------------------------------------------------------- 
    # summary stats on goal differences 
    mean_goal_diff = match_data['goal_difference'].mean()
    min_goal_diff = match_data['goal_difference'].min()
    max_goal_diff = match_data['goal_difference'].max()
    stdev_goal_diff = match_data['goal_difference'].std()

    print("mean goal difference: ", mean_goal_diff) # 1.36329
    print("minimum goal diffference: ", min_goal_diff) # 0
    print("maximum goal difference: ", max_goal_diff) # 10
    print("goal diffeence standard deviation: ", stdev_goal_diff) # 1.2106

    # # correlation matrix -- doesn't really tell us anything for our purposes
    # result = df.corr(method = 'pearson')
    # sns.heatmap(result)
    # plt.show()

    # ------------------------------------------------------------------- 
    # team summary stats 
    team_data = match_data[['home_team_api_id', 'home_team_name', 
                            'away_team_api_id', 'away_team_name', 
                            'winner', 'goal_difference']]
    
    # names = team_data['home_team_name'].unique() # team names
    
    team_dict = {}
    
    for team in team_data['home_team_api_id'].unique(): # iterate through teams
        # name = team_data[~team_data['home_team_name'].where(team_data['home_team_api_id'] == team).isnull()]
        # print(name)
        
        wins_home  = len(team_data[(team_data['home_team_api_id'] == team) & (team_data['winner'] == 'Home')])
        
        wins_away = len(team_data[(team_data['away_team_api_id'] == team) & (team_data['winner'] == 'Away')])
         
        losses_home = len(team_data[(team_data['home_team_api_id'] == team) & (team_data['winner'] == 'Away')])
        
        losses_away = len(team_data[(team_data['away_team_api_id'] == team) & (team_data['winner'] == 'Home')])
        
        ties_home = len(team_data[(team_data['home_team_api_id'] == team) & (team_data['winner'] == 'Tie')])
    
        ties_away = len(team_data[(team_data['away_team_api_id'] == team) & (team_data['winner'] == 'Tie')])
    
        total_games = wins_home + wins_away + losses_away + losses_home + ties_home + ties_away
        
        team_dict[team] = { 'team_id': team,
                            'percent_wins': (wins_home + wins_away) / total_games,
                            'total_wins': wins_home + wins_away,
                            'wins_home': wins_home, 
                            'wins_away': wins_away, 
                            'total_losses': losses_home + losses_away,
                            'losses_home': losses_home,
                            'losses_away': losses_away, 
                            'total_ties': ties_home + ties_away,
                            'ties_home': ties_home, 
                            'ties_away': ties_away}
    
    team_stats = pd.DataFrame(team_dict).T
    # print(team_stats) # doesn't include team names, can't figure out the join rn
    # team_stats = pd.merge(team_stats, team_data[['home_team_name']], 
    #                       how = 'right',
    #                       left_on = team_stats['team_id'],
    #                       right_on = team_data['home_team_api_id'])

    # ------------------------------------------------------------------- 
    
    # PRINT/GRAPH TEAMS WITH TOP/LOWEST PERCENTAGE WINS
    # MORE QUERIES TO SHOW HOW MANY DRAWS, AND HOW MANY OF THOSE WINS WERE PLAYED AT HOME

    # ------------------------------------------------------------------- 
    # get most recent matches
    format_str = "%Y-%m-%d %H:%M:%S"
    dt_obj = match_data['date'].apply(lambda x: datetime.strptime(x, format_str))
    # extract date
    date = [d.date() for d in dt_obj]
    match_data.insert(loc = 0, column = 'match date', value = date)
    match_data = match_data.drop(columns=['date'])
    home_match_data_sorted = match_data.groupby(['home_team_api_id']).apply(lambda x: x.sort_values(['match date'], ascending = False))
    home_result = pd.DataFrame(home_match_data_sorted)
    home_result.to_csv('home_match_data.csv', index = False)
    away_match_data_sorted = match_data.groupby(['away_team_api_id']).apply(lambda x: x.sort_values(['match date'], ascending = False))
    away_result = pd.DataFrame(away_match_data_sorted)
    away_result.to_csv('away_match_data.csv', index = False)
    
    
if __name__ == "__main__":
    main()
