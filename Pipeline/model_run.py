import numpy as numpy
import pandas as pd
import logging
import sys
import datetime
import yaml
from modeling import models
from modeling import evaluation
import json
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import train_test_split



#conn = setup_environment.set_connection('2. Pipeline/config.json')

def load_config(config_file_path):
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.load(f)
    except:
        raise 'cannot read config file'
    return config

def grab_col_list(dic):
    rv = []
    for k, v in dic.items():
        if v:
            rv.append(k)
    return rv

def clf_loop(models_to_run, grid, label, cols_to_use, raw_data):
    result_dictionary_list=[]
    training_data, testing_data = train_test_split(raw_data, test_size=0.3)
    for index, clf in enumerate(models_to_run):
        model_name = models_to_run[index]
        model_params = grid[model_name]
        for p in ParameterGrid(model_params):
            try:
                model = models.Model(model_name, p, label, cols_to_use, training_data, testing_data)
                result_dictionary = model.run()
                result_dictionary_list.append(result_dictionary)
            except IndexError as e:
                    print ('Error:',e)
                    continue

    result_df = pd.DataFrame(result_dictionary_list)
    result_df.to_csv('output/model_results_{}.csv'.format(datetime.datetime.now()))
    return result_df


def main(config_file_path, data_path):
    config = load_config(config_file_path)
    models_to_run = grab_col_list(config['models'])
    grid = config['parameters']
    label = config['label']
    cols_to_use = grab_col_list(config['features'])
    raw_data = pd.read_csv(data_path)
    result_df = clf_loop(models_to_run, grid, label, cols_to_use, raw_data)
    return result_df

if __name__=="__main__":
    if len(sys.argv) != 3:
        print('Input format: python run.py <config file path> <data file path> ') 
        sys.exit(1)

    config_file_path = sys.argv[1]
    data_path = sys.argv[2]
    main(config_file_path, data_path)
    


