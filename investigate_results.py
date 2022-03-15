import pandas as pd
import os
import sys
import datasets
import nltk
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
import docx
import numpy as np
import datetime
from box import Box
import yaml
from tqdm import tqdm
from prompts import *
from config_parser import *
import matplotlib.pyplot as plt


def generate_graphs(df):
    """
    Generate graphs from the dataframe
    """
    df_unstacked = df.unstack(level=0)
    ax = df_unstacked.plot(kind='bar', rot=0, figsize=(9, 7), layout=(2, 3))
    # set x-axis as df.index.names[1]
    if len(df.index.names) > 1:
        ax.set_xlabel(df.index.names[1])
        # set title
        ax.set_title("{} score for different {} and {}".format(df.keys()[0], df.index.names[0], df.index.names[1]))
    else:
        ax.set_xlabel(df.index.names[0])
        # set title
        ax.set_title("{} score for different {}".format(df.keys()[0], df.index.names[0]))
    # set y-axis as df.keys()[0]
    ax.set_ylabel(df.keys()[0])
    # change Legends names
    names = [name for name in df.index.get_level_values(0).unique()]
    ax.legend(loc='upper center', labels=names)
    plt.show()


    # merge columns with the same name
    # def get_notnull(x): return ';'.join(x[x.notnull()].astype(str))

    # df_clean = df.groupby(level=0, axis=1).apply(lambda x: x.apply(get_notnull, axis=1))

    # change dataframe values from string to int
    # df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

    # models = ['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-1.3B']
    # prompts = ['lyrics_meaning', 'lyrics_meaning_with_metadata', 'song', 'song_with_metadata',
    # 'question_context', 'question_context_with_metadata', None]

    # plot bar plots

def combine_before_and_after(pickle_before, pickle_after):
    df_before = pd.read_pickle(pickle_before)
    df_after = pd.read_pickle(pickle_after)
    # take only relevant columns
    df_before_narrowed = df_before[['decode_method','example_index', 'input_prompt', 'prompt_type', 'model', 'gt_meaning', 'predicted_meaning']]
    after_prompt = df_after['prompt_type'][0]
    df_before_narrowed = df_before_narrowed[df_before_narrowed['prompt_type'] == after_prompt]

    df_before_narrowed = df_before_narrowed[df_before_narrowed['model'] == 'gpt2-medium']

    df_after_narrowed = df_after[['predicted_meaning', 'example_index']]
    # for each example_index in df_before_narrowed, find the corresponding example_index in df_after_narrowed
    # and add the predicted_meaning to the df_before_narrowed
    for index, row in df_before_narrowed.iterrows():
        example_index = row['example_index']
        prediction_after = df_after_narrowed.loc[df_after_narrowed['example_index'] == example_index,
                              'predicted_meaning']
        df_before_narrowed.loc[index, 'predicted_meaning_after'] = prediction_after.values[0]

    # print result to docx file
    doc = docx.Document()
    doc.add_heading('Before and after', 0)
    doc.add_paragraph('The following table shows the predicted meaning before and after training.')
    for index, row in df_before_narrowed.iterrows():
        paragraph = \
            doc.add_paragraph("input prompt:\n{}\n\ngt meaning:\n{}\n\ndecode method:\n{}\n\n".format(
                                    row['input_prompt'],
                                    row['gt_meaning'],
                                    row['decode_method']))
        # highlight the predicted meaning
        paragraph.add_run("predicted meaning before training:\n{})".format(
            row['predicted_meaning'])).bold = True
        paragraph.add_run("\n\npredicted meaning after training:\n{})".format(
            row['predicted_meaning_after'])).font.highlight_color = docx.enum.text.WD_COLOR_INDEX.YELLOW
    doc.save('before_and_after.docx')


if __name__ == '__main__':
    main_path = private_args.path.main_path
    eval_path = "post_eval"
    before_folder = training_args.path_args.pretraining_folder #'before_training'
    after_folder = training_args.path_args.after_training_folder #'after_training'
    results_folder = training_args.path_args.results_path
    # Load pickle as a dataframe
    pickle_name = "predictions_before_training_2022-03-12-17-41-00.pkl"
    # if pickel name has 'before' in it, load before pickle
    if 'before' in pickle_name:
        folder = before_folder
    else:
        folder = after_folder
    pickles_folder = os.path.join(private_args.path.main_path, results_folder, folder)

    pickle_path = os.path.join(pickles_folder, pickle_name)

    # combine before and after pickles
    pickle_before = r'/home/tok/TRBLLmaker/results/pretraining/predictions_before_training_2022-03-12-17-41-00.pkl'
    pickle_after =r'/home/tok/TRBLLmaker/results/after_training/predictions_after_training_2022-03-14-11-17-57.pkl'
    combine_before_and_after(pickle_before, pickle_after)

    # pickl_name = "analysis_cos_pred_label_['decode_method', 'model', 'prompt_type']_2022-03-12 19:37:39.523430.pkl"#"full_analysis_120322_1939.pkl" # full_analysis_120322_1937.pkl
    pickls_path = os.path.join(main_path, eval_path)
    # iterate over all pickles
    for pickl_name in os.listdir(pickls_path):
        if pickl_name.endswith(".pkl") and pickl_name.startswith("analysis"):
            # load pickle
            pickl_path = os.path.join(pickls_path, pickl_name)
            df = pd.read_pickle(pickl_path)
            # generate graphs
            generate_graphs(df)

    # compare between models
    # bar plot for all models and compare all score.
    # x-axis represents the prompts, y-axis represents the scores values