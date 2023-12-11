#Authored by Sentient Generals on 11 December 2023

import pandas as pd
import numpy as np
import cv2
import csv
from itertools import islice
import matplotlib.pyplot as plt
import os
import sys
np.set_printoptions(threshold=sys.maxsize)


def main():
    
    labeled_df,challenge_df = get_labels()
    #print(labeled_df)

    
    new_labeled_list = np.zeros((3000,6)).tolist()
    new_challenge_list =np.zeros((700,6)).tolist()

    make_and_save_histos(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False)
    
    make_and_save_pics(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False)

    df_headers = ['feature_vector_1_histogram','feature_vector_2_histogram','feature_vector_3_histogram','sequence_histogram','features_picture','sequence_picture']
    
    new_labeled_df = pd.DataFrame(new_labeled_list, columns=df_headers)
    new_labeled_df['class'] = labeled_df['class']
    new_labeled_df.to_csv('new_labeled.csv')

    new_challenge_df = pd.DataFrame(new_challenge_list, columns=df_headers)
    new_challenge_df.to_csv('new_challenge.csv')
    

#--------------------------------------------------------------------------------------------#
#-----------------------------------GET DATA-------------------------------------------------#
#--------------------------------------------------------------------------------------------#

def get_labels():
    labeled_df = pd.read_csv('labeled.csv', engine='python', error_bad_lines=False)
    labeled_df = turn_into_ints(labeled_df)
    challenge_df = pd.read_csv('challenge.csv')
    challenge_df = turn_into_ints(challenge_df)

    return labeled_df, challenge_df

def turn_into_ints(df):
    df['feature_vector_1'] = df['feature_vector_1'].map(lambda row: into_ints(row))
    df['feature_vector_2'] = df['feature_vector_2'].map(lambda row: into_ints(row))
    df['feature_vector_3'] = df['feature_vector_3'].map(lambda row: into_ints(row))
    df['sequence'] = df['sequence'].map(lambda row: seq_into_ints(row))

    return df



def into_ints(row):
    return [int(element) for element in row.replace(' ', '').replace('[', '').replace(']', '')[0:-1].split('.')]

def seq_into_ints(row):
    return [int(element) for element in row.replace('[', '').replace(']', '').split()]


#--------------------------------------------------------------------------------------------#
#-----------------------------------HISTOGRAMS-----------------------------------------------#
#--------------------------------------------------------------------------------------------#

def make_and_save_histos(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False,):
    if save == True:
        os.makedirs(os.path.join(os.getcwd(),'histograms'), exist_ok=True)
    #make and save for labeled
    make_and_save_h(labeled_df, 'labeled', new_labeled_list, save)
    #make and save for challenge
    make_and_save_h(challenge_df, 'challenge', new_challenge_list, save)

def make_and_save_h(df, file_prefix, new_list, save):
    for idx, row in df.iterrows():
        #make and save histos for every vector
        vectors = ['feature_vector_1','feature_vector_2','feature_vector_3','sequence']
        for i, vector in enumerate(vectors):
            
            #find counts and add to new list
            unique,counts = np.unique(row[vector], return_counts=True)
            histo_as_dict = dict(zip(unique,counts))
            histogram = []
            max = 20
            if vector == 'sequence':
                max = 372
            for element in range(0,max):
                if histo_as_dict.get(element) is not None:
                    histogram.append(histo_as_dict.get(element))
                else:
                    histogram.append(0)
            new_list[idx][i] = np.array(histogram)

            #save a histogram picture if save is true
            if save == True:
                if(vector == 'sequence'):
                    plt.hist(row[vector], bins=np.arange(0, 372))
                    plt.xticks(np.arange(0, 372+1, 50))
                else:
                    plt.hist(row[vector], bins=np.arange(0, 20))
                    plt.xticks(np.arange(0, 20+1, 1.0))

                #axes false
                plt.title(f'samplte {idx}: {vector}')
                plt.ylabel('count')
                output_name = f'{file_prefix}_sample_{idx}_{vector}'
                save_path = os.path.join(os.getcwd(),f'histograms/{output_name}')
                plt.savefig(save_path)
                plt.clf()


#--------------------------------------------------------------------------------------------#
#-----------------------------------PICTURES-------------------------------------------------#
#--------------------------------------------------------------------------------------------#
def make_and_save_pics(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False):
    if save == True:
        os.makedirs(os.path.join(os.getcwd(),'pictures'), exist_ok=True)
    make_and_save_p(labeled_df, 'labeled',  new_labeled_list, save)
    make_and_save_p(challenge_df, 'challenge', new_challenge_list, save)

def make_and_save_p(df, file_prefix, new_list, save):
    for idx, row in df.iterrows():
        #R plane
        imgR = into_plane_feature(row, 'feature_vector_1')
        #G plane
        imgG = into_plane_feature(row, 'feature_vector_2')
        #B plane
        imgB = into_plane_feature(row, 'feature_vector_3')
        #RGB together
        imgRGB = np.array([imgR, imgG, imgB])

        new_list[idx][4] = imgRGB

        #black and white picture
        imgBW = np.array(row['sequence'])
        imgBW = into_pixels_sequence(imgBW).astype(int)
        imgBW.resize((32,32))

        new_list[idx][5] = imgBW

def into_plane_feature(row, feature):
    img = np.array(row[feature])
    img = into_pixels_feature(img).astype(int)
    img.resize((32,32))

    return img


def into_pixels_feature(num):
    num = 255*(num/19)
    return num

def into_pixels_sequence(num):
    num = 255*(num/371)
    return num

if __name__ == "__main__":
    main()