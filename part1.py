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
    
    #get the files as panda dataframes and integers
    labeled_df,challenge_df = get_labels()

    #create lists to hold results from the following functions that will later be turned into a DF
    new_labeled_list = np.zeros((3000,6)).tolist()
    new_challenge_list =np.zeros((700,6)).tolist()

    #turn into histograms and save into the lists and possibly save into actual png images if save = true
    make_and_save_histos(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False)
    #turn into images and save into the lists and possibly save into actual png images if save = true
    make_and_save_pics(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False)

    #the headers for our dataframes
    df_headers = ['feature_vector_1_histogram','feature_vector_2_histogram','feature_vector_3_histogram','sequence_histogram','features_picture','sequence_picture']
    
    #turn the labeled list into a dataframe, add the class column, then turn into csv
    new_labeled_df = pd.DataFrame(new_labeled_list, columns=df_headers)
    new_labeled_df['class'] = labeled_df['class']
    new_labeled_df.to_csv('new_labeled.csv')

    #turn the challenge list into a dataframe, then turn into csv
    new_challenge_df = pd.DataFrame(new_challenge_list, columns=df_headers)
    new_challenge_df.to_csv('new_challenge.csv')
    

#--------------------------------------------------------------------------------------------#
#-----------------------------------GET DATA-------------------------------------------------#
#--------------------------------------------------------------------------------------------#

#read in the csv files and turn them into dataframes with integers
def get_labels():
    labeled_df = pd.read_csv('labeled.csv', engine='python', error_bad_lines=False) #read it in
    labeled_df = turn_into_ints(labeled_df) #turn columns into ints
    challenge_df = pd.read_csv('challenge.csv') #read it in
    challenge_df = turn_into_ints(challenge_df) #turn columns into ints

    return labeled_df, challenge_df

#takes the columns that should be int arrays and turn them for strings to ints
def turn_into_ints(df):
    #the same thing happens for each one, so i will only explain once: take the entire column, run into_ints for every row in column
    df['feature_vector_1'] = df['feature_vector_1'].map(lambda row: into_ints(row))
    df['feature_vector_2'] = df['feature_vector_2'].map(lambda row: into_ints(row))
    df['feature_vector_3'] = df['feature_vector_3'].map(lambda row: into_ints(row))
    #take the entire column, run seq_into_ints for every row in column (seq because it is laid out differently with no periods in csv file)
    df['sequence'] = df['sequence'].map(lambda row: seq_into_ints(row))

    return df


#goes over the entire array and splits on period, turn the strings into ints, returns array of ints
def into_ints(row):
    #take everything out but periods, split on periods, turn into ints
    return [int(element) for element in row.replace(' ', '').replace('[', '').replace(']', '')[0:-1].split('.')]

#goes over the entire array and splits the string into ints, returns array of ints
def seq_into_ints(row):
    #take everything out except spaces and split on whitespace, turn the strings into ints, returns array of ints
    return [int(element) for element in row.replace('[', '').replace(']', '').split()]


#--------------------------------------------------------------------------------------------#
#-----------------------------------HISTOGRAMS-----------------------------------------------#
#--------------------------------------------------------------------------------------------#

#makes and saves histograms from both files
def make_and_save_histos(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False,):
    
    if save == True: #make the folder for the saved histograms in the current directory
        os.makedirs(os.path.join(os.getcwd(),'histograms'), exist_ok=True)
    #make and save for labeled
    make_and_save_h(labeled_df, 'labeled', new_labeled_list, save)
    #make and save for challenge
    make_and_save_h(challenge_df, 'challenge', new_challenge_list, save)

#creates histograms for a single file and saves them in the new_list. alsol saves histograms as pictures if save = True
def make_and_save_h(df, file_prefix, new_list, save):
    for idx, row in df.iterrows(): #for every sample
        
        vectors = ['feature_vector_1','feature_vector_2','feature_vector_3','sequence']
        for i, vector in enumerate(vectors): #for every vector of interest
            
            #find counts and add to new list
            unique,counts = np.unique(row[vector], return_counts=True)
            histo_as_dict = dict(zip(unique,counts)) #dictionary of counts, does not include all 20 numbers if they were not present in line
            histogram = []
            max = 20
            if vector == 'sequence':
                max = 372
            for element in range(0,max): #we need to create counts for the numbers not present in the line
                if histo_as_dict.get(element) is not None:
                    histogram.append(histo_as_dict.get(element))
                else:
                    histogram.append(0)
            new_list[idx][i] = np.array(histogram) #add the histogram to our list at the current sample and index

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

#makes and saves pictures from both files
def make_and_save_pics(labeled_df, challenge_df, new_labeled_list, new_challenge_list, save = False):
    if save == True: #make the folder for the saved pictures in the current directory
        os.makedirs(os.path.join(os.getcwd(),'pictures'), exist_ok=True)
    #make and save for labeled
    make_and_save_p(labeled_df, 'labeled',  new_labeled_list, save)
    #make and save for challenge
    make_and_save_p(challenge_df, 'challenge', new_challenge_list, save)

#makes and saves pictures from a single file and saves it into new_list and save picture to computer if save = true
def make_and_save_p(df, file_prefix, new_list, save):
    for idx, row in df.iterrows(): #for every sample
        #R plane
        imgR = into_plane_feature(row, 'feature_vector_1')
        #G plane
        imgG = into_plane_feature(row, 'feature_vector_2')
        #B plane
        imgB = into_plane_feature(row, 'feature_vector_3')
        #RGB together
        imgRGB = np.array([imgR, imgG, imgB]) #just put them together to get the full RGB picture

        new_list[idx][4] = imgRGB


        #black and white picture
        imgBW = np.array(row['sequence'])
        imgBW = into_pixels_sequence(imgBW).astype(int)
        imgBW.resize((32,32)) #resize to be the 32x32 picture

        new_list[idx][5] = imgBW

        if save == True: #save the pictures as an actual jpg file if save = True
            imgRGB = np.rollaxis(imgRGB,0,3) #change from 3X32x32 into 32x32x3 so that it in cv2 RGB format
            output_name = f'{file_prefix}_sample_{idx}_imgRGB.jpg'
            save_path = os.path.join(os.getcwd(),f'pictures/{output_name}')
            cv2.imwrite(save_path, imgRGB) #save the image at the given path

            output_name = f'{file_prefix}_sample_{idx}_imgBW.jpg'
            save_path = os.path.join(os.getcwd(),f'pictures/{output_name}')
            cv2.imwrite(save_path, imgBW) #save the image at the given path

#turns a feature vector into a 32x32 image plane
def into_plane_feature(row, feature):
    img = np.array(row[feature]) #get the specified feature vector
    img = into_pixels_feature(img).astype(int) #turn into pixel values
    img.resize((32,32)) #resize to be the 32x32 picture plane

    return img

#turns a given feature vector int into its corresponding pixel value
def into_pixels_feature(num):
    num = 255*(num/19)
    return num

#turns a given sequence vector int into its corresponding pixel value
def into_pixels_sequence(num):
    num = 255*(num/371)
    return num

if __name__ == "__main__":
    main()