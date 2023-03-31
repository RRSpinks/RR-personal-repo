
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse
import os

# Function to interpret datetime format
def parse_datetime(date_string):
    try:
        # Try parsing the original ISO 8601 format
        date_string = date_string.replace("Z", "+00:00")
        dt = datetime.fromisoformat(date_string)
    except ValueError:
        # If the original format is not valid, try the custom format
        dt = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S.%f %z')
    weekday = dt.strftime('%A')
    date = dt.date()
    return date, weekday

# Data analysis functions
def calc_mean_job_count_df(df):
    # Calculate mean and SD of daily job count
    df[['date', 'weekday']] = df['created_at'].apply(parse_datetime).apply(pd.Series)
    daily_job_df= df.groupby(['date', 'template_id', 'name']).size().reset_index(name='count')
    mean_jobs_df = daily_job_df.groupby(['template_id', 'name']).agg({'count': ['mean', 'std', 'size']}).reset_index()
    mean_jobs_df.columns = ['template_id', 'name', 'mean', 'std', 'count']
    return df, mean_jobs_df
def calc_mean_attempts_df(df):
    # Calculate mean and SD of attempts count
    mean_attempts_df = df.groupby(['template_id', 'name']).agg({'attempts': ['mean', 'std', 'size']}).reset_index()
    mean_attempts_df.columns = ['template_id', 'name', 'mean', 'std', 'count']
    return mean_attempts_df
def calc_mean_weekdays_df(df):
    # Calculate mean, std and count for each weekday
    df['weekday_count'] = df.groupby(['weekday', 'date'])['name'].transform('count')
    stats_weekday = df.groupby('weekday').agg({'weekday_count': ['mean', 'std', 'size']}).reset_index()
    stats_weekday.columns = ['weekday', 'mean', 'std', 'count']
    weekday_mapping = {'Monday':0, 'Tuesday':1, 'Wednesday':2,'Thursday':3, 'Friday':4, 'Saturday':5,'Sunday':6}
    stats_weekday['weekday_no'] = stats_weekday['weekday'].replace(weekday_mapping, regex=True)
    stats_weekday = stats_weekday.sort_values('weekday_no')
    return stats_weekday

# Function to create table of feature IDs and corresponding names
def plot_id_table(df, output_dir):
    # Plot table with IDs and corresponding names
    id_name_table = df[['template_id', 'name']].values.tolist()
    fig, ax = plt.subplots(figsize=(6, len(id_name_table) * 0.3))  # Adjust the height of the figure based on the number of rows in the table
    ax.axis('off')  # Remove the axis
    table = plt.table(cellText=id_name_table, colLabels=['ID', 'Name'], loc='upper center', cellLoc='left', colWidths=[0.15, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    plt.tight_layout()
    # Save and show figure
    fig.savefig(f"{output_dir}/Feature_ID_table.png")    
    #plt.show()
    
# Function to plot daily job count vs feature IDs
def plot_daily_job_count(df, output_path):
    # Create figure
    fig, ax = plt.subplots()
    ax.bar(df['template_id'].astype(str), df['mean'], yerr=df['std'])
    plt.xticks(rotation=0, ha='center')
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.set_xticklabels(df['template_id'].astype(str))
    ax.set_ylabel('Jobs  /  Day')
    ax.set_title('Daily Count for Feature ID (mean +/- SD)')
    plt.tight_layout()
    fig.savefig(f"{output_path}/Feature_ID_daily_count.png")    
    #plt.show()

# Function to plot daily job count vs weekday    
def plot_weekday_job_count(df, output_path):
    # Create figure
    fig, ax = plt.subplots()
    ax.bar(df['weekday'], df['mean'], yerr=df['std'])
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Jobs  /  Day')
    ax.set_title('Daily Count for Weekdays (mean +/- SD)')
    plt.tight_layout()
    # Save and show figure
    fig.savefig(f"{output_path}/Weekday_daily_count.png")
    #plt.show()

# Function to plot attempt job count vs feature IDs
def plot_job_attempts_count(df, output_dir):
    # Create figure
    fig, ax = plt.subplots()
    ax.bar(df['template_id'].astype(str), df['mean'], yerr=df['std'])
    plt.axhline(y=8, color='r', linestyle='--')
    plt.text(-1,8.1,'Failed')
    plt.xticks(rotation=0, ha='center')
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.set_xticklabels(df['template_id'].astype(str))
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Attempts  /  Job')
    ax.set_title('Attempt Count for Feature ID (mean +/- SD)')
    plt.tight_layout()
    # Save and show figure
    fig.savefig(f"{output_dir}/Feature_ID_attempt_count.png")    
    #plt.show()

# Define paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
input_dir = os.path.join(parent_dir, "Data", "Feature_01", "Input")
output_dir = os.path.join(parent_dir, "Data", "Feature_01", "Output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Import data
job_df = pd.read_csv(os.path.join(input_dir, "job_202303291424.csv"))
features_df = pd.read_csv(os.path.join(input_dir, "_template__202303291642.csv"))    
job_merged_df = job_df.merge(features_df[['id', 'name']], left_on='template_id', right_on='id', how='left')

# Execute functions
job_dt_df, mean_jobs_df = calc_mean_job_count_df(job_merged_df)
mean_attempts_df = calc_mean_attempts_df(job_merged_df)
mean_weekdays_df = calc_mean_weekdays_df(job_dt_df)
plot_id_table(mean_jobs_df,output_dir)       
plot_daily_job_count(mean_jobs_df, output_dir)
plot_weekday_job_count(mean_weekdays_df, output_dir)
plot_job_attempts_count(mean_attempts_df, output_dir)