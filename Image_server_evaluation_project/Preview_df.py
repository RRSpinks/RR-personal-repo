import pandas as pd

def print_df_preview(df):
    print("Data Frame preview:")
    # Print DataFrame name
    for name, obj in globals().items():
        if isinstance(obj, pd.DataFrame) and obj is df:
            print(f"Name: {name}")
            break

    # Create a new DataFrame for the unique values preview
    preview_data = {}
    for column in df.columns:
        unique_values = df[column].unique()
        if len(unique_values) <= 5:
            preview_data[column] = unique_values
        else:
            preview_data[column] = [*unique_values[:3], '...']
    preview_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in preview_data.items()]))
    # Print the unique values preview as a table
    print(preview_df.head(10))


#Test            
csv_file_path = "/home/richard/Richard/RS_git/Image_sever_evaluation_project/_template__202303291642.csv"
df = pd.read_csv(csv_file_path)
#print(df.head())
#print_df_preview(df)

