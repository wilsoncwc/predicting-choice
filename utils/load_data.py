import pandas as pd

from utils.constants import dataset_root

def split_row(row):
    code1, code2 = row.iloc[0]['Code'].split('/')
    name1, name2 = row.iloc[0]['Name'].split('/')
    row1 = row.copy()
    row1['Code'] = code1
    row1['Name'] = name1
    row2 = row.copy()
    row2['Code'] = code2
    row2['Name'] = name2
    return pd.concat([row1, row2])
    
def load_city_cluster_df(numeric=True):
    df = pd.read_excel(f'{dataset_root}/clustermembershipv2.xls', sheet_name='Clusters by Local Authority', header=9)
    df = df[['Code', 'Name', 'Supergroup Name', 'Group Name', 'Subgroup Name']]
    
    # Remove last two lines (empty rows)
    df = df[:-2]
    
    # Unmerge London/Westminster and Cornwall/Isles of Scilly
    lon_wes_row = df.query('Name == "City of London/Westminster"')
    corn_isles_row = df.query('Name == "Cornwall/Isles of Scilly"')
    df = pd.concat([df, split_row(lon_wes_row), split_row(corn_isles_row)])
    
    df['Supergroup Name'] = df['Supergroup Name'].astype('category')
    df['Group Name'] = df['Group Name'].astype('category')
    df['Subgroup Name'] = df['Subgroup Name'].astype('category')
    
    if numeric:
        # Turn categorical data into numeric
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    # subtract one to make it 0 indexed
    
    return df

def load_gva_df(): 
    gva_df = pd.read_excel(f'{dataset_root}/regionalgvaibylainuk.xls', sheet_name='Total GVA', header=2)
    return gva_df

def load_deprivation_df():
    sheets_and_cols = {
        'IMD': 'IMD',
        'Income': 'Income',
        'Employment': 'Employment',
        'Education': 'Education, Skills and Training',
        'Health': 'Health Deprivation and Disability',
        'Crime': 'Crime',
        'Barriers': 'Barriers to Housing and Services',
        'Living': 'Living Environment',
        'IDACI': 'Income Deprivation Affecting Children Index (IDACI)',
        'IDAOPI': 'Income Deprivation Affecting Older People (IDAOPI)',
    }
    id_cols = ['Local Authority District code (2013)', 'Local Authority District name (2013)']
    dep_metrics = ['Average rank', 'Average score']
    
    out_df = None
    for sheet in sheets_and_cols:
        dep_df = pd.read_excel(f'{dataset_root}/File_10_ID2015_Local_Authority_District_Summaries.xlsx',
                               sheet_name=sheet, header=0)
        cols_to_read = id_cols + [f'{sheets_and_cols[sheet]} - {metric}' for metric in dep_metrics]  
        dep_df = dep_df[cols_to_read]
        if out_df is not None:
            out_df = out_df.merge(dep_df, on=id_cols)
        else:
            out_df = dep_df
    return out_df