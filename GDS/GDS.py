import GEOparse # opens GDS data
import pandas as pd # data wrangling
from bioinfokit.visuz import gene_exp



import matplotlib as plt
#filepath of GDS data
filepath = input()
gds = GEOparse.GEOparse.parse_GDS(filepath,{})
GDS_data = gds.table
Metadata= gds.metadata
Column_info=gds.columns



def clean_data():
    '''removes unnecessary columns and duplicates, makes genes the index'''
    Table = GDS_data
    Table = Table.dropna().drop(columns = "ID_REF")
    Table = Table.groupby('IDENTIFIER').mean().reset_index()
    Table = Table.set_index('IDENTIFIER')
    return Table


def heatmap_100():
    '''returns heatmap of 100 genes with highest standard deviation'''
    table=clean_data()
    gene_sd=(table.std(axis=1)).to_frame()
    gene_sd=gene_sd.sort_values(by=[0],ascending=False)
    top_100=gene_sd[:100]
    top_data= table.loc[table.index.isin(list(top_100.index))]
    gene_exp.hmap(df=top_data, cmap='RdYlGn', dim=(4, 8), tickfont=(6, 4),r=500)


#open TF matrix with TFs as columns and targets as index, made from trrust raw data
TF_matrix= pd.read_table('tf_adjacency.csv', index_col=0)

Targets = list(TF_matrix.index) # list of all targets in TF matrix

#In order for pls to work GDS data and TF matrix must have the
#same number of genes and in same order

def TF_targets_in_data() :
    '''subsets GDS table with list of targets'''
    data = clean_data()
    data = data.loc[data.index.isin(Targets)]

    return data

def Matrix_of_data():
    '''subsets TF matrix with list of targets'''
    table=TF_targets_in_data()
    Matrix=TF_matrix.loc[TF_matrix.index.isin(list(table.index))]
    return Matrix



heatmap_100()