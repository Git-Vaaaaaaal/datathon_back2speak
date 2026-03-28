import pandas as pd

# Tu fais ta correspondance toi-même
input = ""
output = ""
path_csv = "Donnees\ch\exercices_a_realiser.csv" #Chemin du csv

dict_syl = {
    "ISO01" : "ch",
    "SYL01" : "cha",
    "SYL02" : "chi",
    "SYL03" : "cho",
    "W_I01" : "chat",
    "W_L02" : "chien",
    "W_L03" : "chaise",
    "W_I04" : "chaussure",
    "W_I05" : "chapeau",
    "W_I06" : "cheval",
    "W_M01" : "machine",
    "W_M02" : "bouchon",
    "W_M03" : "échelle",
    "W_M04" : "t-shirt",
    "W_M05" : "fourchette",
    "W_M06" : "rocher",
    "W_F01" : "bouche",
    "W_F02" : "fleche",
    "W_F03" : "niche",
    "W_F04" : "manche",
    "W_F05" : "vache",
    "W_F06" : "poche",
    "P01" : "chouquette a la crème",
    "P02" : "chocolat au lait",
    "P03" : "Charriot de courses",
    "P04" : "La fourchette tombe",
    "P05" : "Le bouchon est bleu",
    "P06" : "Il lave le tee-shirt",
    "P07" : "Il dort dans la niche",
    "P08" : "Elle colorie une vache",
    "P09" : "Le garcon peche"
}

def str_value(name, dict_syl=dict_syl):
    for k, v in dict_syl.items():
        if k in name:
            return v
    return None

# Ajout traduction dans le csv
""" df = pd.read_csv(path_csv)
df['Traduction'] = df['audio_file'].apply(str_value)
df.to_csv("ton_fichier_avec_value.csv", index=False) """

def sort_byword(df, name=str, dict_syl=dict_syl):
    str_trad = dict_syl.values(name)
    df_sort = df[df["Traduction"] == str_trad]
    return df_sort

def sort_byage(df, age=int, dict_syl=dict_syl):
    df_sort = df[df["age (en annees)"] == age]
    return df_sort

def sort_bysex(df, sex=str, dict_syl=dict_syl):
    df_sort = df[df["sexe"] == sex]
    return df_sort
