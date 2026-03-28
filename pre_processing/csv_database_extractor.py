import pandas as pd

# Tu fais ta correspondance toi-même
""" input = ""
output = ""
path_csv = "" #Chemin du csv """

dict_syl = {
    "ISO01" : "ch",
    "SYL01" : "cha",
    "SYL02" : "chi",
    "SYL03" : "cho",
    "I01" : "chat",
    "L02" : "chien",
    "L03" : "chaise",
    "I04" : "chaussure",
    "I05" : "chapeau",
    "I06" : "cheval",
    "M01" : "machine",
    "M02" : "bouchon",
    "M03" : "échelle",
    "M04" : "t-shirt",
    "M05" : "fourchette",
    "M06" : "rocher",
    "F01" : "bouche",
    "F02" : "fleche",
    "F03" : "niche",
    "F04" : "manche",
    "F05" : "vache",
    "F06" : "poche",
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
path_csv = "audio_db.csv"
df = pd.read_csv(path_csv)
df['Traduction'] = df['audio_id'].apply(str_value)
df.to_csv("ton_fichier_avec_value.csv", index=False)

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