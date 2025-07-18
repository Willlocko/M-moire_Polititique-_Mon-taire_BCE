# Script Python Complet pour l'Analyse Économétrique de Panel
# Ce script est conçu pour la thèse de Master et compile toutes les étapes de l'analyse :
# - Chargement et préparation des données de panel.
# - Estimation des modèles de panel (Effets Fixes et Effets Aléatoires).
# - Tests statistiques, notamment le test de Hausman.
# - Visualisation des données et des résultats des modèles.
# - Simulation prospective de l'HICP.
# ==============================================================================

# --- 1. Installation et Importation des Bibliothèques Nécessaires ---
# Ces commandes vérifient et installent les bibliothèques Python requises.
# 'openpyxl' est né# ==============================================================================
cessaire pour lire les fichiers .xlsx.
print("Vérification et installation des bibliothèques nécessaires...")
# La commande suivante est commentée car elle est spécifique à un environnement interactif.
# Si vous exécutez ce script dans un environnement local, assurez-vous que ces bibliothèques sont installées
# via 'pip install linearmodels openpyxl pandas numpy statsmodels matplotlib seaborn'.
# !pip install linearmodels openpyxl

# Importation des modules Python essentiels pour l'analyse de données et l'économétrie.
import pandas as pd
import numpy as np
import statsmodels.api as sm # Pour l'ajout de la constante au modèle
from linearmodels.panel import PanelOLS, RandomEffects, HausmanTest # Pour les modèles de panel
import matplotlib.pyplot as plt # Pour la création de graphiques
import seaborn as sns # Pour des visualisations statistiques améliorées

print("Bibliothèques importées avec succès.")

# ==============================================================================
# --- 2. Chargement des Données ---
# Cette section charge votre fichier de données Excel (.xlsx).
# Assurez-vous que le fichier 'Base_Panel1.xlsx' est bien situé dans le
# même répertoire que ce script, ou fournissez le chemin complet du fichier.
# ==============================================================================
excel_file_name = 'Base_Panel1.xlsx'
sheet_name_data = 'Base de données' # Nom de la feuille contenant vos données

try:
    df_raw = pd.read_excel(excel_file_name, sheet_name=sheet_name_data)
    print(f"\nFichier '{excel_file_name}' - Feuille '{sheet_name_data}' chargé avec succès.")
    print("Aperçu des premières lignes du jeu de données brut :")
    print(df_raw.head())
    print(f"\nNombre d'observations brutes : {len(df_raw)}")
    print(f"Colonnes brutes : {df_raw.columns.tolist()}")

except FileNotFoundError:
    print(f"\nERREUR : Le fichier '{excel_file_name}' n'a pas été trouvé.")
    print("Veuillez vérifier que le fichier Excel est dans le bon répertoire ou que le nom est correct.")
    raise SystemExit("Arrêt de l'exécution : Fichier de données introuvable.")
except Exception as e:
    print(f"\nERREUR lors du chargement du fichier Excel : {e}")
    print(f"Vérifiez que le nom de la feuille '{sheet_name_data}' est correct ou que le fichier n'est pas corrompu.")
    raise SystemExit("Arrêt de l'exécution : Problème de lecture du fichier Excel.")

# --- Titre Suggéré pour votre mémoire : Tableau X.Y : Aperçu des Données Brutes Importées ---
# --- Légende Suggérée : Ce tableau présente les cinq premières observations du jeu de données brut après importation, illustrant les colonnes originales et leur format.


# ==============================================================================
# --- 3. Préparation des Données de Panel ---
# Cette section nettoie et restructure les données pour qu'elles soient
# conformes au format requis pour l'analyse de panel (MultiIndex avec Entité et Temps).
# ==============================================================================

# Renommer les colonnes pour une meilleure lisibilité et pour correspondre aux variables de l'analyse.
# Assurez-vous que ces noms correspondent exactement aux en-têtes de votre fichier Excel.
df = df_raw.rename(columns={
    'Pays': 'Country',
    'Trimestre': 'Quarter', # Cette colonne est utilisée pour la conversion en format date
    'HICP (%)': 'HICP', # Variable dépendante
    'PIB (%)': 'GDP',
    'Crédit (%)': 'Credit',
    'FBCF (%)': 'FBCF',
    'Taux BCE (%)': 'ECB_Rate',
    'PEPP': 'PEPP'
})

print("\nColonnes du DataFrame après renommage :")
print(df.columns.tolist())

# Convertir la colonne 'Quarter' (ex: '2005Q1') en un index temporel DatetimeIndex.
# C'est une étape cruciale pour l'analyse de séries temporelles et de panel
# car linearmodels nécessite un index temporel de type date.
try:
    df['Quarter_Datetime'] = pd.PeriodIndex(df['Quarter'], freq='Q').to_timestamp()
    print("\nColonne 'Quarter' convertie en format DatetimeIndex ('Quarter_Datetime').")
except Exception as e:
    print(f"\nERREUR lors de la conversion de la colonne 'Quarter' : {e}")
    print("Vérifiez le format des données dans la colonne 'Trimestre' de votre fichier (format attendu : 'YYYYQn').")
    raise SystemExit("Arrêt : Erreur de format de trimestre.")

# Trier les données par pays puis par trimestre pour une structure de panel cohérente.
df = df.sort_values(by=['Country', 'Quarter_Datetime'])

# Définir l'index de panel (MultiIndex) avec 'Country' comme identifiant de l'entité
# et 'Quarter_Datetime' comme dimension temporelle.
df_panel = df.set_index(['Country', 'Quarter_Datetime'])

print("\nStructure du DataFrame de panel (premières lignes après indexation) :")
print(df_panel.head())
# Afficher le type de l'index temporel pour confirmation
print(f"Type de l'index temporel : {df_panel.index.get_level_values(1).dtype}")

# --- Titre Suggéré pour votre mémoire : Tableau X.Y : Structure du Jeu de Données de Panel après Transformation et Indexation ---
# --- Légende Suggérée : Ce tableau montre les cinq premières observations du DataFrame de panel (`df_panel`), avec les indices 'Country' et 'Quarter_Datetime' définis, et les colonnes renommées.

# Définition des variables dépendante (à expliquer) et indépendantes (explicatives).
dependent_var = df_panel['HICP']
exog_vars = df_panel[['GDP', 'Credit', 'FBCF', 'ECB_Rate', 'PEPP']]

# Ajouter une constante (intercept) au modèle de régression, nécessaire pour l'estimation.
exog_vars = sm.add_constant(exog_vars)

# Gérer les valeurs manquantes (NaN) : suppression des lignes complètes contenant des NaN.
# Ceci assure que le modèle ne rencontre pas de valeurs manquantes lors de l'estimation.
df_model_clean = pd.concat([dependent_var, exog_vars], axis=1).dropna()

# Extraire les variables nettoyées pour l'estimation des modèles.
dependent_var_clean = df_model_clean['HICP']
exog_vars_clean = df_model_clean.drop('HICP', axis=1)

print(f"\nNombre total d'observations dans le fichier d'origine : {len(df_panel)}")
print(f"Nombre d'observations utilisées après suppression des NaN : {len(dependent_var_clean)}")

if len(dependent_var_clean) == 0:
    print("ERREUR : Aucune observation valide après suppression des NaN. Vérifiez vos données manquantes.")
    raise SystemExit("Arrêt : Pas de données valides pour le modèle.")
elif len(dependent_var_clean) < 100:
    print("Avertissement : Le nombre d'observations utilisées est relativement faible (moins de 100). Cela peut affecter la robustesse des résultats.")


# ==============================================================================
# --- 4. Estimation du Modèle à Effets Fixes (FE) ---
# Ce modèle est utilisé lorsque les effets spécifiques aux entités (pays)
# sont corrélés avec les variables explicatives et sont constants dans le temps.
# ==============================================================================
print("\n" + "="*70)
print("--- ESTIMATION DU MODÈLE À EFFETS FIXES (FE) ---")
print("Ce modèle contrôle les caractéristiques inobservables et invariantes dans le temps spécifiques à chaque pays.")
print("="*70)

# Estimation du modèle FE avec des erreurs standard robustes aux clusters par entité.
# L'option 'entity_effects=True' ajoute une constante différente pour chaque pays.
# 'cov_type='clustered'' avec 'cluster_entity=True' corrige les erreurs standard pour la corrélation
# des observations au sein d'un même pays au fil du temps, ce qui est crucial pour les panels.
model_fe = PanelOLS(dependent_var_clean, exog_vars_clean, entity_effects=True)
results_fe = model_fe.fit(cov_type='clustered', cluster_entity=True)
print(results_fe)

# --- Titre Suggéré pour votre mémoire : Tableau X.Y : Résultats de l'Estimation du Modèle à Effets Fixes (FE) pour l'HICP ---
# --- Légende Suggérée : Ce tableau présente les coefficients estimés, les erreurs standard, les statistiques T, les p-values et les intervalles de confiance à 95% pour le modèle à effets fixes, incluant les effets inobservables spécifiques aux pays.


# ==============================================================================
# --- 5. Estimation du Modèle à Effets Aléatoires (RE) et Test de Hausman ---
# Le modèle RE est plus efficace si les effets spécifiques aux entités sont
# aléatoires et non corrélés avec les variables explicatives.
# Le Test de Hausman aide à choisir entre le modèle FE et RE.
# ==============================================================================
print("\n" + "="*70)
print("--- ESTIMATION DU MODÈLE À EFFETS ALÉATOIRES (RE) ---")
print("Ce modèle suppose que les effets spécifiques aux pays sont aléatoires et non corrélés avec les régresseurs.")
print("="*70)

# Estimation du modèle RE avec des erreurs standard robustes aux clusters par entité.
model_re = RandomEffects(dependent_var_clean, exog_vars_clean)
results_re = model_re.fit(cov_type='clustered', cluster_entity=True)
print(results_re)

# --- Titre Suggéré pour votre mémoire : Tableau X.Y : Résultats de l'Estimation du Modèle à Effets Aléatoires (RE) pour l'HICP ---
# --- Légende Suggérée : Ce tableau récapitule les coefficients estimés, les erreurs standard robustes, les statistiques T, les p-values et les intervalles de confiance à 95% pour le modèle à effets aléatoires.

print("\n" + "="*70)
print("--- TEST DE HAUSMAN (Pour le Choix entre Modèle FE et RE) ---")
print("H0 : La différence entre les estimateurs FE et RE n'est pas systématique (RE est consistent et efficace).")
print("H1 : La différence est systématique (FE est consistent, RE est inconsistent).")
print("="*70)

# Exécution du test de Hausman pour comparer les deux modèles.
# Le test de Hausman aide à déterminer si les effets spécifiques aux entités sont corrélés avec les régresseurs.
hausman_test_results = HausmanTest(results_fe, results_re)
print(hausman_test_results)

# --- Titre Suggéré pour votre mémoire : Tableau X.Y : Résultats du Test de Hausman pour le Choix entre Effets Fixes et Effets Aléatoires ---
# --- Légende Suggérée : Ce tableau présente la statistique de Hausman, ses degrés de liberté et sa p-value, permettant de déterminer le modèle (FE ou RE) le plus approprié pour l'analyse des données de panel.

print("\n**Conclusion du Test de Hausman :**")
if hausman_test_results.pvalue >= 0.05:
    print("La P-value est supérieure ou égale à 0.05. Nous ne rejetons pas l'hypothèse nulle.")
    print("Cela signifie que les effets aléatoires ne sont probablement pas corrélés avec les régresseurs.")
    print("Le modèle à **Effets Aléatoires (RE)** est préféré car il est plus efficace.")
    preferred_model_results = results_re # Définit le modèle préféré pour les analyses ultérieures
else:
    print("La P-value est inférieure à 0.05. Nous rejetons l'hypothèse nulle.")
    print("Cela signifie que la différence entre les estimateurs est significative, suggérant une corrélation.")
    print("Le modèle à **Effets Fixes (FE)** est préférable car il est consistent (non biaisé).")
    preferred_model_results = results_fe # Définit le modèle préféré pour les analyses ultérieures

print("\n" + "="*70)
print("--- Fin de l'Analyse Économétrique ---")
print("="*70)


# ==============================================================================
# --- 6. Visualisation des Résultats et des Données ---
# Cette section génère divers graphiques pour illustrer les données et les
# résultats des modèles, utiles pour votre rapport.
# ==============================================================================
sns.set_style("whitegrid") # Définit le style des graphiques pour une meilleure esthétique
plt.rcParams['figure.figsize'] = (12, 7) # Taille par défaut des figures pour une meilleure lisibilité

# ------------------------------------------------------------------------------
# 6.1 Graphique des Tendances de l'HICP pour quelques pays
# Observer l'évolution de l'inflation au fil du temps pour des pays sélectionnés.
# ------------------------------------------------------------------------------
print("\n[Figure X.Y] : Évolution Trimestrielle de l'HICP pour une Sélection de Pays de la Zone Euro (2005Q1-2023Q4)")
plt.figure()
ax = plt.gca() # Obtient les axes du graphique actuel

# Liste des pays à inclure dans le graphique. Adaptez cette liste à vos besoins.
countries_to_plot = ['Allemagne', 'France', 'Italie', 'Espagne', 'Belgique']
available_countries = df_panel.index.get_level_values('Country').unique().tolist()

for country in countries_to_plot:
    if country in available_countries:
        country_hicp_data = df_panel.loc[country, 'HICP']
        if not country_hicp_data.empty:
            country_hicp_data.plot(ax=ax, label=country)
        else:
            print(f"Avertissement : Aucune donnée HICP valide trouvée pour le pays '{country}'.")
    else:
        print(f"Avertissement : Le pays '{country}' n'existe pas dans les données de panel. Vérifiez l'orthographe.")

plt.title('HICP pour une Sélection de Pays de la Zone Euro')
plt.xlabel('Trimestre')
plt.ylabel('HICP (%)')
plt.legend(title='Pays', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Légende Suggérée : Ce graphique en série temporelle présente les tendances de l'indice des prix à la consommation harmonisé (HICP) pour les pays clés de l'échantillon, permettant d'observer les dynamiques d'inflation sur la période étudiée.


# ------------------------------------------------------------------------------
# 6.2 Nuage de points HICP vs. PIB
# Visualise la relation brute entre l'inflation et la croissance économique.
# ------------------------------------------------------------------------------
print("\n[Figure X.Y] : Relation entre l'HICP et la Croissance du PIB dans la Zone Euro (2005Q1-2023Q4)")
plt.figure()

# Réinitialiser l'index pour que 'Country' et 'Quarter_Datetime' deviennent des colonnes.
df_for_plot_scatter = df_panel.reset_index()

# Convertir les colonnes en numérique et gérer les NaN spécifiquement pour ce graphique.
df_for_plot_scatter['GDP'] = pd.to_numeric(df_for_plot_scatter['GDP'], errors='coerce')
df_for_plot_scatter['HICP'] = pd.to_numeric(df_for_plot_scatter['HICP'], errors='coerce')
df_for_plot_scatter.dropna(subset=['GDP', 'HICP'], inplace=True)

sns.scatterplot(data=df_for_plot_scatter, x='GDP', y='HICP', hue='Country', alpha=0.6, s=50)
plt.title('HICP en fonction de la Croissance du PIB par Pays')
plt.xlabel('Croissance du PIB (%)')
plt.ylabel('HICP (%)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title='Pays')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Légende Suggérée : Ce nuage de points visualise la relation observée entre le taux d'inflation (HICP) et la croissance du PIB pour l'ensemble des pays et trimestres de l'échantillon.


# ------------------------------------------------------------------------------
# 6.3 Évolution de l'HICP moyenne et période du PEPP
# Montre la tendance agrégée de l'inflation et la période d'une politique clé.
# ------------------------------------------------------------------------------
print("\n[Figure X.Y] : Évolution de l'HICP Moyenne de la Zone Euro et Période du Programme d'Achats d'Urgence Pandémique (PEPP)")
plt.figure()

# Calculer la moyenne de l'HICP sur tous les pays pour chaque trimestre.
hicp_mean = df_panel.groupby(level='Quarter_Datetime')['HICP'].mean()
hicp_mean.plot(label='HICP Moyenne Zone Euro', color='blue', linewidth=2)

# Définir la période du PEPP. Ajustez ces dates si elles sont différentes pour votre analyse.
pepp_start = pd.to_datetime('2020-03-01') # Date de début du PEPP (Mars 2020)
pepp_end = pd.to_datetime('2022-12-31') # Date de fin des achats nets du PEPP (Décembre 2022)

# Ajoute une zone ombrée pour visualiser la période du PEPP.
plt.axvspan(pepp_start, pepp_end, color='gray', alpha=0.15, label='Période PEPP (Mars 2020 - Décembre 2022)')

# Ajoute du texte pour marquer le début et la fin de la période du PEPP.
y_text_pos = plt.ylim()[1] * 0.95 # Positionne le texte en haut du graphique
plt.text(pepp_start, y_text_pos, 'Début PEPP', rotation=90, va='top', ha='right', color='darkgray', fontsize=9)
plt.text(pepp_end, y_text_pos, 'Fin PEPP', rotation=90, va='top', ha='left', color='darkgray', fontsize=9)

plt.title('HICP Moyenne de la Zone Euro et Période du PEPP')
plt.xlabel('Date du Trimestre')
plt.ylabel('HICP (%)')
plt.legend(loc='upper left', title='Légende')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Légende Suggérée : Ce graphique illustre la tendance de l'HICP moyenne à travers les pays de la zone euro et met en évidence la période d'activation du Programme d'Achats d'Urgence Pandémique (PEPP) de la BCE.


# ------------------------------------------------------------------------------
# 6.4 Graphique des Coefficients du Modèle RE avec Intervalles de Confiance
# Permet de visualiser la magnitude et la significativité statistique des effets.
# ------------------------------------------------------------------------------
print("\n[Figure X.Y] : Coefficients Estimés du Modèle à Effets Aléatoires (RE) pour l'HICP avec Intervalles de Confiance à 95%")
plt.figure()

params_re = results_re.params
conf_int_re = results_re.conf_int()

# Création d'un DataFrame pour faciliter le tracé des coefficients et de leurs intervalles.
df_coef_re = pd.DataFrame({
    'Coefficient': params_re,
    'Lower CI': conf_int_re['lower'], # Accès par nom de colonne
    'Upper CI': conf_int_re['upper']  # Accès par nom de colonne
})
df_coef_re = df_coef_re.drop('const', errors='ignore') # Exclut la constante du graphique pour une meilleure lisibilité.

ax_coef = df_coef_re[['Coefficient']].plot(kind='barh', figsize=(10, 6), legend=False)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8) # Ligne de référence à zéro.

# Ajout des barres d'erreur représentant les intervalles de confiance.
for i, (index, row) in enumerate(df_coef_re.iterrows()):
    ax_coef.errorbar(x=row['Coefficient'], y=i,
                xerr=[[row['Coefficient'] - row['Lower CI']], [row['Upper CI'] - row['Coefficient']]],
                fmt='o', color='black', capsize=5) # 'fmt='o'' pour un marqueur au centre

plt.title('Coefficients du Modèle à Effets Aléatoires (RE) avec Intervalles de Confiance')
plt.xlabel('Valeur du Coefficient')
plt.ylabel('Variables Explicatives')
plt.grid(axis='x', linestyle='--', alpha=0.7) # Grille sur l'axe des X pour la lecture
plt.tight_layout() # Ajuste la mise en page pour éviter les chevauchements
plt.show()

# --- Légende Suggérée : Ce graphique visualise l'ampleur et la significativité statistique des coefficients de régression pour le modèle RE, où les barres d'erreur représentent les intervalles de confiance à 95%. Les coefficients dont l'intervalle traverse zéro ne sont pas statistiquement significatifs.


# ------------------------------------------------------------------------------
# 6.5 Graphique des Effets Spécifiques aux Pays (Modèle à Effets Fixes)
# Montre l'hétérogénéité inobservée entre les pays, telle qu'estimée par le modèle FE.
# ------------------------------------------------------------------------------
if hasattr(results_fe, 'entity_effects'): # Vérifie si les effets fixes ont été estimés
    print("\n[Figure X.Y] : Estimation des Effets Spécifiques aux Pays (Modèle à Effets Fixes)")
    plt.figure(figsize=(10, 6))
    entity_effects = results_fe.entity_effects.sort_values(ascending=False) # Trie les effets pour une meilleure lecture
    sns.barplot(x=entity_effects.values, y=entity_effects.index, palette='viridis')
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Effets Spécifiques aux Pays (Modèle à Effets Fixes)')
    plt.xlabel('Effet Fixe (contribution à l\'HICP)')
    plt.ylabel('Pays')
    plt.grid(axis='x', linestyle='--', alpha=0.7) # Grid on X-axis for readability
    plt.tight_layout() # Adjust layout to prevent overlaps
    plt.show()
    print("\n--- Graphique des Effets Spécifiques aux Pays (Modèle FE) généré avec succès ---")
else:
    print("\nAttention : Les effets spécifiques aux entités du modèle FE n'ont pas pu être extraits.")
    print("Veuillez vous assurer que le modèle FE a été exécuté avec 'entity_effects=True' dans la section 4.")
    print("Si le modèle FE n'a pas été estimé ou si 'results_fe' n'est pas défini, cette erreur apparaîtra.")


# ------------------------------------------------------------------------------
# 6.6 Graphiques des Résidus du Modèle RE (Nuage de points et Histogramme)
# Ces graphiques aident à diagnostiquer l'adéquation du modèle.
# ------------------------------------------------------------------------------
print("\n[Figure X.Y] : Nuage de Points des Résidus du Modèle RE vs. Valeurs Prédites de l'HICP")
plt.figure()
residuals_re = results_re.resids.squeeze() # Get and ensure 1D residuals
fitted_values_re = results_re.fitted_values.squeeze() # Get and ensure 1D fitted values

sns.scatterplot(x=fitted_values_re, y=residuals_re, alpha=0.6, s=50)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8) # Line at zero for residuals
plt.title('Résidus du Modèle à Effets Aléatoires vs. Valeurs Prédites')
plt.xlabel('Valeurs Prédites (HICP)')
plt.ylabel('Résidus')
plt.grid(True, linestyle='--', alpha=0.7) # Grid for readability
plt.tight_layout() # Adjust layout to prevent overlaps
plt.show()

# --- Légende Suggérée : Ce nuage de points des résidus par rapport aux valeurs prédites aide à vérifier l'absence de motifs, ce qui est indicatif d'une spécification de modèle adéquate.


print("\n[Figure X.Y] : Distribution des Résidus du Modèle à Effets Aléatoires")
plt.figure()
sns.histplot(residuals_re, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution des Résidus du Modèle à Effets Aléatoires')
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Légende Suggérée : Cet histogramme illustre la distribution des résidus du modèle RE, idéalement centrée autour de zéro et présentant une forme proche d'une distribution normale.


# ------------------------------------------------------------------------------
# 6.7 Box Plots des variables clés par pays
# Fournit un aperçu de la distribution et de la variabilité des variables clés
# pour chaque pays sur la période d'étude.
# ------------------------------------------------------------------------------
print("\n[Figures X.Y à X.Z] : Distributions des variables clés par pays (Box Plots)")
vars_to_plot_boxplot = ['HICP', 'GDP', 'Credit', 'ECB_Rate'] # Variables à visualiser

for var in vars_to_plot_boxplot:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_panel.reset_index(), x='Country', y=var, palette='pastel')
    plt.title(f'Distribution de {var} par Pays de la Zone Euro (2005Q1-2023Q4)')
    plt.xlabel('Pays')
    plt.ylabel(f'{var} (%)')
    plt.xticks(rotation=45, ha='right') # Rotation des étiquettes des pays pour la lisibilité
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Légende Suggérée (pour chaque graphique de box plot) : Ce graphique en boîtes illustre la distribution trimestrielle de la variable [Nom de la Variable] pour chaque pays du panel, offrant un aperçu de la variabilité, des valeurs médianes et des valeurs extrêmes.


# ==============================================================================
# --- 7. Simulation de l'HICP jusqu'en 2030 ---
# Cette section projette l'HICP future basée sur le modèle préféré (RE).
# ==============================================================================
print("\n" + "="*70)
print("--- Simulation de l'HICP jusqu'en 2030 ---")
print("Cette simulation projette l'HICP future en utilisant les coefficients du modèle à effets aléatoires (RE).")
print("ATTENTION : Elle repose sur l'hypothèse simpliste que toutes les variables explicatives (GDP, Credit, FBCF, ECB_Rate, PEPP)")
print("restent constantes à leur dernière valeur observée pour chaque pays. Les résultats sont donc indicatifs.")
print("="*70)

last_observed_date = df_panel.index.get_level_values('Quarter_Datetime').max()
print(f"\nDernière période observée dans les données : {last_observed_date.strftime('%Y-%m-%d')}")

# Définit les trimestres futurs jusqu'à fin 2030.
future_dates = pd.date_range(start=last_observed_date + pd.DateOffset(months=3),
                             end='2030-12-31',
                             freq='QS-OCT') # 'QS-OCT' pour le 1er jour de Janvier, Avril, Juillet, Octobre

print(f"Période de projection : du {future_dates.min().strftime('%Y-%m-%d')} au {future_dates.max().strftime('%Y-%m-%d')}")

# Prend les dernières valeurs observées des variables explicatives pour chaque pays.
last_values_per_country = df_panel.groupby(level='Country').last()[['GDP', 'Credit', 'FBCF', 'ECB_Rate', 'PEPP']]

# Construit un DataFrame avec les valeurs des variables explicatives pour les périodes futures.
future_data_list = []
countries = df_panel.index.get_level_values('Country').unique()

for country in countries:
    last_vals = last_values_per_country.loc[country]
    df_country_future = pd.DataFrame(index=future_dates)
    for col in last_vals.index:
        df_country_future[col] = last_vals[col]

    df_country_future['Country'] = country
    df_country_future.set_index('Country', append=True, inplace=True)
    df_country_future = df_country_future.reorder_levels(['Country', df_country_future.index.names[0]])
    df_country_future.index.names = ['Country', 'Quarter_Datetime']
    future_data_list.append(df_country_future)

future_exog = pd.concat(future_data_list)
future_exog_with_const = sm.add_constant(future_exog) # Ajoute la constante pour la prédiction

# Effectue la prédiction de l'HICP pour les périodes futures en utilisant le modèle RE.
predicted_hicp_future = results_re.predict(exog=future_exog_with_const)
predicted_hicp_future = predicted_hicp_future.rename('Predicted_HICP')

# Prépare les données observées et prédites pour la visualisation.
observed_hicp = df_panel['HICP'].rename('HICP_Value')
observed_hicp_df = observed_hicp.to_frame()
observed_hicp_df['Type'] = 'Observé'

predicted_hicp_future_df = predicted_hicp_future.to_frame()
predicted_hicp_future_df['HICP_Value'] = predicted_hicp_future_df['Predicted_HICP']
predicted_hicp_future_df['Type'] = 'Prédit (hypothèse constante)'
predicted_hicp_future_df = predicted_hicp_future_df.drop(columns=['Predicted_HICP'])

# Concatène les données historiques et les prévisions.
combined_hicp_for_plot = pd.concat([observed_hicp_df, predicted_hicp_future_df])
combined_hicp_for_plot = combined_hicp_for_plot.sort_index()

# ------------------------------------------------------------------------------
# Visualisation des résultats de simulation (HICP observée vs prédite)
# ------------------------------------------------------------------------------
print("\n[Figure X.Y] : HICP Observée et Projections jusqu'en 2030 (Modèle RE)")
# Sélectionner quelques pays à visualiser dans la simulation.
countries_to_simulate_plot = ['Allemagne', 'France', 'Italie', 'Espagne']

for country in countries_to_simulate_plot:
    plt.figure(figsize=(15, 8))
    country_data_plot = combined_hicp_for_plot.loc[country].reset_index()

    sns.lineplot(data=country_data_plot, x='Quarter_Datetime', y='HICP_Value', hue='Type', style='Type', markers=False, linewidth=2)

    # Ajoute une ligne pour marquer le passage des données observées aux données prédites.
    plt.axvline(last_observed_date, color='red', linestyle='--', label='Début Prédiction', linewidth=1.5)

    plt.title(f'HICP Observée et Projections pour {country} (jusqu\'à 2030)')
    plt.xlabel('Date du Trimestre')
    plt.ylabel('HICP (%)')
    plt.legend(title='Type de Donnée')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Légende Suggérée : Ce graphique présente l'évolution historique de l'HICP et sa projection jusqu'en 2030, obtenue à partir du modèle à effets aléatoires. Note : Ces projections sont basées sur l'hypothèse simpliste que les variables explicatives restent constantes à leurs dernières valeurs observées.

print("\n" + "="*70)
print("--- Exécution du script d'analyse économétrique terminée ---")
print("Vous pouvez maintenant copier les sorties textuelles et enregistrer les graphiques pour votre mémoire.")
print("==============================================================================")
