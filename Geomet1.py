import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import shap
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Récupération Métallurgique",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .author {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.markdown("<h1 class='main-header'>Prédiction de Récupération Métallurgique</h1>", unsafe_allow_html=True)
st.markdown("<p class='author'>Auteur: Didier Ouedraogo, P.Geo</p>", unsafe_allow_html=True)

# Fonction pour télécharger le DataFrame
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Sidebar pour la navigation
st.sidebar.title("Navigation")
pages = ["Accueil", "Importation des données", "Exploration des données", "Modélisation", "Prédiction"]
page = st.sidebar.radio("Aller à", pages)

# Variables de session pour stocker les données
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_pipeline' not in st.session_state:
    st.session_state.model_pipeline = None

# Page d'accueil
if page == "Accueil":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ## Application de prédiction de récupération métallurgique
    
    Cette application vous permet de prédire la récupération métallurgique en utilisant des données de tests géométallurgiques et des algorithmes de machine learning.
    
    ### Fonctionnalités principales
    
    - **Importation de données**: Téléchargez vos données géométallurgiques au format CSV ou Excel
    - **Exploration des données**: Visualisez les corrélations et tendances dans vos données
    - **Modélisation**: Entraînez et évaluez différents modèles de machine learning
    - **Prédiction**: Utilisez le modèle entraîné pour prédire la récupération métallurgique de nouveaux échantillons
    
    ### Comment utiliser cette application
    
    1. Importez vos données dans l'onglet "Importation des données"
    2. Explorez vos données dans l'onglet "Exploration des données"
    3. Créez un modèle prédictif dans l'onglet "Modélisation"
    4. Faites des prédictions sur de nouveaux échantillons dans l'onglet "Prédiction"
    
    ### Données requises
    
    Pour obtenir les meilleurs résultats, vos données doivent inclure:
    - Propriétés minéralogiques (teneur, taille des grains, etc.)
    - Paramètres de traitement (pH, densité de pulpe, etc.)
    - Résultats des tests (récupération métallurgique)
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Exemple de dataset
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Exemple de dataset")
    st.markdown("Si vous n'avez pas de données, vous pouvez utiliser notre jeu de données d'exemple:")
    
    # Création d'un jeu de données d'exemple
    np.random.seed(42)
    n_samples = 100
    
    example_data = pd.DataFrame({
        'Teneur_Cu': np.random.uniform(0.5, 3.0, n_samples),
        'Teneur_Au': np.random.uniform(0.1, 1.5, n_samples),
        'Taille_Grain': np.random.uniform(10, 150, n_samples),
        'Durete': np.random.uniform(1, 5, n_samples),
        'pH': np.random.uniform(7, 11, n_samples),
        'Temps_Flottation': np.random.uniform(5, 15, n_samples),
        'Densite_Pulpe': np.random.uniform(20, 40, n_samples),
        'Recuperation': np.random.uniform(60, 95, n_samples)
    })
    
    # Ajout d'une relation pour simuler des données réalistes
    example_data['Recuperation'] = 70 + 5 * example_data['Teneur_Cu'] - 3 * example_data['Taille_Grain'] / 100 + 2 * example_data['pH'] + np.random.normal(0, 3, n_samples)
    example_data['Recuperation'] = example_data['Recuperation'].clip(60, 95)
    
    st.dataframe(example_data.head())
    
    if st.button("Utiliser cet exemple de données"):
        st.session_state.data = example_data
        st.success("Données d'exemple chargées avec succès!")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
# Page d'importation des données
elif page == "Importation des données":
    st.markdown("<h2 class='sub-header'>Importation des Données</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ### Instructions
    
    Téléchargez votre fichier de données géométallurgiques au format CSV ou Excel.
    Assurez-vous que votre fichier contient:
    
    - Une variable cible (récupération métallurgique)
    - Des variables prédictives (propriétés minéralogiques, paramètres de traitement, etc.)
    """)
    
    upload_option = st.radio("Choisissez une option", ["Télécharger un fichier", "Utiliser l'exemple de données"])
    
    if upload_option == "Télécharger un fichier":
        uploaded_file = st.file_uploader("Télécharger votre fichier de données", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.success(f"Fichier chargé avec succès! Dimensions: {data.shape[0]} lignes × {data.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
    else:
        if st.button("Charger les données d'exemple"):
            # Création d'un jeu de données d'exemple (comme dans la page d'accueil)
            np.random.seed(42)
            n_samples = 100
            
            example_data = pd.DataFrame({
                'Teneur_Cu': np.random.uniform(0.5, 3.0, n_samples),
                'Teneur_Au': np.random.uniform(0.1, 1.5, n_samples),
                'Taille_Grain': np.random.uniform(10, 150, n_samples),
                'Durete': np.random.uniform(1, 5, n_samples),
                'pH': np.random.uniform(7, 11, n_samples),
                'Temps_Flottation': np.random.uniform(5, 15, n_samples),
                'Densite_Pulpe': np.random.uniform(20, 40, n_samples),
                'Recuperation': np.random.uniform(60, 95, n_samples)
            })
            
            # Ajout d'une relation pour simuler des données réalistes
            example_data['Recuperation'] = 70 + 5 * example_data['Teneur_Cu'] - 3 * example_data['Taille_Grain'] / 100 + 2 * example_data['pH'] + np.random.normal(0, 3, n_samples)
            example_data['Recuperation'] = example_data['Recuperation'].clip(60, 95)
            
            st.session_state.data = example_data
            st.success("Données d'exemple chargées avec succès!")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Affichage des données si elles sont disponibles
    if st.session_state.data is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Aperçu des Données")
        
        data = st.session_state.data
        st.dataframe(data.head())
        
        st.markdown("### Statistiques Descriptives")
        st.dataframe(data.describe())
        
        st.markdown("### Informations sur les Données")
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        # Vérification des valeurs manquantes
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            st.warning("Votre jeu de données contient des valeurs manquantes:")
            st.dataframe(missing_values[missing_values > 0])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Option pour télécharger les données nettoyées
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Préparation des Données")
        
        # Nettoyage des données
        if st.checkbox("Nettoyer automatiquement les données"):
            # Supprimer les lignes avec valeurs manquantes
            data_cleaned = data.dropna()
            
            # Afficher les résultats du nettoyage
            st.write(f"Données avant nettoyage: {data.shape[0]} lignes")
            st.write(f"Données après nettoyage: {data_cleaned.shape[0]} lignes")
            
            # Option pour remplacer le jeu de données original
            if st.button("Utiliser les données nettoyées"):
                st.session_state.data = data_cleaned
                st.success("Données nettoyées appliquées avec succès!")
                
            # Lien de téléchargement pour les données nettoyées
            st.markdown(get_download_link(data_cleaned, "donnees_nettoyees.csv", "Télécharger les données nettoyées"), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Page d'exploration des données
elif page == "Exploration des données":
    st.markdown("<h2 class='sub-header'>Exploration des Données</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez aller à la page 'Importation des données' pour charger vos données.")
    else:
        data = st.session_state.data
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Sélection de la Variable Cible")
        
        # Sélection de la variable cible
        target_col = st.selectbox("Sélectionnez la variable cible (récupération métallurgique)", data.columns)
        st.session_state.target = target_col
        
        # Sélection des features
        st.markdown("### Sélection des Variables Prédictives")
        feature_cols = st.multiselect("Sélectionnez les variables prédictives", 
                                     [col for col in data.columns if col != target_col],
                                     default=[col for col in data.columns if col != target_col])
        st.session_state.features = feature_cols
        
        if not feature_cols:
            st.error("Veuillez sélectionner au moins une variable prédictive.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if feature_cols:
            # Visualisation des distributions
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Distribution des Variables")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.subheader("Distribution de la Variable Cible")
                fig = px.histogram(data, x=target_col, nbins=20, 
                                   title=f"Distribution de {target_col}",
                                   labels={target_col: target_col},
                                   color_discrete_sequence=['#3498db'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats de la variable cible
                st.markdown(f"**Statistiques pour {target_col}:**")
                st.write(f"Moyenne: {data[target_col].mean():.2f}")
                st.write(f"Médiane: {data[target_col].median():.2f}")
                st.write(f"Écart-type: {data[target_col].std():.2f}")
                st.write(f"Min: {data[target_col].min():.2f}")
                st.write(f"Max: {data[target_col].max():.2f}")
                
            with viz_col2:
                # Graphique pour les distributions des variables explicatives
                var_to_plot = st.selectbox("Sélectionnez une variable à visualiser", feature_cols)
                fig = px.histogram(data, x=var_to_plot, nbins=20,
                                  title=f"Distribution de {var_to_plot}",
                                  labels={var_to_plot: var_to_plot},
                                  color_discrete_sequence=['#2ecc71'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Matrice de corrélation
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Analyse des Corrélations")
            
            # Calcul de la matrice de corrélation
            selected_cols = feature_cols + [target_col]
            corr_matrix = data[selected_cols].corr()
            
            # Visualisation de la matrice de corrélation avec plotly
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           color_continuous_scale="RdBu_r",
                           title="Matrice de Corrélation",
                           zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Corrélations avec la variable cible
            st.subheader(f"Corrélations avec {target_col}")
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
            
            fig = px.bar(x=target_corr.index, y=target_corr.values,
                        labels={'x': 'Variables', 'y': 'Coefficient de Corrélation'},
                        title=f"Corrélations avec {target_col}",
                        color=target_corr.values,
                        color_continuous_scale="RdBu_r")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Graphiques de dispersion
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Graphiques de Dispersion")
            
            # Sélectionner les variables pour le graphique de dispersion
            x_var = st.selectbox("Sélectionnez la variable pour l'axe X", feature_cols)
            
            fig = px.scatter(data, x=x_var, y=target_col, 
                            title=f"{target_col} vs {x_var}",
                            trendline="ols",
                            labels={x_var: x_var, target_col: target_col})
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique à variables multiples
            st.subheader("Graphique à variables multiples")
            col_subset = st.multiselect("Sélectionnez 2 à 4 variables à visualiser", 
                                       feature_cols, 
                                       default=feature_cols[:min(3, len(feature_cols))])
            
            if len(col_subset) >= 2:
                if len(col_subset) <= 4:
                    color_var = st.selectbox("Variable pour la couleur", col_subset + [None], index=len(col_subset))
                    
                    if color_var is not None:
                        fig = px.scatter_matrix(data, 
                                              dimensions=col_subset,
                                              color=color_var,
                                              title="Graphique de dispersion multiple")
                    else:
                        fig = px.scatter_matrix(data, 
                                              dimensions=col_subset,
                                              title="Graphique de dispersion multiple")
                        
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Veuillez sélectionner au maximum 4 variables pour la visualisation.")
            st.markdown("</div>", unsafe_allow_html=True)

# Page de modélisation
elif page == "Modélisation":
    st.markdown("<h2 class='sub-header'>Modélisation</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez aller à la page 'Importation des données' pour charger vos données.")
    elif st.session_state.features is None or st.session_state.target is None:
        st.warning("Variables prédictives et/ou variable cible non définies. Veuillez aller à la page 'Exploration des données'.")
    else:
        data = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Configuration du Modèle")
        
        # Division train/test
        test_size = st.slider("Pourcentage des données pour le test", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Seed aléatoire", 0, 1000, 42)
        
        # Prétraitement
        preprocessing = st.checkbox("Normaliser les données", True)
        
        # Sélection du modèle
        model_type = st.selectbox("Sélectionnez le type de modèle", 
                                 ["Random Forest", "Gradient Boosting", "Régression Linéaire"])
        
        # Configuration spécifique au modèle
        if model_type == "Random Forest":
            n_estimators = st.slider("Nombre d'arbres", 10, 500, 100)
            max_depth = st.slider("Profondeur maximale", 2, 30, 10)
            
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state
            }
            
            model = RandomForestRegressor(**model_params)
            
        elif model_type == "Gradient Boosting":
            n_estimators = st.slider("Nombre d'estimateurs", 10, 500, 100)
            learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01)
            max_depth = st.slider("Profondeur maximale", 2, 10, 3)
            
            model_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "random_state": random_state
            }
            
            model = GradientBoostingRegressor(**model_params)
            
        else:  # Régression Linéaire
            model = LinearRegression()
        
        # Préparation des données
        X = data[features]
        y = data[target]
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Construction du pipeline
        steps = []
        
        if preprocessing:
            scaler = StandardScaler()
            steps.append(('scaler', scaler))
            st.session_state.scaler = scaler
        
        steps.append(('model', model))
        pipeline = Pipeline(steps)
        
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.model = model
        st.session_state.model_pipeline = pipeline
        
        # Entraînement du modèle
        if st.button("Entraîner le modèle"):
            with st.spinner("Entraînement du modèle en cours..."):
                pipeline.fit(X_train, y_train)
                
                # Évaluation du modèle
                y_pred = pipeline.predict(X_test)
                
                # Métriques
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Validation croisée
                cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
                
                st.success("Modèle entraîné avec succès!")
                
                # Affichage des métriques dans un tableau
                metrics_data = {
                    "Métrique": ["MSE", "RMSE", "MAE", "R²", "R² CV (moyenne)", "R² CV (écart-type)"],
                    "Valeur": [mse, rmse, mae, r2, cv_scores.mean(), cv_scores.std()]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df)
                
                # Visualisation des prédictions vs valeurs réelles
                fig = px.scatter(
                    x=y_test, y=y_pred, 
                    labels={"x": "Valeurs Réelles", "y": "Valeurs Prédites"},
                    title="Prédictions vs Valeurs Réelles"
                )
                
                # Ajout de la ligne de référence (y=x)
                fig.add_trace(
                    go.Scatter(
                        x=[y_test.min(), y_test.max()], 
                        y=[y_test.min(), y_test.max()],
                        mode="lines", 
                        line=dict(color="red", dash="dash"),
                        name="Ligne de référence"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Importance des variables
                if model_type in ["Random Forest", "Gradient Boosting"]:
                    if preprocessing:
                        # Pour les pipelines avec prétraitement, on obtient l'importance directement du modèle
                        importances = pipeline.named_steps['model'].feature_importances_
                    else:
                        importances = model.feature_importances_
                    
                    importance_df = pd.DataFrame({
                        'Variable': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df, 
                        x='Variable', 
                        y='Importance',
                        title="Importance des Variables"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # SHAP values pour l'interprétabilité
                    if st.checkbox("Afficher l'analyse SHAP (interprétabilité avancée)"):
                        with st.spinner("Calcul des valeurs SHAP en cours..."):
                            # Création de l'explainer SHAP
                            if preprocessing:
                                # Pour les données normalisées, on doit d'abord transformer les données
                                X_test_transformed = pipeline.named_steps['scaler'].transform(X_test)
                                explainer = shap.TreeExplainer(pipeline.named_steps['model'])
                                shap_values = explainer.shap_values(X_test_transformed)
                            else:
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(X_test)
                            
                            # Conversion du plot SHAP en figure matplotlib
                            st.subheader("Graphique de résumé SHAP")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            if preprocessing:
                                shap.summary_plot(shap_values, X_test_transformed, feature_names=features, show=False)
                            else:
                                shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
                            st.pyplot(fig)
                            plt.clf()
                            
                            # Graphique de dépendance SHAP
                            if len(features) > 0:
                                st.subheader("Graphique de dépendance SHAP")
                                feature_to_plot = st.selectbox("Sélectionnez une variable pour le graphique de dépendance", features)
                                
                                feature_idx = features.index(feature_to_plot)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                if preprocessing:
                                    shap.dependence_plot(feature_idx, shap_values, X_test_transformed, feature_names=features, show=False)
                                else:
                                    shap.dependence_plot(feature_idx, shap_values, X_test, feature_names=features, show=False)
                                st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# Page de prédiction
elif page == "Prédiction":
    st.markdown("<h2 class='sub-header'>Prédiction</h2>", unsafe_allow_html=True)
    
    if st.session_state.model_pipeline is None:
        st.warning("Aucun modèle n'a été entraîné. Veuillez aller à la page 'Modélisation' pour entraîner un modèle.")
    elif st.session_state.features is None:
        st.warning("Variables prédictives non définies. Veuillez aller à la page 'Exploration des données'.")
    else:
        pipeline = st.session_state.model_pipeline
        features = st.session_state.features
        target = st.session_state.target
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Prédiction pour un nouvel échantillon")
        
        # Méthode pour rentrer les valeurs
        input_method = st.radio("Méthode d'entrée des données", ["Saisie manuelle", "Importer un fichier"])
        
        if input_method == "Saisie manuelle":
            # Création d'un formulaire pour entrer les valeurs des features
            input_data = {}
            
            for feature in features:
                # Obtenir la plage des valeurs dans les données d'entraînement pour définir les min/max
                if st.session_state.X_train is not None:
                    min_val = float(st.session_state.X_train[feature].min())
                    max_val = float(st.session_state.X_train[feature].max())
                    mean_val = float(st.session_state.X_train[feature].mean())
                    
                    # Ajuster légèrement les min/max pour éviter les problèmes de types
                    min_val = min_val * 0.9 if min_val > 0 else min_val * 1.1
                    max_val = max_val * 1.1 if max_val > 0 else max_val * 0.9
                    
                    input_data[feature] = st.slider(
                        f"{feature}", 
                        min_val, 
                        max_val, 
                        mean_val
                    )
                else:
                    input_data[feature] = st.number_input(f"{feature}")
            
            # Bouton pour faire la prédiction
            if st.button("Prédire"):
                # Préparation des données d'entrée
                input_df = pd.DataFrame([input_data])
                
                # Faire la prédiction
                prediction = pipeline.predict(input_df)[0]
                
                # Afficher la prédiction
                st.success(f"Prédiction de la récupération métallurgique: **{prediction:.2f}%**")
                
                # Jauge pour visualiser la prédiction
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Prédiction de {target}"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, 50], 'color': "#e74c3c"},
                            {'range': [50, 80], 'color': "#f39c12"},
                            {'range': [80, 100], 'color': "#2ecc71"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
        else:  # Importation d'un fichier
            st.markdown("""
            Téléchargez un fichier CSV ou Excel contenant les données pour lesquelles vous souhaitez faire des prédictions.
            Le fichier doit contenir les colonnes suivantes:
            """)
            
            # Afficher les features requises
            for feature in features:
                st.markdown(f"- {feature}")
            
            # Upload du fichier
            uploaded_file = st.file_uploader("Télécharger votre fichier de données", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    # Chargement des données
                    if uploaded_file.name.endswith('.csv'):
                        predict_data = pd.read_csv(uploaded_file)
                    else:
                        predict_data = pd.read_excel(uploaded_file)
                    
                    # Vérifier que toutes les colonnes requises sont présentes
                    missing_cols = [col for col in features if col not in predict_data.columns]
                    
                    if missing_cols:
                        st.error(f"Colonnes manquantes dans le fichier: {', '.join(missing_cols)}")
                    else:
                        # Extraire seulement les colonnes nécessaires
                        predict_data = predict_data[features]
                        
                        # Faire les prédictions
                        predictions = pipeline.predict(predict_data)
                        
                        # Ajouter les prédictions au dataframe
                        results = predict_data.copy()
                        results[f"{target}_predit"] = predictions
                        
                        # Afficher les résultats
                        st.subheader("Résultats des prédictions")
                        st.dataframe(results)
                        
                        # Histogramme des prédictions
                        fig = px.histogram(
                            results, 
                            x=f"{target}_predit", 
                            nbins=20,
                            title=f"Distribution des prédictions de {target}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Option pour télécharger les résultats
                        csv = results.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="resultats_predictions.csv">Télécharger les résultats des prédictions</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Section pour la simulation et l'optimisation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Optimisation des Paramètres")
        
        st.markdown("""
        Vous pouvez utiliser cette section pour explorer l'effet de différents paramètres sur la récupération métallurgique 
        et identifier les paramètres optimaux pour maximiser la récupération.
        """)
        
        # Sélection des paramètres à optimiser
        params_to_optimize = st.multiselect(
            "Sélectionnez les paramètres à optimiser",
            features,
            default=features[:min(2, len(features))]
        )
        
        if len(params_to_optimize) >= 1:
            # Valeurs par défaut pour les paramètres qui ne sont pas optimisés
            default_values = {}
            for feature in features:
                if feature not in params_to_optimize:
                    if st.session_state.X_train is not None:
                        default_values[feature] = float(st.session_state.X_train[feature].mean())
                    else:
                        default_values[feature] = 0.0
            
            # Créer des sliders pour les plages des paramètres à optimiser
            param_ranges = {}
            
            for param in params_to_optimize:
                if st.session_state.X_train is not None:
                    min_val = float(st.session_state.X_train[param].min())
                    max_val = float(st.session_state.X_train[param].max())
                    
                    # Ajuster légèrement les min/max pour éviter les problèmes de types
                    min_val = min_val * 0.9 if min_val > 0 else min_val * 1.1
                    max_val = max_val * 1.1 if max_val > 0 else max_val * 0.9
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        param_min = st.number_input(f"Min pour {param}", value=min_val)
                    with col2:
                        param_max = st.number_input(f"Max pour {param}", value=max_val)
                    
                    param_ranges[param] = (param_min, param_max)
                else:
                    param_ranges[param] = (0, 10)  # Valeurs par défaut
            
            # Bouton pour lancer la simulation
            if st.button("Lancer la simulation"):
                with st.spinner("Simulation en cours..."):
                    if len(params_to_optimize) == 1:
                        # Simulation 1D
                        param = params_to_optimize[0]
                        param_range = np.linspace(param_ranges[param][0], param_ranges[param][1], 100)
                        
                        sim_results = []
                        for val in param_range:
                            input_data = default_values.copy()
                            input_data[param] = val
                            input_df = pd.DataFrame([input_data])
                            prediction = pipeline.predict(input_df)[0]
                            sim_results.append(prediction)
                        
                        # Visualisation
                        fig = px.line(
                            x=param_range, 
                            y=sim_results,
                            labels={"x": param, "y": f"Prédiction de {target}"},
                            title=f"Effet de {param} sur {target}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Trouver la valeur optimale
                        optimal_idx = np.argmax(sim_results)
                        optimal_val = param_range[optimal_idx]
                        optimal_result = sim_results[optimal_idx]
                        
                        st.success(f"Valeur optimale de {param}: {optimal_val:.2f} → {optimal_result:.2f}% de récupération")
                        
                    elif len(params_to_optimize) == 2:
                        # Simulation 2D
                        param1, param2 = params_to_optimize
                        param1_range = np.linspace(param_ranges[param1][0], param_ranges[param1][1], 30)
                        param2_range = np.linspace(param_ranges[param2][0], param_ranges[param2][1], 30)
                        
                        # Créer une grille de valeurs
                        param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
                        sim_results = np.zeros_like(param1_grid)
                        
                        # Calculer les prédictions pour chaque combinaison
                        for i in range(len(param1_range)):
                            for j in range(len(param2_range)):
                                input_data = default_values.copy()
                                input_data[param1] = param1_grid[j, i]
                                input_data[param2] = param2_grid[j, i]
                                input_df = pd.DataFrame([input_data])
                                sim_results[j, i] = pipeline.predict(input_df)[0]
                        
                        # Visualisation de la surface de réponse
                        fig = go.Figure(data=[go.Surface(z=sim_results, x=param1_range, y=param2_range)])
                        fig.update_layout(
                            title=f"Surface de réponse pour {target}",
                            scene=dict(
                                xaxis_title=param1,
                                yaxis_title=param2,
                                zaxis_title=target
                            ),
                            width=700,
                            height=700
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Graphique de contour pour une visualisation plus claire
                        fig = px.contour(
                            x=param1_range, 
                            y=param2_range, 
                            z=sim_results,
                            labels=dict(x=param1, y=param2, z=target),
                            title=f"Contours de {target}",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Trouver les valeurs optimales
                        optimal_idx = np.unravel_index(np.argmax(sim_results), sim_results.shape)
                        optimal_val1 = param1_range[optimal_idx[1]]
                        optimal_val2 = param2_range[optimal_idx[0]]
                        optimal_result = sim_results[optimal_idx]
                        
                        st.success(f"Valeurs optimales: {param1}={optimal_val1:.2f}, {param2}={optimal_val2:.2f} → {optimal_result:.2f}% de récupération")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Développé par Didier Ouedraogo, P.Geo - Application de Prédiction de Récupération Métallurgique © 2025</div>", unsafe_allow_html=True)