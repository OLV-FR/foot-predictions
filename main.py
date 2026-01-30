import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# === CHARGEMENT DES DONNÉES (multi-ligues, saisons récentes) ===
st.write("Chargement des données historiques...")

leagues = {
    'EPL': 'E0',      # England Premier League
    'Ligue1': 'F1',   # France Ligue 1
    'LaLiga': 'SP1',  # Spain La Liga
    'Bundesliga': 'D1',  # Germany Bundesliga
    'Primeira': 'P1'  # Portugal Primeira Liga
}

seasons = ['2324', '2425', '2526']  # 2023-2026

urls = [f'https://www.football-data.co.uk/mmz4281/{s}/{code}.csv' for s in seasons for code in leagues.values()]

dfs = []
for url in urls:
    try:
        df_temp = pd.read_csv(url)
        dfs.append(df_temp)
    except:
        pass

if dfs:
    df = pd.concat(dfs, ignore_index=True)
    df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')  # Pour trier par date
    df = df.sort_values('Date')
else:
    st.error("Impossible de charger les données pour l'instant. Réessaie plus tard.")
    st.stop()

# === CALCUL DES STATS DES ÉQUIPES ===
teams = sorted(pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel()))
home_attack = df.groupby('HomeTeam')['FTHG'].mean()
home_defense = df.groupby('HomeTeam')['FTAG'].mean()
away_attack = df.groupby('AwayTeam')['FTAG'].mean()
away_defense = df.groupby('AwayTeam')['FTHG'].mean()

avg_home = df['FTHG'].mean()
avg_away = df['FTAG'].mean()

# === FONCTION DE PRÉDICTION ===
def predict_match(home_team, away_team):
    if home_team not in home_attack or away_team not in away_defense:
        return 33.0, 33.0, 34.0, 1.5, 1.2, 50.0  # Defaults + over2.5
    
    # Buts attendus
    exp_home = home_attack[home_team] * away_defense[away_team] / avg_home
    exp_away = away_attack[away_team] * home_defense[home_team] / avg_away
    
    # Probabilités avec loi de Poisson
    max_g = 10
    home_win_p, draw_p, away_win_p, over_25_p = 0, 0, 0, 0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson.pmf(h, exp_home) * poisson.pmf(a, exp_away)
            if h > a:
                home_win_p += p
            elif h == a:
                draw_p += p
            else:
                away_win_p += p
            if h + a > 2:
                over_25_p += p
    
    return (
        round(home_win_p*100, 1), round(draw_p*100, 1), round(away_win_p*100, 1),
        round(exp_home, 1), round(exp_away, 1), round(over_25_p*100, 1)
    )

# === INTERFACE DU SITE ===
st.title("⚽ Prédictions Foot IA - Multi-Ligues")
st.markdown("Sélectionne deux équipes (de la même ligue pour plus de précision) et clique sur **Prédire** ! (basé sur stats 2023-2026)")

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Équipe à domicile", teams)
with col2:
    away = st.selectbox("Équipe à l'extérieur", teams, index=1 if len(teams)>1 else 0)

if st.button("🚀 Prédire le match !", type="primary"):
    hw, d, aw, eh, ea, o25 = predict_match(home, away)
    st.balloons()
    
    st.markdown(f"### {home} vs {away}")
    st.markdown(f"**{home} gagne : {hw}%**")
    st.markdown(f"**Match nul : {d}%**")
    st.markdown(f"**{away} gagne : {aw}%**")
    st.markdown(f"**Buts attendus : {home} {eh} - {ea} {away}**")
    st.markdown(f"**Over 2.5 buts : {o25}%**")
    st.markdown(f"**Under 2.5 buts : {100 - o25}%**")
    
    # Graphique probs résultats
    st.bar_chart(pd.DataFrame({"Résultat": ["Domicile", "Nul", "Extérieur"], "Probabilité (%)": [hw, d, aw]}).set_index("Résultat"))
    
    # Tableaux stats
    st.subheader("Head-to-Head (derniers 5 matchs)")
    h2h = df[((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) | ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))].tail(5)
    if not h2h.empty:
        st.dataframe(h2h[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']])
    else:
        st.info("Pas de matchs head-to-head récents.")
    
    st.subheader(f"Derniers 5 matchs de {home}")
    home_last = df[(df['HomeTeam'] == home) | (df['AwayTeam'] == home)].tail(5)
    if not home_last.empty:
        st.dataframe(home_last[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']])
    else:
        st.info("Pas de matchs récents pour cette équipe.")
    
    st.subheader(f"Derniers 5 matchs de {away}")
    away_last = df[(df['HomeTeam'] == away) | (df['AwayTeam'] == away)].tail(5)
    if not away_last.empty:
        st.dataframe(away_last[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']])
    else:
        st.info("Pas de matchs récents pour cette équipe.")

st.caption("Ton site est prêt ! Teste avec 'Run', modifie si besoin, puis 'Deploy' pour partager.")
