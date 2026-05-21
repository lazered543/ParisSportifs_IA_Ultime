from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="IA Paris Sportifs Ultime", layout="wide")
st.title("IA Paris Sportifs Ultime - Football + Tennis")

pred_path = Path("data/predictions/predictions_today.csv")
value_path = Path("data/predictions/value_bets_today.csv")

if not pred_path.exists():
    st.warning("Aucune prédiction trouvée. Lance d'abord run_full_pipeline.bat")
    st.stop()

df = pd.read_csv(pred_path)
values = pd.read_csv(value_path) if value_path.exists() else pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs / marchés analysés", len(df))
c2.metric("Value bets", len(values))
c3.metric("Meilleure value", f"{df['value'].max():.2%}" if not df.empty else "0%")
c4.metric("Mise max conseillée", f"{df['suggested_stake'].max():.2f} €" if not df.empty else "0 €")

st.subheader("Paris à considérer selon le modèle")
if values.empty:
    st.info("Aucun value bet selon les filtres actuels. Ne force pas un pari.")
else:
    st.dataframe(values.sort_values("value", ascending=False), use_container_width=True)

st.subheader("Toutes les prédictions")
st.dataframe(df.sort_values("value", ascending=False), use_container_width=True)

st.subheader("Graphiques")
if not df.empty:
    fig = px.bar(df.sort_values("value", ascending=False).head(20), x="market", y="value", color="decision",
                 hover_data=["home_team","away_team","bookmaker_odds","ai_probability"])
    st.plotly_chart(fig, use_container_width=True)

st.caption("Aucun pari n'est sûr. Le dashboard sort des value bets statistiques, à valider avant de miser.")
