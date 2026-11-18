"""
Machine Learning and Advanced Modelling Module - RR SuperRR Selector Season 2
Supervised Learning (Outcome Prediction), Permutation Importance, and Player Clustering.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, classification_report
from sklearn.inspection import permutation_importance
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r"c:\Atharva\RR-SuperRR-Selector-2027"
PROCESSED = os.path.join(BASE, "data", "processed")
OUTPUTS = os.path.join(BASE, "outputs")
CHARTS = os.path.join(OUTPUTS, "charts")
REPORTS = os.path.join(OUTPUTS, "reports")

def run_ml_pipeline():
    print("Running Machine Learning Pipeline...")
    bbb = pd.read_parquet(os.path.join(PROCESSED, "bbb_with_pressure.parquet"))
    
    # Filter for relevant deliveries (IPL + T20I, legal balls)
    df = bbb[(bbb["is_ipl"] | bbb["is_t20i"]) & ~bbb["is_super_over"] & bbb["is_legal_ball"]].copy()
    
    # -------------------------------------------------------------
    # 1. Supervised Learning: Pressure Delivery Success Prediction
    # Target for batter: boundary hit (is_four or is_six) OR dot ball forced (target for bowler)
    # Let's model Batter Pressure Scoring: Predicting High-Impact Shot (4 or 6) under Pressure
    # -------------------------------------------------------------
    press_df = df[df["batting_pressure_score"] >= 50].copy()
    press_df["target_boundary"] = (press_df["is_four"] | press_df["is_six"]).astype(int)
    
    # Features available before delivery
    features = [
        "batting_pressure_score", "bowling_pressure_score", "balls_remaining",
        "innings_stage", "overid", "wickets_remaining"
    ]
    if "rrr" in press_df.columns:
        press_df["rrr_filled"] = press_df["rrr"].fillna(press_df["crr"])
        features.append("rrr_filled")
        
    X = press_df[features].fillna(0)
    y = press_df["target_boundary"]
    
    # Chronological Train-Test Split based on match_id
    matches = press_df["match_id"].unique()
    split_idx = int(len(matches) * 0.75)
    train_matches = set(matches[:split_idx])
    test_matches = set(matches[split_idx:])
    
    train_mask = press_df["match_id"].isin(train_matches)
    test_mask = press_df["match_id"].isin(test_matches)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
    }
    
    model_eval = {}
    fitted_models = {}
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds_proba = m.predict_proba(X_test)[:, 1]
        preds = (preds_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, preds_proba)
        brier = brier_score_loss(y_test, preds_proba)
        loss = log_loss(y_test, preds_proba)
        
        model_eval[name] = {
            "ROC_AUC": round(float(auc), 4),
            "Brier_Score": round(float(brier), 4),
            "Log_Loss": round(float(loss), 4)
        }
        fitted_models[name] = m
        print(f"Model {name} - AUC: {auc:.4f}, Brier: {brier:.4f}, LogLoss: {loss:.4f}")
        
    # Save Model Evaluation Report
    with open(os.path.join(REPORTS, "model_evaluation_report.json"), "w") as f:
        json.dump(model_eval, f, indent=2)
        
    # Feature Importance via XGBoost & Random Forest
    rf_feat_imp = dict(zip(features, [round(float(v), 4) for v in fitted_models["RandomForest"].feature_importances_]))
    xgb_feat_imp = dict(zip(features, [round(float(v), 4) for v in fitted_models["XGBoost"].feature_importances_]))
    
    # Permutation importance
    perm_imp = permutation_importance(fitted_models["RandomForest"], X_test, y_test, n_repeats=5, random_state=42)
    perm_dict = dict(zip(features, [round(float(v), 4) for v in perm_imp.importances_mean]))
    
    explainability = {
        "RandomForest_Feature_Importance": rf_feat_imp,
        "XGBoost_Feature_Importance": xgb_feat_imp,
        "Permutation_Importance_RF": perm_dict
    }
    with open(os.path.join(REPORTS, "model_explainability_report.json"), "w") as f:
        json.dump(explainability, f, indent=2)
        
    # -------------------------------------------------------------
    # 2. Player Clustering & Archetypes (K-Means)
    # -------------------------------------------------------------
    bat_ranked = pd.read_csv(os.path.join(OUTPUTS, "rankings", "batters_final_ranked.csv"))
    bowl_ranked = pd.read_csv(os.path.join(OUTPUTS, "rankings", "bowlers_final_ranked.csv"))
    
    # Batting Clustering
    bat_cluster_feats = ["comb_press_sr", "comb_press_avg", "cons", "success_rate", "sr_improvement"]
    scaler_bat = StandardScaler()
    X_bat_cl = scaler_bat.fit_transform(bat_ranked[bat_cluster_feats].fillna(0))
    
    kmeans_bat = KMeans(n_clusters=4, random_state=42, n_init=10)
    bat_ranked["cluster"] = kmeans_bat.fit_predict(X_bat_cl)
    cluster_names_bat = {
        0: "Consistent Anchors under Pressure",
        1: "High-Impact Power Finishers",
        2: "Adaptive Middle-Order Stabilizers",
        3: "High-Volatility Accelerators"
    }
    bat_ranked["archetype"] = bat_ranked["cluster"].map(cluster_names_bat)
    bat_ranked.to_csv(os.path.join(OUTPUTS, "rankings", "batters_final_ranked.csv"), index=False)
    
    # Bowling Clustering
    bowl_cluster_feats = ["comb_press_economy", "comb_press_wickets", "bowl_cons", "good_spell_rate", "eco_improvement"]
    scaler_bowl = StandardScaler()
    X_bowl_cl = scaler_bowl.fit_transform(bowl_ranked[bowl_cluster_feats].fillna(0))
    
    kmeans_bowl = KMeans(n_clusters=4, random_state=42, n_init=10)
    bowl_ranked["cluster"] = kmeans_bowl.fit_predict(X_bowl_cl)
    cluster_names_bowl = {
        0: "Death-Over Restrictors",
        1: "Elite Strike Wicket-Takers",
        2: "Defensive Containment Specialists",
        3: "Variable Impact Bowlers"
    }
    bowl_ranked["archetype"] = bowl_ranked["cluster"].map(cluster_names_bowl)
    bowl_ranked.to_csv(os.path.join(OUTPUTS, "rankings", "bowlers_final_ranked.csv"), index=False)
    
    # Visualise Clustering
    plt.figure(figsize=(10, 6))
    plt.scatter(bat_ranked["comb_press_sr"], bat_ranked["comb_press_avg"], c=bat_ranked["cluster"], cmap="viridis", s=60, alpha=0.8)
    for _, r in bat_ranked.head(10).iterrows():
        plt.annotate(r["player_name"].split()[-1], (r["comb_press_sr"], r["comb_press_avg"]), fontsize=9, color="red", fontweight="bold")
    plt.title("Batter Pressure Profiles: K-Means Clustering", fontsize=14, fontweight="bold")
    plt.xlabel("Pressure Strike Rate")
    plt.ylabel("Pressure Batting Average")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CHARTS, "09_batting_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Machine learning pipeline finished and reports saved.")

if __name__ == "__main__":
    run_ml_pipeline()

