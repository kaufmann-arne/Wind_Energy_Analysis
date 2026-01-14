import pandas as pd

# Load the three Bundesland tables
planning = pd.read_csv("planning_phase_bundesland.csv")
permitting = pd.read_csv("permitting_phase_bundesland.csv")
total = pd.read_csv("total_bundesland.csv")

# Remove rows without projects or without mean
planning = planning.dropna(subset=["mean_eur_per_kw", "projects"])
permitting = permitting.dropna(subset=["mean_eur_per_kw", "projects"])
total = total.dropna(subset=["mean_eur_per_kw", "projects"])

def weighted_mean(df):
    df = df.dropna(subset=["mean_eur_per_kw", "projects"])
    return (df["mean_eur_per_kw"] * df["projects"]).sum() / df["projects"].sum()

# Compute national values
national_planning = weighted_mean(planning)
national_permitting = weighted_mean(permitting)
national_total = weighted_mean(total)

# Derive precheck
national_precheck = national_total - national_planning - national_permitting

# Create output table
national_costs = pd.DataFrame({
    "phase": ["Vorpruefung", "Planung", "Genehmigung", "Total_bis_Genehmigung"],
    "mean_eur_per_kw": [
        national_precheck,
        national_planning,
        national_permitting,
        national_total
    ]
})

national_costs.to_csv("national_wind_project_development_costs.csv", index=False)

print(national_costs)
