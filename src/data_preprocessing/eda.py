import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


PROCESSED_DATA_PATH = Path("data/processed")
EDA_PATH = PROCESSED_DATA_PATH / "eda"
EDA_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(name):
    """Helper to save figures consistently"""
    plt.savefig(EDA_PATH / f"{name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_target_distribution(df):
    """Plot distribution of the target variable (disposition)"""
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x="disposition_edstays", data=df,
                      order=df["disposition_edstays"].value_counts().index,
                      palette="viridis", hue="disposition_edstays", legend=False)
    plt.xticks(rotation=45)
    plt.title("Distribution of Disposition Outcomes")
    
    # percentages annotation
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    save_fig("target_distribution")


def plot_vital_signs_by_disposition(df):
    """Plot distributions of vital signs by disposition"""
    vital_signs = [
        ('heartrate', 'Heart Rate'),
        ('resprate', 'Respiratory Rate'),
        ('o2sat', 'O2 Saturation'),
        ('sbp', 'Systolic BP'),
        ('dbp', 'Diastolic BP'),
        ('temperature', 'Temperature'),
        ('pain', 'Pain Score')
    ]
    
    for sign, label in vital_signs:
        for source in ['triage', 'vitalsign']:
            col = f"{sign}_{source}"
            plt.figure(figsize=(10, 6))
            sns.boxplot(x="disposition_edstays", y=col, data=df, 
                        palette="Set3", showfliers=False, 
                        hue="disposition_edstays", legend=False)
            plt.xticks(rotation=45)
            plt.title(f"{label} ({source}) Distribution by Disposition")
            plt.tight_layout()
            save_fig(f"{sign}_{source}_by_disposition")


def plot_acuity_distribution(df):
    """Plot triage acuity distribution across dispositions"""
    g = sns.catplot(x="acuity_triage", col="disposition_edstays",
                   kind="count", col_wrap=3, data=df,
                   order=sorted(df["acuity_triage"].unique()),
                   palette="pastel", height=4, aspect=1.2, 
                   hue="acuity_triage", legend=False)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Triage Acuity Distribution Across Dispositions")
    save_fig("acuity_distribution")


def plot_demographic_distributions(df):
    """Plot distributions of demographic features"""
    demographic_features = [
        ('gender_edstays', 'Gender Distribution'),
        ('race_edstays', 'Race Distribution'),
        ('arrival_transport_edstays', 'Arrival Transport Method')
    ]
    
    for feature, title in demographic_features:
        plt.figure(figsize=(10, 5))
        ax = sns.countplot(x=feature, data=df, 
                          order=df[feature].value_counts().index,
                          palette="viridis", hue=feature, legend=False)
        plt.xticks(rotation=45)
        plt.title(title)
        
        # percentages annotation
        total = len(df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height()/total:.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        
        save_fig(f"{feature}_distribution")


def plot_correlation_matrix(df):
    """Plot correlation matrix of numeric features"""
    numeric_cols = df.select_dtypes(include=np.number)
    
    # drop IDs
    numeric_cols = numeric_cols.drop(columns=['subject_id', 'stay_id'], errors='ignore')
    
    corr = numeric_cols.corr(method="spearman")
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool)) # mask upper triangle
    sns.heatmap(corr, cmap="vlag", annot=True, fmt=".2f", 
               square=True, mask=mask, center=0,
               linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Spearman Correlation Between Numeric Variables")
    plt.tight_layout()
    save_fig("correlation_matrix")


def plot_time_series_comparison(df):
    """Compare triage vs vitalsign measurements"""
    measures = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    
    for measure in measures:
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=df, x=f"{measure}_triage", label="Triage", fill=True)
        sns.kdeplot(data=df, x=f"{measure}_vitalsign", label="Vitalsign", fill=True)
        plt.title(f"Distribution Comparison: {measure.capitalize()} (Triage vs Vitalsign)")
        plt.legend()
        save_fig(f"{measure}_triage_vs_vitalsign")


def generate_eda_report(df):
    """Generate complete EDA report"""
    plot_target_distribution(df)
    plot_vital_signs_by_disposition(df)
    plot_acuity_distribution(df)
    plot_demographic_distributions(df)
    plot_correlation_matrix(df)
    plot_time_series_comparison(df)


if __name__ == "__main__":
    joined_data_path = PROCESSED_DATA_PATH / "data_preprocessing/joined_df.csv"
    joined_df = pd.read_csv(joined_data_path)
    generate_eda_report(joined_df)
    print(f"EDA complete! Results saved to {EDA_PATH}")