from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pipeline import DATA_PATH, REPORT_DIR, load_data, assess_quality

REPORT_DIR.mkdir(exist_ok=True)
df = load_data(DATA_PATH)
quality = assess_quality(df)
(REPORT_DIR / 'data_quality.json').write_text(__import__('json').dumps(quality, indent=2, default=str), encoding='utf-8')

sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
df['emi_eligibility'].value_counts().plot(kind='bar', ax=axes[0], color=['#dc6b5b','#3677a8','#e8b44a'])
axes[0].set_title('Eligibility labels in sample'); axes[0].set_ylabel('Profiles'); axes[0].tick_params(axis='x', rotation=20)
df.groupby('emi_scenario')['max_monthly_emi'].median().sort_values().plot(kind='barh', ax=axes[1], color='#3677a8')
axes[1].set_title('Median safe EMI by scenario'); axes[1].set_xlabel('INR per month')
fig.tight_layout(); fig.savefig(REPORT_DIR / 'sample_eda.png', dpi=180, bbox_inches='tight')
print(__import__('json').dumps(quality, indent=2, default=str))
