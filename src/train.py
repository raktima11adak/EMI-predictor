from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import train_all

if __name__ == '__main__':
    result = train_all(sample_mode='--sample' in sys.argv)
    print(f"Classification winner: {result['best_classification_model']}")
    print(f"Regression winner: {result['best_regression_model']}")
    if result.get('mlflow_status'):
        print(result['mlflow_status'])
