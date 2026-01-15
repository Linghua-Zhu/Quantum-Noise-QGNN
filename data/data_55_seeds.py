"""
Complete experimental data for 55 QGNN models with different random initializations.

Each model was trained on QM9 HOMO-LUMO gap prediction and evaluated at 
noise levels ε ∈ {0, 0.005, 0.010, 0.015}.

Data fields:
- seed: Random seed for initialization
- baseline_r2: R² score at ε=0 (noiseless)
- improvement: Best performance change (%) relative to baseline
- best_noise: Noise level achieving best performance
- best_r2: Best R² score across all noise levels
"""

data_55_seeds = {
    'seed': [4392, 3303, 8985, 3064, 9499, 9849, 6307, 6534, 7999, 1379, 
             4491, 1571, 4971, 3144, 5180, 3947, 3799, 4050, 9312, 5407,
             420, 33, 9, 6821, 956, 89, 10, 28, 2022, 1210,
             777, 756, 93, 653, 181, 6749, 2003, 8295, 6503, 334,
             779, 2, 780, 46794, 6898, 25, 1, 1220, 6019, 2018,
             5587, 356, 5777, 7988, 8888],
             
    'baseline_r2': [0.6873, 0.6798, 0.7423, 0.7246, 0.7032, 0.7283, 0.6671, 0.7368, 0.7422, 0.7440,
                    0.7312, 0.6776, 0.7173, 0.7200, 0.7294, 0.7168, 0.6922, 0.7212, 0.6900, 0.7112,
                    0.6759, 0.6927, 0.7461, 0.7256, 0.7259, 0.7116, 0.6834, 0.6922, 0.7224, 0.7262,
                    0.7071, 0.7004, 0.6495, 0.7010, 0.6587, 0.7099, 0.7168, 0.7428, 0.7156, 0.7337,
                    0.7133, 0.7299, 0.6832, 0.7182, 0.6512, 0.7467, 0.7427, 0.7147, 0.6468, 0.7499,
                    0.7264, 0.6896, 0.7581, 0.7070, 0.6850],
                    
    'improvement': [9.0, 5.4, 1.5, -2.5, -3.3, 3.1, 11.4, 3.5, 0.8, -6.4,
                    -1.3, 8.3, -0.3, 4.3, 2.1, -2.1, 1.0, -9.1, 8.1, -7.0,
                    1.7, 1.5, 1.4, -0.7, -0.1, 0.2, 4.1, 8.7, 1.6, 1.0,
                    1.0, 1.8, 9.8, 2.4, 6.4, 4.1, 1.0, -2.6, -2.1, -0.3,
                    0.0, 4.1, 4.5, 0.4, 8.7, -0.7, -0.2, 5.9, 6.3, 0.3,
                    -1.3, 1.5, 1.4, 1.2, 1.8],
                    
    'best_noise': [0.005, 0.005, 0.005, 0.000, 0.000, 0.005, 0.005, 0.005, 0.005, 0.000,
                   0.000, 0.005, 0.000, 0.005, 0.010, 0.000, 0.010, 0.000, 0.005, 0.000,
                   0.010, 0.010, 0.005, 0.000, 0.000, 0.005, 0.010, 0.005, 0.005, 0.005,
                   0.005, 0.005, 0.010, 0.005, 0.005, 0.005, 0.005, 0.000, 0.000, 0.000,
                   0.010, 0.005, 0.005, 0.005, 0.015, 0.000, 0.000, 0.010, 0.005, 0.015,
                   0.000, 0.010, 0.005, 0.005, 0.010],
                   
    'best_r2': [0.7492, 0.7166, 0.7533, 0.7246, 0.7032, 0.7511, 0.7430, 0.7624, 0.7480, 0.7440,
                0.7312, 0.7337, 0.7173, 0.7509, 0.7447, 0.7168, 0.6993, 0.7212, 0.7456, 0.7112,
                0.6871, 0.7028, 0.7566, 0.7256, 0.7259, 0.7129, 0.7114, 0.7527, 0.7339, 0.7334,
                0.7143, 0.7132, 0.7130, 0.7176, 0.7009, 0.7388, 0.7240, 0.7428, 0.7156, 0.7337,
                0.7134, 0.7596, 0.7142, 0.7208, 0.7082, 0.7467, 0.7427, 0.7567, 0.6873, 0.7520,
                0.7264, 0.6996, 0.7690, 0.7154, 0.6973]
}

# Classification thresholds
BENEFICIAL_THRESHOLD = 2.0  # >2% improvement
DETRIMENTAL_THRESHOLD = -2.0  # <-2% degradation

def classify_model(improvement):
    """Classify model response to noise"""
    if improvement > BENEFICIAL_THRESHOLD:
        return 'Beneficial'
    elif improvement < DETRIMENTAL_THRESHOLD:
        return 'Detrimental'
    else:
        return 'Marginal'

def get_statistics():
    """Get summary statistics"""
    import pandas as pd
    df = pd.DataFrame(data_55_seeds)
    
    stats = {
        'total': len(df),
        'beneficial': len(df[df['improvement'] > BENEFICIAL_THRESHOLD]),
        'detrimental': len(df[df['improvement'] < DETRIMENTAL_THRESHOLD]),
        'marginal': len(df[abs(df['improvement']) <= BENEFICIAL_THRESHOLD]),
        'mean_improvement': df['improvement'].mean(),
        'std_improvement': df['improvement'].std(),
    }
    
    return stats

if __name__ == "__main__":
    stats = get_statistics()
    print("="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total models: {stats['total']}")
    print(f"Beneficial: {stats['beneficial']} ({stats['beneficial']/stats['total']*100:.1f}%)")
    print(f"Detrimental: {stats['detrimental']} ({stats['detrimental']/stats['total']*100:.1f}%)")
    print(f"Marginal: {stats['marginal']} ({stats['marginal']/stats['total']*100:.1f}%)")
    print(f"\nMean improvement: {stats['mean_improvement']:.2f}% ± {stats['std_improvement']:.2f}%")
