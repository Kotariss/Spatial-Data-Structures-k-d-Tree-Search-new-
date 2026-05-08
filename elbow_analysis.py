import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

def run_c_elbow(csv_file, k_max, fuzziness=2.0, executable="./robot"):
    """Запуск C программы в цикле и сбор J_m для каждого k"""
    data = []
    for k in range(1, k_max + 1):
        try:
            result = subprocess.run(
                [executable, csv_file, "-cmeans", str(k), str(fuzziness)],
                capture_output=True, text=True, check=True
            )
            # Ищем скрытую строку J_m: <value> в выводе C
            for line in result.stdout.split('\n'):
                if line.startswith('J_m:'):
                    jm = float(line.split(':')[1].strip())
                    data.append({'k': k, 'J_m': jm})
                    break
        except subprocess.CalledProcessError as e:
            print(f"Error for k={k}: {e.stderr.strip()}")
    return data

def find_optimal_k(data):
    """Поиск оптимального числа кластеров методом локтя"""
    ks = [d['k'] for d in data]
    jm_values = [d['J_m'] for d in data]
    if len(jm_values) < 3:
        return ks[-1] if ks else 1
    diffs = np.diff(jm_values)
    accels = np.diff(diffs)
    return ks[np.argmax(accels) + 2]

def plot_elbow_method(data, save_path=None):
    """Построение графика метода локтя"""
    ks = [d['k'] for d in data]
    jm_values = [d['J_m'] for d in data]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(ks, jm_values, 'bo-', linewidth=2.5, markersize=10,
            markerfacecolor='blue', markeredgecolor='darkblue',
            markeredgewidth=1.5, label='J_m (Objective Function)')
    
    for k, jm in zip(ks, jm_values):
        ax.annotate(f'{jm:.2f}', (k, jm), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    ax.set_xlabel('Number of Clusters (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Objective Function J_m', fontsize=13, fontweight='bold')
    ax.set_title('Elbow Method for Optimal k\n(Fuzzy C-Means Clustering)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper right', fontsize=10)

    elbow_k = find_optimal_k(data)
    elbow_idx = ks.index(elbow_k)

    ax.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Optimal k = {elbow_k}')
    ax.plot(elbow_k, jm_values[elbow_idx], 'ro', markersize=15,
            markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
    ax.text(elbow_k + 0.3, jm_values[elbow_idx],
            f'Elbow Point\nk = {elbow_k}',
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    plt.show()
    return elbow_k

def main():
    if len(sys.argv) < 2:
        print("Usage: python elbow_analysis.py <csv_file> [k_max] [fuzziness]")
        print("Example: python elbow_analysis.py sample.csv 10 2.0")
        sys.exit(1)

    csv_file = sys.argv[1]
    k_max = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    fuzziness = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    executable = "./robot"
    if not os.path.exists(executable):
        executable = "./robot.exe"
    if not os.path.exists(executable):
        print(f"Error: Executable {executable} not found!")
        print("Please compile the C program first:")
        print("  gcc main.c kdtree.c fcm.c dbscan.c -o robot -lm")
        sys.exit(1)

    print("ELBOW METHOD ANALYSIS")
    print(f"CSV file: {csv_file}")
    print(f"Max clusters: {k_max}")
    print(f"Fuzziness: {fuzziness}")

    data = run_c_elbow(csv_file, k_max, fuzziness, executable)
    if not data:
        print("Error: No data received!")
        sys.exit(1)

    print(f"\n{'k':<5} {'J_m':<15}")
    for d in data:
        print(f"{d['k']:<5} {d['J_m']:<15.6f}")

    optimal_k = find_optimal_k(data)
    print(f"OPTIMAL NUMBER OF CLUSTERS: k = {optimal_k}")

    plot_elbow_method(data, save_path="elbow_method.png")
    print(f"\nAnalysis complete! Optimal k = {optimal_k}")

if __name__ == "__main__":
    main()
