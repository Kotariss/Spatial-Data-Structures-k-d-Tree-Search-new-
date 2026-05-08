import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import re
import os

def run_c_command(csv_file, command, *args, executable="./robot"):
    """Запуск C-программы и возврат stdout"""
    cmd = [executable, csv_file, command] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения: {e.stderr}")
        sys.exit(1)

def parse_fcm_output(output, n_points, n_clusters):
    """Парсинг вывода fcm_print: извлечение центров и матрицы принадлежности"""
    lines = output.split('\n')
    centroids = []
    memberships = []
    
    in_centers = False
    for line in lines:
        if "Cluster centers:" in line:
            in_centers = True
            continue
        if in_centers and line.strip().startswith("K"):
            match = re.search(r'\[([\d\.,\s-]+)\]', line)
            if match:
                coords = [float(x.strip()) for x in match.group(1).split(',')]
                centroids.append(coords)
        if "Point membership" in line:
            break
            
    in_membership = False
    for line in lines:
        if "Point membership" in line:
            in_membership = True
            continue
        if in_membership and line.strip().startswith("P"):
            match = re.search(r'\[([\d\.,\s-]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                memberships.append(values)
                
    return np.array(centroids), np.array(memberships)

def parse_dbscan_output(output, n_points):
    """Парсинг вывода dbscan_print: извлечение меток кластеров"""
    lines = output.split('\n')
    labels = np.zeros(n_points, dtype=int)
    for line in lines:
        match = re.search(r'Point\s+(\d+).*->\s+cluster\s+(-?\d+)', line)
        if match:
            idx = int(match.group(1))
            label = int(match.group(2))
            if idx < n_points:
                labels[idx] = label
    return labels

def load_csv_data(csv_file):
    """Чтение CSV. Строго проверяет, что данных ровно 2 измерения."""
    data = []
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [float(x) for x in line.split(',')]
                data.append(row)
    arr = np.array(data)
    if arr.shape[1] != 2:
        print(f"Ошибка: скрипт поддерживает ТОЛЬКО 2D-данные. В файле найдено {arr.shape[1]} измерений (x,y).")
        sys.exit(1)
    return arr

def plot_fcm(data, centroids, memberships, save_path=None):
    """Визуализация FCM: точки окрашены по максимальной принадлежности, прозрачность = уверенность"""
    n_points = len(data)
    n_clusters = len(centroids)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_clusters, 10)))
    hard_labels = np.argmax(memberships, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_points):
        k = hard_labels[i]
        alpha = memberships[i, k]  # прозрачность = степень принадлежности
        ax.scatter(data[i, 0], data[i, 1], c=[colors[k]], s=40, alpha=alpha, edgecolors='white', linewidth=0.3)
        
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=250, edgecolors='black',
               label='Centroids', linewidths=2.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Fuzzy C-Means Clustering (2D)')
    ax.legend(); ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
    plt.show()

def plot_dbscan(data, labels, save_path=None):
    """Визуализация DBSCAN: разные цвета для кластеров, шум — серый"""
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l > 0])
    if n_clusters == 0:
        print("Нет найденных кластеров (только шум)")
        
    colors = ['gray'] + list(plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters))))
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, s=40, alpha=0.8, edgecolors='white', linewidth=0.3)
    
    handles = []
    if 0 in unique_labels:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Noise'))
    for k in range(1, max(labels)+1):
        if k in unique_labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[k], markersize=10, label=f'Cluster {k}'))
            
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title(f'DBSCAN Clustering (Found {n_clusters} clusters)')
    ax.legend(handles=handles); ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ График сохранён: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Визуализация кластеризации FCM/DBSCAN (ТОЛЬКО 2D)')
    parser.add_argument('csv_file', help='CSV файл с данными (строго 2D: x,y)')
    parser.add_argument('-fcm', type=int, metavar='K', help='Запустить FCM с K кластерами')
    parser.add_argument('-dbscan', type=str, metavar='EPS,MINPTS', help='Запустить DBSCAN (например: 0.5,2)')
    parser.add_argument('--save', type=str, metavar='FILE', help='Сохранить график в файл (PNG)')
    parser.add_argument('--exe', default='./robot', help='Путь к исполняемому файлу')
    args = parser.parse_args()
    
    if not args.fcm and not args.dbscan:
        parser.print_help()
        sys.exit(1)
        
    data = load_csv_data(args.csv_file)
    n_points = data.shape[0]
    print(f"Загружено {n_points} точек, размерность: 2D")
    
    if args.fcm:
        print(f"Запуск FCM с {args.fcm} кластерами...")
        output = run_c_command(args.csv_file, "-cmeans", str(args.fcm), "2.0", executable=args.exe)
        centroids, memberships = parse_fcm_output(output, n_points, args.fcm)
        print(f"✓ Найдено {len(centroids)} центроидов")
        plot_fcm(data, centroids, memberships, save_path=args.save)
        
    elif args.dbscan:
        eps, minpts = map(float, args.dbscan.split(','))
        print(f"Запуск DBSCAN (eps={eps}, minPts={int(minpts)})...")
        output = run_c_command(args.csv_file, "-dbscan", f"{eps},{int(minpts)}", executable=args.exe)
        labels = parse_dbscan_output(output, n_points)
        unique, counts = np.unique(labels, return_counts=True)
        noise = counts[0] if 0 in unique else 0
        n_clusters = len([l for l in unique if l > 0])
        print(f"✓ Найдено кластеров: {n_clusters}, шум: {noise} точек")
        plot_dbscan(data, labels, save_path=args.save)

if __name__ == "__main__":
    main()
