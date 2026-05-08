#include "fcm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// Создание структуры FCM
// pts — точки данных, n_t — их число, n_k — кластеров, fuz — нечёткость
FCM* fcm_create(Point *pts, int n_t, int n_k, double fuz) {
    FCM *fc = (FCM*)malloc(sizeof(FCM));
    fc->n_points = n_t;
    fc->n_clusters = n_k;
    fc->fuzziness = fuz; //нечеткость
    fc->centroids = (Point*)malloc(n_k * sizeof(Point));// Массив центров
    for (int i = 0; i < n_k; i++) { // i это класстеры
        // память под координаты центра
        fc->centroids[i].coords = (double*)malloc(pts[0].razmer * sizeof(double));// память под координаты центра
        fc->centroids[i].razmer = pts[0].razmer; // Запоминаем, сколько координат у этого центра, чтобы потом правильно обращаться к массиву и освобождать память.
    }
    // Выделяем память под матрицу принадлежности [точки × кластеры]
    fc->membership = (double**)malloc(n_t * sizeof(double*));
    for (int i = 0; i < n_t; i++) { // Для каждой точки
        fc->membership[i] = (double*)malloc(n_k * sizeof(double)); // Выделяем память под одну строку
        double row_sum = 0.0; // Сумма для нормировки
        for (int j = 0; j < n_k; j++) { // Для каждого кластера
            // Случайное число [0.5, 1.5] чтобы избежать нулей
            fc->membership[i][j] = 0.5 + (double)rand() / RAND_MAX;
            row_sum += fc->membership[i][j];// Складываем все сгенерированные значения, чтобы потом нормировать строку.
        }
        for (int j = 0; j < n_k; j++) {// Нормировка строки
            fc->membership[i][j] /= row_sum; // делим на сумму
            //До: [0.8, 1.2, 0.6], row_sum = 2.6
            //после: [0.8/2.6, 1.2/2.6, 0.6/2.6] = [0.31, 0.46, 0.23]
            //Проверка: 0.31 + 0.46 + 0.23 = 1.0
        }
    }
    return fc;
}
// Обновление матрицы принадлежности
void fcm_update_membership(FCM *fc, Point *pts) {
    double m = fc->fuzziness;// нечеткость
    double st = 2.0 / (m - 1.0);//показатель степени для формулы
    for (int i = 0; i < fc->n_points; i++) {
        double summ = 0.0; //накопитель знаменателя
        double *rass = (double*)malloc(fc->n_clusters * sizeof(double));// Массив расстояний до центров
        int zero_cluster = -1;// Флаг
        for (int j = 0; j < fc->n_clusters; j++) {
            // Считаем евклидово расстояние
            rass[j] = euclidean_distance(pts[i], fc->centroids[j]);
            if (rass[j] == 0.0) { // Если точка == центр
                zero_cluster = j;// запоминаем индекс
                break;
            }
        }
        if (zero_cluster != -1) {// если нашли точное совпадение
            for (int j = 0; j < fc->n_clusters; j++) {
                // 1.0 своему, 0.0 чужим
                fc->membership[i][j] = (j == zero_cluster) ? 1.0 : 0.0;
            }
            free(rass);
            continue;
        }
        for (int j = 0; j < fc->n_clusters; j++) { // Считаем знаменатель
            // Суммируем (1/d)^st для нормировки
            summ += pow(1.0 / rass[j], st);
        }
        for (int j = 0; j < fc->n_clusters; j++) { // Вычисляем доли
            // Формула принадлежности: числитель / знаменатель
            fc->membership[i][j] = pow(1.0 / rass[j], st) / summ;
        }
        free(rass);
    }
}
// Обновление центров кластеров (центроиды кластеров)
void fcm_update_centers(FCM *fc, Point *pts) {
    double m = fc->fuzziness;//нечеткость
    int razm = pts[0].razmer; //размерность пространства
    for (int j = 0; j < fc->n_clusters; j++) {
        //по формуле
        double *chisl = (double*)calloc(razm, sizeof(double)); // числитель
        double znam = 0.0; //знаменатель
        for (int i = 0; i < fc->n_points; i++) {
            // Вес точки: u^m
            double u = pow(fc->membership[i][j], m);
            for (int d = 0; d < razm; d++) {
                // Накопление: вес × координата точки
                chisl[d] += u * pts[i].coords[d];
            }
            znam += u;//накопление суммы весов
        }
        for (int d = 0; d < razm; d++) {
            // Формула: числитель / знаменатель (с защитой от деления на 0)
            fc->centroids[j].coords[d] = (znam > 1e-10) ? chisl[d] / znam : 0.0;
        }
        free(chisl);
    }
}
//метод локтя
//вычисление индекса
double fcm_xie_beni_index(FCM *fc, Point *pts) {
    if (fc->n_clusters <= 1) return INFINITY;
    double compactness = fcm_compute_error(fc, pts) / fc->n_points;
    double min_separation = INFINITY;
    for (int i = 0; i < fc->n_clusters; i++) {
        for (int j = i + 1; j < fc->n_clusters; j++) {
            double dist = euclidean_distance(fc->centroids[i], fc->centroids[j]);
            if (dist < min_separation) min_separation = dist;
        }
    }
    if (min_separation < 1e-10) return INFINITY; // защита от деления на 0
    return compactness / (min_separation * min_separation);
}
// Вычисление индекса разделения
double fcm_separation_index(FCM *fc) {
    if (fc->n_clusters <= 1) return 0.0;
    double total_separation = 0.0;
    int pairs = 0;
    for (int i = 0; i < fc->n_clusters; i++) {
        for (int j = i + 1; j < fc->n_clusters; j++) {
            total_separation += euclidean_distance(fc->centroids[i], fc->centroids[j]);
            pairs++;
        }
    }
    return (pairs > 0) ? total_separation / pairs : 0.0; // защита от деления на 0
}
// САМ ЦИКЛ АЛГОРИТМА!!!
void fcm_run(FCM *fc, Point *pts, int max_it, double epsil) {
    fcm_update_centers(fc, pts);// вычисляем центры из случайной матрицы U
    for (int iter = 0; iter < max_it; iter++) {
        Point *star = (Point*)malloc(fc->n_clusters * sizeof(Point)); //для сохранения старых центров(для проверки сходимости)
        for (int j = 0; j < fc->n_clusters; j++) {
            star[j] = create_point(fc->centroids[j].coords, pts[0].razmer);
        }
        fcm_update_membership(fc, pts);//обновляем матрицу U
        fcm_update_centers(fc, pts); //обновляем центры V
        double osh = 0.0; //cуммарный сдвиг центров
        for (int j = 0; j < fc->n_clusters; j++) {
            // Считаем расстояние сдвига центра
            osh += euclidean_distance(fc->centroids[j], star[j]);
            free_point(&star[j]);
        }
        free(star);
        // Критерий остановки: центры почти не двигаются
        if (osh < epsil) {
            // Вывод сообщения о сходимости
            printf("Converged at iteration %d (shift: %e < %e)\n", iter, osh, epsil);
            break;
        }
    }
}
// Вывод результатов
void fcm_print(FCM *fc, Point *pts) {
    printf("\nFCM RESULTS\n");
    // Параметры алгоритма
    printf("Clusters: %d, Points: %d, Fuzziness: %.2f\n",
           fc->n_clusters, fc->n_points, fc->fuzziness);
    printf("Cluster centers:\n");
    for (int j = 0; j < fc->n_clusters; j++) {
        printf("  K%d: [", j);// Номер кластера
        for (int d = 0; d < pts[0].razmer; d++) {
            // Печать координаты с запятой или без
            printf("%.3f%s", fc->centroids[j].coords[d], (d < pts[0].razmer-1) ? ", " : "");
        }
        printf("]\n");
    }
        printf("\nPoint membership:\n");
    int limit = fc->n_points; // Выводим для ВСЕХ точек
    for (int i = 0; i < limit; i++) {
        printf("  P%d: [", i);// Номер точки
        for (int j = 0; j < fc->n_clusters; j++) {
            // Печать степени принадлежности
            printf("%.3f%s", fc->membership[i][j], (j < fc->n_clusters-1) ? ", " : "");
        }
        printf("]\n");
    }
}
// Вычисление целевой функции J_m (ошибка кластеризации)
double fcm_compute_error(FCM *fc, Point *pts) {
    double J_m = 0.0;
    double m = fc->fuzziness;
    for (int i = 0; i < fc->n_points; i++) {
        for (int j = 0; j < fc->n_clusters; j++) {
            // Расстояние от точки до центра кластера
            double dist = euclidean_distance(pts[i], fc->centroids[j]);
            // Взвешенный вклад: (принадлежность)^m * (расстояние)^2
            J_m += pow(fc->membership[i][j], m) * dist * dist;
        }
    }
    return J_m;
}
// Очистка памяти
void fcm_free(FCM *fc) {
    for (int j = 0; j < fc->n_clusters; j++) {
        free_point(&fc->centroids[j]);
    }
    free(fc->centroids);
    for (int i = 0; i < fc->n_points; i++) {
        free(fc->membership[i]);
    }
    free(fc->membership);
    free(fc);
}