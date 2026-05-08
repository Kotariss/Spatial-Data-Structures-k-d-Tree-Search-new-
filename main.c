#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kdtree.h"
#include "fcm.h"
#include "dbscan.h"

int main(int argc, char *argv[]) {
    if (argc < 3) return 1;
    
    int n_t, razm;
    Point *pts = read_points(argv[1], &n_t, &razm);
    if (!pts) { printf("Error: cannot read file %s\n", argv[1]); return 1; }
    printf("Loaded %d points of dimension %d\n", n_t, razm);

    KDNode *drv = NULL;
    for (int i = 0; i < n_t; i++) drv = kd_insert(drv, pts[i], 0);

    if (strcmp(argv[2], "-kd_insert") == 0) {
        if (argc < 4) { printf("Error: specify coordinates\n"); return 1; }
        double *crd = (double*)malloc(razm * sizeof(double));
        char *tok = strtok(argv[3], ",");
        for (int i = 0; i < razm && tok; i++) { crd[i] = atof(tok); tok = strtok(NULL, ","); }
        Point nov = create_point(crd, razm);
        drv = kd_insert(drv, nov, 0);
        printf("Point added\n");
        free(crd);
    }
    else if (strcmp(argv[2], "-kd_nearest") == 0) {
        if (argc < 4) { printf("Error: specify coordinates\n"); return 1; }
        double *t_crd = (double*)malloc(razm * sizeof(double));
        char *tok = strtok(argv[3], ",");
        for (int i = 0; i < razm && tok; i++) { t_crd[i] = atof(tok); tok = strtok(NULL, ","); }
        Point cel = create_point(t_crd, razm);
        double luch_d = 1e9; KDNode *luch = NULL;
        kd_find_nearest(drv, cel, 0, &luch, &luch_d);
        if (luch) {
            printf("Nearest point: ");
            for (int i = 0; i < razm; i++) printf("%.2f ", luch->point.coords[i]);
            printf("\nDistance: %.2f\n", luch_d);
        }
        free_point(&cel); free(t_crd);
    }
    else if (strcmp(argv[2], "-cmeans") == 0) {
        if (argc < 4) { printf("Error: specify number of clusters\n"); return 1; }
        int k_t = atoi(argv[3]);
        double fuz = (argc >= 5) ? atof(argv[4]) : 2.0;
        FCM *fc = fcm_create(pts, n_t, k_t, fuz);
        fcm_run(fc, pts, 100, 0.001);
        fcm_print(fc, pts);
        // Скрытый вывод для парсинга Python-скриптом
        printf("J_m: %.6f\n", fcm_compute_error(fc, pts));
        fcm_free(fc);
    }
    else if (strcmp(argv[2], "-dbscan") == 0) {
        if (argc < 4) { printf("Error: specify eps,minPts parameters\n"); return 1; }
        char *tok = strtok(argv[3], ","); double eps = atof(tok);
        tok = strtok(NULL, ","); int min_t = tok ? atoi(tok) : 3;
        DBSCAN *db = dbscan_run(pts, n_t, eps, min_t);
        dbscan_print(db, pts);
        dbscan_free(db);
    }
    else {
        fprintf(stderr, "Unknown command: %s\n", argv[2]);
    }

    kd_free(drv);
    for (int i = 0; i < n_t; i++) free_point(&pts[i]);
    free(pts);
    return 0;
}