#pragma once

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gramschmidt(int ni, int nj, Arr2D &A, Arr2D &R, Arr2D &Q) {
    int i, j, k;

    DATA_TYPE nrm;

    for (k = 0; k < nj; k++) {
        // Consideriamo la colonna k-esima di A
        nrm = 0;

        //  Calcoliamo la norma di A^(k)
        for (i = 0; i < ni; i++)
            nrm += A[i][k] * A[i][k];

        //  che viene salvata in nel k-esimo elemento diagonale di R
        R[k][k] = sqrt(nrm);

        // la k-esima colonna di Q è la normalizzazione della k-esima colonna di A
        // R[k][k] è una very busy expression
        for (i = 0; i < ni; i++)
            Q[i][k] = A[i][k] / R[k][k];

        // Per ogni colonna successiva alla k-esima (definita nell'outer loop)
        for (j = k + 1; j < nj; j++) {
            R[k][j] = 0;

            // R alla riga k, colonna j è il prodotto della k-esima colonna di Q per la j-esima colonna di A
            for (i = 0; i < ni; i++)
                R[k][j] += Q[i][k] * A[i][j];

            // aggiorno la colonna i-esima di A con il prodotto element-wise tra colonna k-esima di Q e j-esima di R
            for (i = 0; i < ni; i++)
                A[i][j] = A[i][j] - Q[i][k] * R[k][j];
        }
    }
}

