#include <petsc.h>

enum append_type {
    ROW_APPEND,
    COLUMN_APPEND,
};

PetscErrorCode update_A(Mat A, Vec q, Vec r)
{
    PetscInt M, N, q_size, r_size;
    const PetscScalar *q_data;
    const PetscScalar *r_data;

    CHKERRQ(MatGetSize(A, &M, &N));
    CHKERRQ(VecGetSize(q, &q_size));
    CHKERRQ(VecGetSize(r, &r_size));

    CHKERRQ(VecGetArrayRead(q, &q_data));
    CHKERRQ(VecGetArrayRead(r, &r_data));

    if ((M != q_size) || (N != r_size)) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Dimensions mismatch");
        return 1;
    }

    for (PetscInt i = 0; i < M; i++) {
        for (PetscInt j = 0; j < N; j++) {
            CHKERRQ(MatSetValue(A, i, j, -1 * q_data[i]*r_data[j], ADD_VALUES));
        }
    }

    CHKERRQ(VecRestoreArrayRead(q, &q_data));
    CHKERRQ(VecRestoreArrayRead(r, &r_data));

    return 0;
}

PetscErrorCode matrix_append_vector(Mat mat, Vec vec, PetscInt idx, PetscInt *idxm, PetscInt *idxn, enum append_type type)
{
    const PetscScalar *vals;
    PetscInt vec_size;
    PetscInt M, N;
    PetscInt ncols, nrows;

    CHKERRQ(VecGetSize(vec, &vec_size));
    CHKERRQ(MatGetSize(mat, &M, &N));

    if (type == ROW_APPEND) {
        if (N != vec_size) {
            SETERRQ(PETSC_COMM_WORLD, 1, "Vec size and matrix size does not match");
            return 1;
        }

        for (PetscInt i = 0; i < N; i++) idxn[i] = i;
        nrows = 1;
        ncols = vec_size;
        idxm[0] = idx;
    } else {
        if (M != vec_size) {
            SETERRQ(PETSC_COMM_WORLD, 1, "Vec size and matrix size does not match");
            return 1;
        }

        for (PetscInt i = 0; i < M; i++) idxm[i] = i;
        nrows = vec_size;
        ncols = 1;
        idxn[0] = idx;
    }

    CHKERRQ(VecGetArrayRead(vec, &vals));
    CHKERRQ(MatSetValues(mat, nrows, idxm, ncols, idxn, vals, INSERT_VALUES));
    CHKERRQ(VecRestoreArrayRead(vec, &vals));

    return 0;
}

PetscErrorCode naive_qr_decomposition(PetscInt M, PetscInt N)
{
    Mat A, Q, R;
    Vec q, r;
    PetscInt P = PetscMin(M, N);
    PetscInt A_m, A_n, cstart, cend, *idxm, *idxn;
    PetscReal *norms;
    PetscRandom rctx;

    CHKERRQ(PetscMalloc1(N, &norms));
    CHKERRQ(PetscMalloc2(M, &idxm, N, &idxn));

    // create A - mxn matrix
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &A));

    CHKERRQ(MatGetOwnershipRange(A, &cstart, &cend));

    CHKERRQ(MatGetLocalSize(A, &A_m, &A_n));

    // create Q - mxp
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, A_m, PETSC_DECIDE, M, P, NULL, &Q));
    // create R - pxn
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, A_n, P, N, NULL, &R));

    CHKERRQ(MatCreateVecs(A, &r, &q));

    //set random A with random values
    CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    CHKERRQ(MatSetRandom(A, rctx));
    CHKERRQ(PetscRandomDestroy(&rctx));

    CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    //TODO: remove debug
    PetscPrintf(PETSC_COMM_WORLD, "Matrix A, start\n");
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    for (PetscInt i = 0; i < P; i++) {
        PetscInt max_index = 0;
        PetscReal max_norm;

        CHKERRQ(MatGetColumnNorms(A, NORM_2, norms));
        max_norm = norms[0];
        for (PetscInt c = cstart; c < cend; c++) {
            max_norm = PetscMax(max_norm, norms[i]);
            if (max_norm == norms[i]) {
                max_index = i;
            }
        }

        CHKERRQ(MatGetColumnVector(A, q, max_index));
        CHKERRQ(VecNormalize(q, NULL));

        //TODO: remove debug
        PetscPrintf(PETSC_COMM_WORLD, "Vector q, itertion %d\n", i);
        VecView(q, PETSC_VIEWER_STDOUT_WORLD);

        CHKERRQ(MatMultHermitianTranspose(A, q, r));

        //TODO: remove debug
        PetscPrintf(PETSC_COMM_WORLD, "Vector r, itertion %d\n", i);
        VecView(r, PETSC_VIEWER_STDOUT_WORLD);

        CHKERRQ(matrix_append_vector(Q, q, i, idxm, idxn, COLUMN_APPEND));
        CHKERRQ(MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY));

        CHKERRQ(matrix_append_vector(R, r, i, idxm, idxn, ROW_APPEND));
        CHKERRQ(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));

        CHKERRQ(update_A(A, q, r));
        CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));

        CHKERRQ(MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY));
        CHKERRQ(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));
        CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

        //TODO: remove debug
        MatView(A, PETSC_VIEWER_STDOUT_WORLD);
        MatView(Q, PETSC_VIEWER_STDOUT_WORLD);
        MatView(R, PETSC_VIEWER_STDOUT_WORLD);

        //TODO: show norm in each iteration
    }

    CHKERRQ(PetscFree(norms));

    CHKERRQ(VecDestroy(&q));
    CHKERRQ(VecDestroy(&r));

    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&Q));
    CHKERRQ(MatDestroy(&R));

    return 0;
}

int main(int argc, char **argv)
{
    CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));

    CHKERRQ(naive_qr_decomposition(3, 3));

    PetscFinalize();
}
