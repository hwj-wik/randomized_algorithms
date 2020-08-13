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

PetscErrorCode matrix_set_random(Mat A)
{
    PetscRandom rctx;

    //set random A with random values
    CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    CHKERRQ(MatSetRandom(A, rctx));
    CHKERRQ(PetscRandomDestroy(&rctx));

    CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    return 0;
}

PetscErrorCode get_pivot_column_idx(Mat A, PetscReal *norms, PetscInt *idx)
{
    PetscReal min_norm;
    PetscInt cstart, cend;

    CHKERRQ(MatGetOwnershipRange(A, &cstart, &cend));

    CHKERRQ(MatGetColumnNorms(A, NORM_2, norms));
    min_norm = norms[0];
    min_index = cstart;
    for (PetscInt c = cstart; c < cend; c++) {
        min_norm = PetscMin(min_norm, norms[c]);
        if (min_norm == norms[c]) {
            min_index = cstart;
        }
    }

    CHKERRQ(MPIU_Allreduce())

    return 0;
}

PetscErrorCode pivot_matrix(Mat m, PetscInt src, PetscInt dest)
{
}

PetscErrorCode pivot_vector(Vec v, PetscInt src, PetscInt dest)
{
    PetscScalar data[2];
    PetscScalar tmp;
    PetscInt ix[2] = {src, dest};

    CHKERRQ(VecGetValues(v, 2, ix, data));
    ix[0] = dest;
    ix[1] = src;

    CHKERRQ(VecSetValues(v, 2, ix, data, INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
}

PetscErrorCode improved_qr_decomposition(PetscInt M, PetscInt N)
{
    Mat Q, R;
    Vec q, r, j;
    PetscInt P = PetscMin(M, N);
    PetscInt m, n, *idxm, *idxn;
    PetscReal *norms;

    CHKERRQ(PetscMalloc1(N, &norms));
    CHKERRQ(PetscMalloc2(M, &idxm, N, &idxn));

    // create Q - mxn matrix
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &Q));

    CHKERRQ(MatGetLocalSize(Q, &m, &n));
    // create R - pxn
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, n, P, N, NULL, &R));

    CHKERRQ(MatCreateVecs(Q, &j, NULL));

    CHKERRQ(matrix_set_random(A));

    for (int i = 0; i < P; i++) {
        PetscInt pivot_column;
        CHKERRQ(get_pivot_column_idx(Q, &idx, norms));
        CHKERRQ(pivot_vector(j, i, j));
        CHKERRQ(pivot_matrix(Q, i, j));
        CHKERRQ(pivot_matrix(R, i, j));
    }
}

PetscErrorCode naive_qr_decomposition(PetscMat A, PetscMat *Q, PetscMat *R)
{
    Vec q, r;
    PetscInt M, N, P;
    PetscInt A_m, A_n, *idxm, *idxn;
    PetscReal *norms;

    CHKERRQ(MatGetLocalSize(A, &m, &n));
    CHKERRQ(MatGetSize(A, &M, &N));

    // create Q - mxp
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, A_m, PETSC_DECIDE, M, P, NULL, Q));
    // create R - pxn
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, A_n, P, N, NULL, R));

    CHKERRQ(PetscMalloc1(N, &norms));
    CHKERRQ(PetscMalloc2(M, &idxm, N, &idxn));

    CHKERRQ(MatCreateVecs(A, &r, &q));

    for (PetscInt i = 0; i < P; i++) {
        PetscInt pivot_index = 0;

        CHKERRQ(get_pivot_column_idx(A, &pivot_index, norms));

        CHKERRQ(MatGetColumnVector(A, q, pivot_index));
        CHKERRQ(VecNormalize(q, NULL));

        CHKERRQ(MatMultHermitianTranspose(A, q, r));

        CHKERRQ(matrix_append_vector(Q, q, i, idxm, idxn, COLUMN_APPEND));
        CHKERRQ(MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY));

        CHKERRQ(matrix_append_vector(R, r, i, idxm, idxn, ROW_APPEND));
        CHKERRQ(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));

        CHKERRQ(update_A(A, q, r));
        CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));

        CHKERRQ(MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY));
        CHKERRQ(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));
        CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

#ifdef DEBUG
        {
            PetscReal approx_norm, ctrl_norm;
            PetscMat A_approx;
            CHKERRQ(MatMatMult(Q, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_approx));
            CHKERRQ(MatNorm(A_approx, NORM_1, &approx_norm));
            CHKERRQ(MatNorm(A, NORM_1, &ctrl_norm));
            PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] ||A'|| - ||A|| = %lf\n", PetscAbsReal(approx_norm - ctrl_norm));
        }
#endif
    }

    CHKERRQ(PetscFree(norms));

    CHKERRQ(VecDestroy(&q));
    CHKERRQ(VecDestroy(&r));

    return 0;
}

int main(int argc, char **argv)
{
    PetscMat A, Q, R;
    PetscInt M = 3, N = 3;
    CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));

    // create A - mxn matrix
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &A));
    CHKERRQ(matrix_set_random(A));

    CHKERRQ(naive_qr_decomposition(A, &Q, &R));

    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&Q));
    CHKERRQ(MatDestroy(&R));


    PetscFinalize();
}
