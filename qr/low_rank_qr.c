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

    PetscErrorCode ierr = MatGetSize(A, &M, &N);
    CHKERRQ(ierr);
    ierr = VecGetSize(q, &q_size);
    CHKERRQ(ierr);
    ierr = VecGetSize(r, &r_size);
    CHKERRQ(ierr);

    ierr = VecGetArrayRead(q, &q_data);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(r, &r_data);
    CHKERRQ(ierr);

    if ((M != q_size) || (N != r_size)) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Dimensions mismatch");
    }

    for (PetscInt i = 0; i < M; i++) {
        for (PetscInt j = 0; j < N; j++) {
            ierr = MatSetValue(A, i, j, -1 * q_data[i]*r_data[j], ADD_VALUES);
            CHKERRQ(ierr);
        }
    }

    ierr = VecRestoreArrayRead(q, &q_data);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(r, &r_data);
    CHKERRQ(ierr);

    return 0;
}

PetscErrorCode matrix_append_vector(Mat mat, Vec vec, PetscInt idx, PetscInt *idxm, PetscInt *idxn, enum append_type type)
{
    const PetscScalar *vals;
    PetscInt vec_size;
    PetscInt M, N;
    PetscInt ncols, nrows;

    PetscErrorCode ierr = VecGetSize(vec, &vec_size);
    CHKERRQ(ierr);
    ierr = MatGetSize(mat, &M, &N);
    CHKERRQ(ierr);

    if (type == ROW_APPEND) {
        if (N != vec_size) {
            SETERRQ(PETSC_COMM_WORLD, 1, "Vec size and matrix size does not match");
        }

        for (PetscInt i = 0; i < N; i++) idxn[i] = i;
        nrows = 1;
        ncols = vec_size;
        idxm[0] = idx;
    } else {
        if (M != vec_size) {
            SETERRQ(PETSC_COMM_WORLD, 1, "Vec size and matrix size does not match");
        }

        for (PetscInt i = 0; i < M; i++) idxm[i] = i;
        nrows = vec_size;
        ncols = 1;
        idxn[0] = idx;
    }

    ierr = VecGetArrayRead(vec, &vals);
    CHKERRQ(ierr);
    ierr = MatSetValues(mat, nrows, idxm, ncols, idxn, vals, INSERT_VALUES);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec, &vals);
    CHKERRQ(ierr);

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
    PetscErrorCode ierr;

    ierr = PetscMalloc1(N, &norms);
    CHKERRQ(ierr);
    ierr = PetscMalloc2(M, &idxm, N, &idxn);
    CHKERRQ(ierr);

    // create A - mxn matrix
    ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &A);
    CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A, &cstart, &cend);
    CHKERRQ(ierr);

    ierr = MatGetLocalSize(A, &A_m, &A_n);
    CHKERRQ(ierr);

    // create Q - mxp
    ierr = MatCreateDense(PETSC_COMM_WORLD, A_m, PETSC_DECIDE, M, P, NULL, &Q);
    CHKERRQ(ierr);
    // create R - pxn
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, A_n, P, N, NULL, &R);
    CHKERRQ(ierr);

    ierr = MatCreateVecs(A, &r, &q);
    CHKERRQ(ierr);

    //set random A with random values
    ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
    CHKERRQ(ierr);
    ierr = MatSetRandom(A, rctx);
    CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);
    CHKERRQ(ierr);

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "Matrix A, start\n");
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    for (PetscInt i = 0; i < P; i++) {
        PetscInt max_index = 0;
        PetscReal max_norm;

        ierr = MatGetColumnNorms(A, NORM_2, norms);
        CHKERRQ(ierr);
        max_norm = norms[0];
        for (PetscInt c = cstart; c < cend; c++) {
            max_norm = PetscMax(max_norm, norms[i]);
            if (max_norm == norms[i]) {
                max_index = i;
            }
        }

        ierr = MatGetColumnVector(A, q, max_index);
        CHKERRQ(ierr);
        ierr = VecNormalize(q, NULL);

        PetscPrintf(PETSC_COMM_WORLD, "Vector q, itertion %d\n", i);
        VecView(q, PETSC_VIEWER_STDOUT_WORLD);

        CHKERRQ(ierr);
        ierr = MatMultHermitianTranspose(A, q, r);
        CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Vector r, itertion %d\n", i);
        VecView(r, PETSC_VIEWER_STDOUT_WORLD);

        matrix_append_vector(Q, q, i, idxm, idxn, COLUMN_APPEND);
        ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        matrix_append_vector(R, r, i, idxm, idxn, ROW_APPEND);
        ierr = MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);

        update_A(A, q, r);
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);

        ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        ierr = MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);

        MatView(A, PETSC_VIEWER_STDOUT_WORLD);
        MatView(Q, PETSC_VIEWER_STDOUT_WORLD);
        MatView(R, PETSC_VIEWER_STDOUT_WORLD);
    }

    ierr = PetscFree(norms);
    CHKERRQ(ierr);

    ierr = VecDestroy(&q);
    CHKERRQ(ierr);
    ierr = VecDestroy(&r);
    CHKERRQ(ierr);

    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = MatDestroy(&Q);
    CHKERRQ(ierr);
    ierr = MatDestroy(&R);
    CHKERRQ(ierr);

    return 0;
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr) {
        return 1;
    }

    naive_qr_decomposition(3, 3);

    PetscFinalize();
}
