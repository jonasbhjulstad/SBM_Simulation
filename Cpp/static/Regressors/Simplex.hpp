
/*
  What: Simplex in C
  AUTHOR: GPL(C) moshahmed/at/gmail.

  What: Solves LP Problem with Simplex:
    { maximize cx : Ax <= b, x >= 0 }.
  Input: { m, n, Mat[m x n] }, where:
    b = mat[1..m,0] .. column 0 is b >= 0, so x=0 is a basic feasible solution.
    c = mat[0,1..n] .. row 0 is z to maximize, note c is negated in input.
    A = mat[1..m,1..n] .. constraints.
    x = [x1..xm] are the named variables in the problem.
    Slack variables are in columns [m+1..m+n]

  USAGE:
    1. Problem can be specified before main function in source code:
      c:\> vim mosplex.c
      Tableau tab  = { m, n, {   // tableau size, row x columns.
          {  0 , -c1, -c2,  },  // Max: z = c1 x1 + c2 x2,
          { b1 , a11, a12,  },  //  b1 >= a11 x1 + a12 x2
          { b2 , a21, a22,  },  //  b2 >= a21 x1 + a22 x2
        }
      };
      c:\> cl /W4 mosplex.c  ... compile this file.
      c:\> mosplex.exe problem.txt > solution.txt

    2. OR Read the problem data from a file:
      $ cat problem.txt
            m n
            0  -c1 -c2
            b1 a11 a12
            b2 a21 a11
      $ gcc -Wall -g mosplex.c  -o mosplex
      $ mosplex problem.txt > solution.txt
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <FROLS_Eigen.hpp>
#define M 20
#define N 20
namespace FROLS::Simplex {
static const double epsilon = 1.0e-8;
int equal(double a, double b) { return fabs(a - b) < epsilon; }

struct Tableau {
  int m, n; // m=rows, n=columns, mat[m x n]
  double mat[M][N];
  Tableau(const Mat& A, const Vec& b, const Vec& c) {
    m = A.rows() + 1;
    n = A.cols() + 1;
    mat[0][0] = 0;
    for (int j = 1; j < n; j++) {
      mat[0][j] = -c(j - 1);
    }
    for (int i = 1; i < m; i++) {
      mat[i][0] = b(i - 1);
      for (int j = 1; j < n; j++) {
        mat[i][j] = A(i - 1, j - 1);
      }
    }
  }
};

void pivot_on(Tableau *tab, int row, int col) {
  int i, j;
  double pivot;

  pivot = tab->mat[row][col];
  assert(pivot > 0);
  for (j = 0; j < tab->n; j++)
    tab->mat[row][j] /= pivot;
  assert(equal(tab->mat[row][col], 1.));

  for (i = 0; i < tab->m; i++) { // foreach remaining row i do
    double multiplier = tab->mat[i][col];
    if (i == row)
      continue;
    for (j = 0; j < tab->n; j++) { // r[i] = r[i] - z * r[row];
      tab->mat[i][j] -= multiplier * tab->mat[row][j];
    }
  }
}

// Find pivot_col = most negative column in mat[0][1..n]
int find_pivot_column(Tableau *tab) {
  int j, pivot_col = 1;
  double lowest = tab->mat[0][pivot_col];
  for (j = 1; j < tab->n; j++) {
    if (tab->mat[0][j] < lowest) {
      lowest = tab->mat[0][j];
      pivot_col = j;
    }
  }
  if (lowest >= 0) {
    return -1; // All positive columns in row[0], this is optimal.
  }
  return pivot_col;
}

// Find the pivot_row, with smallest positive ratio = col[0] / col[pivot]
int find_pivot_row(Tableau *tab, int pivot_col) {
  int i, pivot_row = 0;
  double min_ratio = -1;
  for (i = 1; i < tab->m; i++) {
    double ratio = tab->mat[i][0] / tab->mat[i][pivot_col];
    printf("%3.2lf, ", ratio);
    if ((ratio > 0 && ratio < min_ratio) || min_ratio < 0) {
      min_ratio = ratio;
      pivot_row = i;
    }
  }
  if (min_ratio == -1)
    return -1; // Unbounded.
  return pivot_row;
}

void add_slack_variables(Tableau *tab) {
  int i, j;
  for (i = 1; i < tab->m; i++) {
    for (j = 1; j < tab->m; j++)
      tab->mat[i][j + tab->n - 1] = (i == j);
  }
  tab->n += tab->m - 1;
}

void check_b_positive(Tableau *tab) {
  int i;
  for (i = 1; i < tab->m; i++)
    assert(tab->mat[i][0] >= 0);
}

// Given a column of identity matrix, find the row containing 1.
// return -1, if the column as not from an identity matrix.
int find_basis_variable(Tableau *tab, int col) {
  int i, xi = -1;
  for (i = 1; i < tab->m; i++) {
    if (equal(tab->mat[i][col], 1)) {
      if (xi == -1)
        xi = i; // found first '1', save this row number.
      else
        return -1; // found second '1', not an identity matrix.

    } else if (!equal(tab->mat[i][col], 0)) {
      return -1; // not an identity matrix column.
    }
  }
  return xi;
}

void print_optimal_vector(Tableau *tab, char *message) {
  int j, xi;
  for (j = 1; j < tab->n; j++) { // for each column.
    xi = find_basis_variable(tab, j);
  }
}

void simplex(Tableau *tab) {
  int loop = 0;
  add_slack_variables(tab);
  check_b_positive(tab);
  while (++loop) {
    int pivot_col, pivot_row;

    pivot_col = find_pivot_column(tab);

    pivot_row = find_pivot_row(tab, pivot_col);
    if (pivot_row < 0) {
      printf("unbounded (no pivot_row).\n");
      break;
    }

    pivot_on(tab, pivot_row, pivot_col);

    if (loop > 20) {
      printf("Too many iterations > %d.\n", loop);
      break;
    }
  }
}

} // namespace FROLS
