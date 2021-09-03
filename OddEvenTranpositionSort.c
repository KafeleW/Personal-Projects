#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

//Swap elements
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

//Prints array elements
void printArray(int arr[], int n) {
  int i;
    for(i = 0; i < n; i++) {
      printf("%d ", arr[i]);
    }
    printf("\n");
}

//Serial bubbleSort
void bubbleSort(int arr[], int n) {
  int i, j;
   for(i = 0; i < n-1; i++) {
     for(j = 0; j < n-i-1; j++) {
       if(arr[j] > arr[j+1]) {
         swap(&arr[j], &arr[j+1]);
       }
     }
   }
}

//Serial oddevenSort
void oddevenSort(int arr[], int n) {
  int phase, i, temp;
  for(phase = 0; phase < n; phase++) {
    //Even phase
    if(phase % 2 == 0) {
      for(i = 1; i < n; i += 2) {
        if(arr[i-1] > arr[i]) {
          swap(&arr[i], &arr[i-1]);
        }
      }
    } else {
      //Odd phase
      for(i = 1; i < n - 1; i += 2) {
        if(arr[i] > arr[i+1]) {
          swap(&arr[i], &arr[i+1]);
        }
      }
    }
  }
}

int main(void) {
  int n = 10000, i, count, myid;
  double stime, ftime, rtime;
  int *arrP = malloc(n* sizeof (long));
  int *a = malloc(n* sizeof (int));
  int *arrT = malloc(n* sizeof (int));
  //int *arrP = malloc(n* sizeof (int));
  //int *arrO = malloc(n* sizeof (int));
  //int *arrB = malloc(n* sizeof (int));

  for(i = 0; i < n; i++) {
    arrP[i] = rand();
    a[i] = arrP[i];
  //  arrO[i] = arrP[i];
  //  arrB[i] = arrP[i];
  }

  int phase, j, temp;
  stime = omp_get_wtime();
  for(phase = 0; phase < n; phase++) {
    //Variable arrP is shared because threads are depenentend on a shared array
    //Variable j is private because swaps are independent in each thread
    #pragma omp parallel shared(arrP) private(j)
    {
    //Even phase
    //Parallel for loop inorder to parallel even swap phase
    if(phase % 2 == 0) {
      #pragma omp for
      for(j = 1; j < n; j += 2) {
        if(arrP[j-1] > arrP[j]) {
          swap(&arrP[j], &arrP[j-1]);
        }
      }
    } else {
      //Odd index phase
      //Parallel for loop inorder to parallel odd swap phase
      #pragma omp for
      for(j = 1; j < n - 1; j += 2) {
        if(arrP[j] > arrP[j+1]) {
          swap(&arrP[j], &arrP[j+1]);
        }
      }
    }
  }
}
  //Computing runtime of parallel OddevenSort
  ftime = omp_get_wtime();
  rtime = ftime - stime;
  printf("Parellel OddevenSort runtime: %.16f\n", rtime);


  #pragma omp parallel shared(arrT,a) private(count,i,j)
  {
    if (myid==0) {
      stime = omp_get_wtime();
    }
    #pragma omp for
    for (i = 0; i < n; i++) {
      count = 0;
      for (j = 0; j < n; j++) {
        if (a[j] < a[i]) {
          count++;
        } else if (a[j] == a[i] && j < i) {
          count++;
        }
      }
      #pragma omp critical
      arrT[count] = a[i];
    }
    #pragma omp barrier
    #pragma omp single
    memcpy(a, arrT, n*sizeof(int));
    free(arrT);

    if (myid==0) {
      ftime = omp_get_wtime();
      rtime = ftime - stime;
      for(i = 0; i < n; i++) {
      //  printf("%d ",a[i]);
      }
      printf("\ncountSort runtime is = %.16f\n",rtime);
    }
  }

  printArray(arrP, n);

  //Calling Serial OddevenSort Function & computing runtime
  stime = omp_get_wtime();
  oddevenSort(arrO, n);
  ftime = omp_get_wtime();
  rtime = ftime - stime;
  printf("Serial OddevenSort runtime: %.16f\n", rtime);
  printArray(arrO, n);

  //Calling Serial BubbleSort Function & computing runtime
  stime = omp_get_wtime();
  bubbleSort(arrB, n);
  ftime = omp_get_wtime();
  rtime = ftime - stime;
  printf("Serial BubbleSort runtime: %.16f\n", rtime);
  printArray(arrB, n);

  return 0;
}
