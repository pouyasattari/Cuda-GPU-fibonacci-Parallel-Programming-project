#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// Function to generate the Fibonacci sequence
void generateFibonacci(thrust::host_vector<int> &fibSeq, int size)
{
    fibSeq.resize(size);
    fibSeq[0] = 0;
    fibSeq[1] = 1;

    for (int i = 2; i < size; i++)
    {
        fibSeq[i] = fibSeq[i - 1] + fibSeq[i - 2];
    }
}

// Function to perform manual sorting using bubble sort
void bubbleSort(thrust::host_vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                // Swap elements
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to merge two sorted subarrays
void merge(thrust::host_vector<int> &arr, int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors for the left and right subarrays
    thrust::host_vector<int> left(n1);
    thrust::host_vector<int> right(n2);

    // Copy data to the temporary vectors
    for (int i = 0; i < n1; i++)
    {
        left[i] = arr[l + i];
    }
    for (int j = 0; j < n2; j++)
    {
        right[j] = arr[m + 1 + j];
    }

    // Merge the temporary arrays back into arr[l..r]
    int i = 0; // Initial index of first subarray
    int j = 0; // Initial index of second subarray
    int k = l; // Initial index of merged subarray

    while (i < n1 && j < n2)
    {
        if (left[i] <= right[j])
        {
            arr[k] = left[i];
            i++;
        }
        else
        {
            arr[k] = right[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of left[] if there are any
    while (i < n1)
    {
        arr[k] = left[i];
        i++;
        k++;
    }

    // Copy the remaining elements of right[] if there are any
    while (j < n2)
    {
        arr[k] = right[j];
        j++;
        k++;
    }
}

// Function to perform Merge Sort
void mergeSort(thrust::host_vector<int> &arr, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

        // Sort the left and right subarrays
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // Merge the sorted subarrays
        merge(arr, l, m, r);
    }
}

// Function to perform manual sorting using QuickSort
void quickSort(thrust::host_vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        // Partition the array
        int pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j <= high - 1; j++)
        {
            if (arr[j] <= pivot)
            {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);

        // Recursively sort the left and right subarrays
        quickSort(arr, low, i);
        quickSort(arr, i + 2, high);
    }
}

// Function to heapify a subtree rooted with node i which is an index in arr[]. n is the size of the heap
void heapify(thrust::host_vector<int> &arr, int n, int i)
{
    int largest = i;   // Initialize largest as root
    int l = 2 * i + 1; // left = 2*i + 1
    int r = 2 * i + 2; // right = 2*i + 2

    // If left child is larger than root
    if (l < n && arr[l] > arr[largest])
        largest = l;

    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest])
        largest = r;

    // If largest is not root
    if (largest != i)
    {
        std::swap(arr[i], arr[largest]);

        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}

// Main function to perform heap sort
void heapSort(thrust::host_vector<int> &arr)
{
    int n = arr.size();

    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // One by one extract an element from heap
    for (int i = n - 1; i >= 0; i--)
    {
        // Move current root to end
        std::swap(arr[0], arr[i]);

        // call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
}

// Functor for transforming values for descending sorting using Thrust
struct SortTransform
{
    __host__ __device__ int operator()(int x) const
    {
        return -x; // Negate the values to sort in descending order
    }
};

int main()
{
    int size = 1000;

    // Allocate memory for input and output arrays on the host
    thrust::host_vector<int> h_a(size);

    // Generate the Fibonacci sequence on the host
    generateFibonacci(h_a, size);

    // Print the generated sequence
    printf("Generated Sequence:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n\n");

    // Create CUDA events for measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Sort the array manually using bubble sort
    cudaEventRecord(start);
    bubbleSort(h_a);
    cudaEventRecord(stop);

    // Synchronize events and calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds_bubble = 0;
    cudaEventElapsedTime(&milliseconds_bubble, start, stop);

    // Copy the sorted array for later comparison
    thrust::host_vector<int> h_a_bubble = h_a;

    // Sort the array manually using merge sort
    cudaEventRecord(start);
    mergeSort(h_a, 0, size - 1);
    cudaEventRecord(stop);

    // Synchronize events and calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds_merge = 0;
    cudaEventElapsedTime(&milliseconds_merge, start, stop);

    // Copy the sorted array from device to host
    thrust::host_vector<int> h_a_merge = h_a;

    // Sort the array manually using QuickSort
    cudaEventRecord(start);
    quickSort(h_a, 0, size - 1);
    cudaEventRecord(stop);

    // Synchronize events and calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds_quick = 0;
    cudaEventElapsedTime(&milliseconds_quick, start, stop);

    // Sort the array manually using HeapSort
    cudaEventRecord(start);
    heapSort(h_a);
    cudaEventRecord(stop);

    // Synchronize events and calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds_heap = 0;
    cudaEventElapsedTime(&milliseconds_heap, start, stop);

    // Allocate memory for input array on the device
    thrust::device_vector<int> d_a = h_a;

    // Sort the array using Thrust
    cudaEventRecord(start);
    thrust::sort(d_a.begin(), d_a.end());
    cudaEventRecord(stop);

    // Synchronize events and calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds_thrust = 0;
    cudaEventElapsedTime(&milliseconds_thrust, start, stop);

    // Copy the sorted array from device to host
    h_a = d_a;

    // Allocate memory for input array on the device
    thrust::device_vector<int> d_a_transformed = h_a;

    // Transform the array using Thrust
    thrust::transform(d_a_transformed.begin(), d_a_transformed.end(), d_a_transformed.begin(), SortTransform());

    // Sort the transformed array using Thrust
    cudaEventRecord(start);
    thrust::sort(d_a_transformed.begin(), d_a_transformed.end());
    cudaEventRecord(stop);

    // Synchronize events and calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds_thrust_transform = 0;
    cudaEventElapsedTime(&milliseconds_thrust_transform, start, stop);

    // Copy the sorted array from device to host
    h_a = d_a_transformed;

    // Print the sorted arrays
    printf("Bubble Sort:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", h_a_bubble[i]);
    }
    printf("\n");

    printf("Merge Sort:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", h_a_merge[i]);
    }
    printf("\n");

    printf("Quick Sort:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    printf("Heap Sort:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    printf("Thrust Sort:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", d_a[i]);
    }
    printf("\n");

    printf("Thrust Sort + Transformation:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    // Print the elapsed times
    printf("\n");
    printf("\n");
    printf("\n");

    printf("~ Manual Sorting Algorithms :");
    printf("\n");
    printf("Bubble Sort Time: %.3f ms\n", milliseconds_bubble);
    printf("Quick Sort Time: %.3f ms\n", milliseconds_quick);
    printf("Merge Sort Time: %.3f ms\n", milliseconds_merge);
    printf("Heap Sort Time: %.3f ms\n", milliseconds_heap);
    printf("\n");
    printf("\n");
    printf("~ Sorting by Thrust library : ");
    printf("\n");
    printf("Thrust Sort Time: %.3f ms\n", milliseconds_thrust);
    printf("\n");
    printf("~Thrust Sorting Algorithm : ");
    printf("\n");
    printf("Thrust Sort + Transformation Time: %.3f ms\n", milliseconds_thrust_transform);
    printf("\n");
    printf("\n");

    return 0;
}
