// **************************************************************************
//  File       [sort_tool.cpp]
//  Author     [Yu-Hao Ho]
//  Synopsis   [The implementation of the SortTool Class]
//  Modify     [2020/9/15 Mu-Ting Wu]
//  Modify     [2023/9/12 Ming-Bang Fan]
//  Modify     [2024/9/2 Meng-Chen Wu]
// **************************************************************************

#include "sort_tool.h"
#include <iostream>
// Constructor
SortTool::SortTool() {}

// Insertsion sort method
void SortTool::InsertionSort(vector<int>& data) {
    // Function : Insertion sort
    // TODO : Please complete insertion sort code here
    // printing array 
    // for (int i = 0; i < data.size(); i++){
    //     cout << data[i] << " ";
    for (int i = 1; i < data.size(); i++) {
        int j = i;
        while (j > 0 && data[j] < data[j - 1]) {
            swap(data[j], data[j - 1]);
            j--;
        }
    }
}
// Quick sort method
void SortTool::QuickSort(vector<int>& data,int f){
    QuickSortSubVector(data, 0, data.size() - 1, f);
}
// Sort subvector (Quick sort)
void SortTool::QuickSortSubVector(vector<int>& data, int low, int high, const int flag) {
    // Function : Quick sort subvector
    // TODO : Please complete QuickSortSubVector code here
    // Hint : recursively call itself
    //        Partition function is needed
    // flag == 0 -> normal QS
    // flag == 1 -> randomized QS
    int middle;
    
    if (low < high) {
        if (flag == 0) {
            middle = Partition(data, low, high);
        } else {
            middle = RandomizedPartition(data, low, high);
        }
        QuickSortSubVector(data, low, middle - 1, flag);
        QuickSortSubVector(data, middle + 1, high, flag);
    }
}

int SortTool::RandomizedPartition(vector<int>& data, int low, int high){
    // Function : RQS's Partition the vector 
    // TODO : Please complete the function
    int randomIdx = low + rand() % (high - low + 1);
    swap(data[high], data[randomIdx]);
    return Partition(data, low, high);
}

int SortTool::Partition(vector<int>& data, int low, int high) {
    // Function : Partition the vector 
    // TODO : Please complete the function
    int pivot = data[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (data[j] <= pivot) {
            i++;
            swap(data[i], data[j]);
        }
    }
    swap(data[high], data[i + 1]);
    return i + 1;
}

// Merge sort method
void SortTool::MergeSort(vector<int>& data){
    MergeSortSubVector(data, 0, data.size() - 1);
}

// Sort subvector (Merge sort)
void SortTool::MergeSortSubVector(vector<int>& data, int low, int high) {
    // Function : Merge sort subvector
    // TODO : Please complete MergeSortSubVector code here
    // Hint : recursively call itself
    //        Merge function is needed

    if (low < high) {
        int middle = low + (high - low) / 2;
        MergeSortSubVector(data, low, middle);
        MergeSortSubVector(data, middle + 1, high);
        Merge(data, low, middle, middle + 1, high);
    }
}

// Merge
void SortTool::Merge(vector<int>& data, int low, int middle1, int middle2, int high) {
    // Function: Merge two sorted subvectors
    // TODO: Please complete the function
    int left = middle1 - low + 1;
    int right = high - middle2 + 1;

    vector<int> leftArray(left);
    vector<int> rightArray(right);

    for (int i = 0; i < left; i++) {
        leftArray[i] = data[low + i];
    }

    for (int j = 0; j < right; j++) {
        rightArray[j] = data[middle2 + j];
    }

    int leftIdx = 0;
    int rightIdx = 0;
    int mergedIdx = low;

    while (leftIdx < left && rightIdx < right) {
        if (leftArray[leftIdx] <= rightArray[rightIdx]) {
            data[mergedIdx] = leftArray[leftIdx];
            leftIdx++;
        } else {
            data[mergedIdx] = rightArray[rightIdx];
            rightIdx++;
        }
        mergedIdx++;
    }

    while (leftIdx < left) {
        data[mergedIdx] = leftArray[leftIdx];
        leftIdx++;
        mergedIdx++;
    }

    while (rightIdx < right) {
        data[mergedIdx] = rightArray[rightIdx];
        rightIdx++;
        mergedIdx++;
    }
}

// bottom-up style implementation of merge sort
void SortTool::BottomUpMergeSort(vector<int>& data) {
  /*TODO :
    Implement merge sort in bottom-up style, in other words,
    without recursive function calls.
    Hint: 
    1. Divide data to n groups of one data each group
    2. Iteratively merge each pair of 2 neighbor groups into one larger group
    3. Finally we obtain exactly one sorted group
  */
    int n = data.size();

    for (int length = 1; length < n; length *= 2) {
        for (int start = 0; start < n; start += 2 * length) {
            int middle = min(start + length - 1, n - 1);
            int end = min(start + 2 * length - 1, n - 1);
            
            if (middle < end) {
                Merge(data, start, middle, middle + 1, end);
            }
        }
    }
}

// Heap sort method, be aware that you are NOT required to implement heap sort in PA1
void SortTool::HeapSort(vector<int>& data) {
    // Build Max-Heap
    BuildMaxHeap(data);
    // 1. Swap data[0] which is max value and data[i] so that the max value will be in correct location
    // 2. Do max-heapify for data[0]
    for (int i = data.size() - 1; i >= 1; i--) {
        swap(data[0],data[i]);
        heapSize--;
        MaxHeapify(data,0);
    }
}

//Max heapify
void SortTool::MaxHeapify(vector<int>& data, int root) {
    // Function : Make tree with given root be a max-heap if both right and left sub-tree are max-heap

}

//Build max heap
void SortTool::BuildMaxHeap(vector<int>& data) {
    heapSize = data.size(); // initialize heap size

}