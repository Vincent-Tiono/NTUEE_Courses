#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <utility>
using namespace std;

class PlanarGraphSolver {
private:
    vector<int> edgeEndpoints;  // Stores the endpoint of each vertex
    vector<vector<int>> memoTable;  // Dynamic programming memoization table
    
    void setupMemoization(int size) {
        memoTable.resize(size);
        for (int i = 0; i < size; i++) {
            memoTable[i].assign(size - i, -1);
        }
    }
    
    int getMemoValue(int left, int right) {
        return (right <= left) ? 0 : memoTable[left][right - left];
    }
    
    int findMaxNonCrossingEdges(int left, int right) {
        // Base case: invalid range
        if (right <= left) return 0;
        
        // Return memoized result if available
        if (memoTable[left][right - left] != -1) 
            return memoTable[left][right - left];
        
        int result;
        if (edgeEndpoints[left] == right) {
            // Direct edge between endpoints
            result = 1 + findMaxNonCrossingEdges(left + 1, right - 1);
        }
        else if (edgeEndpoints[right] > left && edgeEndpoints[right] < right) {
            // Try including or excluding the edge at 'right'
            int withEdge = 1 + findMaxNonCrossingEdges(left, edgeEndpoints[right] - 1) +
                          findMaxNonCrossingEdges(edgeEndpoints[right] + 1, right - 1);
            int withoutEdge = findMaxNonCrossingEdges(left, right - 1);
            result = max(withEdge, withoutEdge);
        }
        else {
            // Skip current position
            result = findMaxNonCrossingEdges(left, right - 1);
        }
        
        memoTable[left][right - left] = result;
        return result;
    }
    
    void reconstructSolution(ofstream& output, int left, int right) {
        if (right <= left) return;
        
        if (edgeEndpoints[left] == right) {
            output << left << " " << right << endl;
            reconstructSolution(output, left + 1, right - 1);
        }
        else if (edgeEndpoints[right] > left && edgeEndpoints[right] < right) {
            int withEdge = 1 + getMemoValue(left, edgeEndpoints[right] - 1) +
                          getMemoValue(edgeEndpoints[right] + 1, right - 1);
                          
            if (getMemoValue(left, right) == withEdge) {
                reconstructSolution(output, left, edgeEndpoints[right] - 1);
                output << edgeEndpoints[right] << " " << right << endl;
                reconstructSolution(output, edgeEndpoints[right] + 1, right - 1);
            }
            else {
                reconstructSolution(output, left, right - 1);
            }
        }
        else {
            reconstructSolution(output, left, right - 1);
        }
    }

public:
    void solve(const string& inputPath, const string& outputPath) {
        ifstream input(inputPath);
        if (!input) {
            throw runtime_error("Failed to open input file: " + inputPath);
        }
        
        ofstream output(outputPath);
        if (!output) {
            throw runtime_error("Failed to open output file: " + outputPath);
        }
        
        int vertexCount;
        input >> vertexCount;
        
        edgeEndpoints.assign(vertexCount, -1);
        
        // Read edges
        int v1, v2;
        while (input >> v1 >> v2) {
            edgeEndpoints[v1] = v2;
            edgeEndpoints[v2] = v1;
        }
        
        setupMemoization(vertexCount);
        
        // Calculate and output result
        output << findMaxNonCrossingEdges(0, vertexCount - 1) << endl;
        reconstructSolution(output, 0, vertexCount - 1);
        
        input.close();
        output.close();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return 1;
    }
    
    try {
        PlanarGraphSolver solver;
        solver.solve(argv[1], argv[2]);
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}