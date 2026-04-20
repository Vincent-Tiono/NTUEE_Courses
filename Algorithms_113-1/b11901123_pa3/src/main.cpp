#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

// AI generated begin
// Edge with source, destination and weight
struct Edge {
    int source;
    int destination;
    int weight;
};

// Node in disjoint set
struct Node {
    int parent;
};

// DFS to detect cycles
class CycleDetector {
private:
    static bool visitNode(vector<vector<int> >& adjacencyList, vector<char>& visitStatus, int currentNode) {
        visitStatus[currentNode] = 'v';  // Mark as visited
        
        for (int neighbor : adjacencyList[currentNode]) {
            if (visitStatus[neighbor] == 'v')  // Cycle detected
                return true;
            if (visitStatus[neighbor] == 'u' && visitNode(adjacencyList, visitStatus, neighbor))
                return true;
        }
        
        visitStatus[currentNode] = 'c';  // Mark as completed
        return false;
    }

public:
    static bool hasCycle(vector<vector<int> >& adjacencyList) {
        vector<char> visitStatus(adjacencyList.size(), 'u');  // u = unvisited, v = visiting, c = completed
        
        for (int node = 0; node < adjacencyList.size(); node++) {
            if (visitStatus[node] == 'u' && visitNode(adjacencyList, visitStatus, node))
                return true;
        }
        return false;
    }
};

// Disjoint-set for Find and Union
class DisjointSet {
private:
    vector<Node> nodes;

public:
    void initialize(int size) {
        nodes.clear();
        for (int i = 0; i < size; i++) {
            nodes.push_back({i});  // Each node is its own parent initially
        }
    }
    
    int findRoot(int node) {
        if (node != nodes[node].parent) {
            nodes[node].parent = findRoot(nodes[node].parent);  // Path compression
        }
        return nodes[node].parent;
    }
    
    void merge(int node1, int node2) {
        nodes[findRoot(node1)].parent = findRoot(node2);
    }
};


// Main class to process the graph
class GraphProcessor {
private:
    vector<Edge> edges;
    vector<Edge> removedEdges;
    vector<Edge> outputEdges;
    vector<vector<int> > adjacencyList;
    DisjointSet disjointSet;
    int totalWeight;
    int vertexCount;
    char graphType;

    static bool compareEdgeWeights(const Edge& first, const Edge& second) {
        return first.weight > second.weight;
    }

    void processUndirectedGraph() {
        for (const Edge& edge : edges) {
            if (disjointSet.findRoot(edge.source) == disjointSet.findRoot(edge.destination)) {
                totalWeight += edge.weight;
                removedEdges.push_back(edge);
            } else {
                adjacencyList[edge.source].push_back(edge.destination);
                disjointSet.merge(edge.source, edge.destination);
            }
        }
    }

    void processDirectedGraph() {
        for (const Edge& edge : removedEdges) {
            if (edge.weight < 0) {
                outputEdges.push_back(edge);
                continue;
            }
            
            adjacencyList[edge.source].push_back(edge.destination);
            
            if (CycleDetector::hasCycle(adjacencyList)) {
                adjacencyList[edge.source].pop_back();
                outputEdges.push_back(edge);
            } else {
                totalWeight -= edge.weight;
            }
        }
    }

public:
    void processInput(ifstream& input) {
        input >> graphType >> vertexCount;
        int edgeCount;
        input >> edgeCount;
        
        adjacencyList.resize(vertexCount);
        disjointSet.initialize(vertexCount);
        
        int source, dest, weight;
        while (input >> source >> dest >> weight) {
            edges.push_back({source, dest, weight});
        }
    }

    void process() {
        totalWeight = 0;
        sort(edges.begin(), edges.end(), compareEdgeWeights);
        processUndirectedGraph();
        
        if (graphType == 'd') {
            processDirectedGraph();
        }
    }

    void writeOutput(ofstream& output) {
        output << totalWeight << endl;
        const vector<Edge>& finalEdges = (graphType == 'd') ? outputEdges : removedEdges;
        
        for (const Edge& edge : finalEdges) {
            output << edge.source << " " << edge.destination << " " << edge.weight << endl;
        }
    }
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " input_file output_file" << endl;
        return 1;
    }

    ifstream inputFile(argv[1]);
    ofstream outputFile(argv[2]);
    
    if (!inputFile || !outputFile) {
        cerr << "Error opening files" << endl;
        return 1;
    }

    GraphProcessor processor;
    processor.processInput(inputFile);
    processor.process();
    processor.writeOutput(outputFile);

    inputFile.close();
    outputFile.close();
    return 0;
}
// AI generated end