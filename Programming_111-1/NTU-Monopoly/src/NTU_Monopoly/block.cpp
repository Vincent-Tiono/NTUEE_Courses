#include "block.h"

// Define the board vector
vector<block*> board;

// Constructor definition without default arguments
block::block(int p, string n) : pos(p), name(n) {
    if (board.size() == 0) {
        InstantiateVector();
    }
    board[p] = this;
}

// InstantiateVector function definition
void block::InstantiateVector() {
    board.resize(40);
}
