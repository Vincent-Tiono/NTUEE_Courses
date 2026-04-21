
#include "text.h"
#include "block.h"
#include "card.h"
#include "Player.h"

// Define the global variables that were declared as extern

// SDL globals
SDL_Renderer* gRenderer = nullptr;
TTF_Font* gFont = nullptr;

// Game messages and state flags
bool yes = false;
bool no = false;
bool blk_run = false;
bool blk_printed = false;
int skips = 0;
std::string turn_message = "Turn:";
std::string money_message = "Balance:";
std::string dice_message = "Press 'D' to roll dice...";
std::string game1_message = "line1";
std::string game2_message = "line2";
std::string game3_message = "line3";
std::string mortage1_message = " ";
std::string mortage2_message = " ";
std::string over_message = "Press any key to continue...";
std::vector<block*> board;

// Global counters and textures
int player::bankrupt_count = 0;
LTexture chance_0, chance_1, chance_2, chance_3, chance_4, chance_5, chance_6, chance_7, chance_8, chance_9;
LTexture chance_10, chance_11, chance_12, chance_13, chance_14, chance_15, chance_16;
LTexture community_0, community_1, community_2, community_3, community_4, community_5, community_6, community_7;
LTexture community_8, community_9, community_10, community_11, community_12, community_13, community_14, community_15, community_16, community_17;

