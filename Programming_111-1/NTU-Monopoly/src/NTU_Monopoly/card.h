
#ifndef CARD_H_INCLUDED
#define CARD_H_INCLUDED

#include <string>
#include "Player.h"
using std::string;

// Declare global variables as extern
extern LTexture chance_0, chance_1, chance_2, chance_3, chance_4, chance_5, chance_6, chance_7, chance_8, chance_9;
extern LTexture chance_10, chance_11, chance_12, chance_13, chance_14, chance_15, chance_16;
extern LTexture community_0, community_1, community_2, community_3, community_4, community_5, community_6, community_7;
extern LTexture community_8, community_9, community_10, community_11, community_12, community_13, community_14, community_15, community_16, community_17;

class card {
public:
    card() : used(false) {}
    virtual ~card() {}
    virtual void run_event(player&) = 0;
protected:
    bool used;
};

#endif // CARD_H_INCLUDED
