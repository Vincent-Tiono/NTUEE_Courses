
#ifndef CHANCEANDCHEST_H_INCLUDED
#define CHANCEANDCHEST_H_INCLUDED

#include "card.h"

// Add extern for any global objects or variables if they exist in this header
// Currently assuming no global variables based on typical card usage

class ChanceAndChest : public card {
public:
    void run_event(player&) override;
};

#endif // CHANCEANDCHEST_H_INCLUDED
