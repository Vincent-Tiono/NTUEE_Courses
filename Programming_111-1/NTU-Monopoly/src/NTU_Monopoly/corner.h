#ifndef CORNER_H_INCLUDED
#define CORNER_H_INCLUDED
#include "block.h"
#include <iostream>
class corner : public block
{
public:
    corner(int pos, string name, int money, int tp, bool skip);
    ~corner(){};
    int getmoney() const {return money;}
    void block_function(player &P);
private:
    const int money;//if activated, modify the player's money by money
    const int tp;
    const bool skip;
};
corner::corner(int pos, string name, int money, int tp = -1, bool skip = false):block(pos, name),money(money), tp(tp), skip(skip){}
void corner::block_function(player &P)
{
    //if (blk_run) return;
    if (pos == 0)
    {
        blk_run = true;
        return;
    }
    if (!blk_run) P.money += money;
    game2_message= P.getname() + " lost " + to_string(-money) + "$." ;

    if (tp != -1)
    {
        if (!blk_run)P.pos = tp;
        game3_message = "Tp:" + board[tp]->getname();
    }
    if (skip)
    {
        if (!blk_run)skips = 1;
        game3_message = P.getname() + " will stop for one turn.";
    }
    blk_run = true;
}
#endif // CORNER_H_INCLUDED
