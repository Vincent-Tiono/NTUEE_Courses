#ifndef EVENT_H_INCLUDED
#define EVENT_H_INCLUDED
#include"block.h"

class event : public block
{
public:
    event(int pos, string name, int money, int tp, bool skip);
    ~event(){};
    int getmoney() {return money;}
    void block_function(player &P);
private:
    const int money;
    const int tp;
    const bool skip;
};
bool paid_flag = false;
event::event(int pos, string name, int money, int tp = -1, bool skip = false):block(pos, name),money(money), tp(tp), skip(skip){}
void event::block_function(player &P)
{
    if (pos == 28)// student acc. fee
    {
        if (P.paid_student)
        {
            game2_message =  "Talk to the hand!";
            blk_run = true;
            return;
        }
        game3_message = P.getname() + " realizes it is useless." ;
        if (!blk_run) paid_flag = true;
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
#endif // EVENT_H_INCLUDED
