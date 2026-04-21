#ifndef PROPERTY_H_INCLUDED
#define PROPERTY_H_INCLUDED
#include "block.h"
#include <iostream>
using namespace std;
class property : public block
{
friend void turn(player&);
friend void chance1(player&);
friend void chance12(player&);
friend void mortage_check(player&);
public:
    property(int pos, string name, int price, int flag_price,int super_big_price);
    ~property();
    bool mortaged(){return mortage;}
    int getrent();
    void block_function(player&);
    int getmortage() const
    {
        return (mortage_value + flag_price*(lv-1)/2);
    }
    LTexture* f1 = &flag;
    LTexture* f2 = &flag;
    LTexture* f3 = &flag;
    LTexture* f4 = &flag;
    LTexture* sb = &superbig;
    void flag_render();
private:
    bool mortage = false;
    const int price;
    const int flag_price;
    const int super_big_price;
    const int mortage_value;
    player* owner;
    int lv = 1; // 1~6
};
property::property(int pos, string name, int price, int flag_price,int super_big_price)
: block(pos,name), price(price), flag_price(flag_price), super_big_price(super_big_price),
mortage_value(price/2){};
property::~property(){}

void property::flag_render()
{
    if (pos < 10)
    {
        if (lv == 6)
        {
            sb->render(40+64*pos + 34,650);
            return;
        }
        if (lv < 2) return;
        f1->render(40+64*pos,646);
        if (lv < 3) return;
        f2->render(40+64*pos + 15,646);
        if (lv < 4) return;
        f3->render(40+64*pos + 30,646);
        if (lv < 5) return;
        f4->render(40+64*pos + 45,646);

    }
    else if (pos < 20)
    {
        if (lv == 6)
        {
            sb->render(660, 647 - 64*(pos-11) - 34, NULL, 270);
            return;
        }
        if (lv < 2) return;
        f1->render(660, 647 - 64*(pos-11) , NULL, 270);
        if (lv < 3) return;
        f2->render(660, 647 - 64*(pos-11) - 15 , NULL, 270);
        if (lv < 4) return;
        f3->render(660, 647 - 64*(pos-11) - 30, NULL, 270);
        if (lv < 5) return;
        f4->render(660, 647 - 64*(pos-11) - 45, NULL, 270);
    }
    else if (pos < 30)
    {
        if (lv == 6)
        {
            sb->render((82 + 64*(10 - (pos-20))) - 34,90, NULL, 180);
            return;
        }
        if (lv < 2) return;
        f1->render((82 + 64*(10 - (pos-20))),92, NULL, 180);
        if (lv < 3) return;
        f2->render((82 + 64*(10 - (pos-20))) - 15,92, NULL, 180);
        if (lv < 4) return;
        f3->render((82 + 64*(10 - (pos-20))) - 30,92, NULL, 180);
        if (lv < 5) return;
        f4->render((82 + 64*(10 - (pos-20))) - 45,92, NULL, 180);
    }
    else
    {
        if (lv == 6)
        {
            sb->render(102, 606 - 64*(39-pos) + 34 , NULL, 90);
            return;
        }
        if (lv < 2) return;
        f1->render(104, 606 - 64*(39-pos) , NULL, 90);
        if (lv < 3) return;
        f2->render(104, 606 - 64*(39-pos) + 15 , NULL, 90);
        if (lv < 4) return;
        f3->render(104, 606 - 64*(39-pos) + 30 , NULL, 90);
        if (lv < 5) return;
        f4->render(104, 606 - 64*(39-pos) + 45 , NULL, 90);
    }
}
int property::getrent()
{
    switch(lv)
    {
    case 1:
        return price / 10;
    case 2:
        return price / 2;
    case 3:
        return 3 * price / 2;
    case 4:
        return 3 * price;
    case 5:
        return 4 * price;
    case 6://super big
        return 11 * price / 2;
    default:
        cout << "Bug!\n";
        break;
    }
}
bool flag1 = false;
bool flag2 = false;
void property::block_function(player &P)
{
    if (!owner || flag1)
    {
        if (!flag1 && price > P.money)
        {
            game2_message = "Not enough money to purchase!";
            blk_printed = true;
            blk_run = true;
            return;
        }
        game2_message =  "Price:" + to_string(price) ;
        game3_message = "Buy " + getname() + "?(Y/N)";
        blk_printed = true;
        if (yes || flag1)
        {
            game2_message = "Purchased.";
            if (!blk_run)
            {
                P.money -= price;
                owner = &P;
                yes = false;
                blk_run = true;
                flag1 = true;
            }
            game3_message = " ";

        }
        else if (no || flag2)
        {
            game2_message = "Didn't purchase.";
            game3_message = " ";
            no = false;
            blk_run = true;
            flag2 = true;
        }

    }
    else if (*owner == P)
    {
        if (lv == 6)
        {
            game2_message =  "Already maxed out! ";
            if (!blk_printed)
            {
                blk_printed = true;
                blk_run = true;
            }
            return;
        }
        else if(lv == 5)
        {
            if (P.money < super_big_price && !flag1)
            {
                game2_message =  "Not enough money to upgrade!";
                if (!blk_printed)
                {
                    blk_printed = true;
                    blk_run = true;
                }
                return;
            }
            game2_message = "Super big price:" + to_string(super_big_price);
            game3_message = "Upgrade to super big?(Y/N)";
            if (!blk_printed)
            {
                blk_printed = true;
            }
            if (yes || flag1)
            {
                if (!blk_run)
                {
                    lv++;
                    P.money -= super_big_price;
                    yes = false;
                    blk_run = true;
                    flag1 = true;
                }
                game2_message = "Upgraded to SUPER BIG.";
                game3_message = " ";
            }
            else if (no || flag2)
            {
                game2_message = "Didn't upgrade.";
                no = false;
                game3_message = " ";
                blk_run = true;
                flag2 = true;
            }
        }
        else
        {
            if (P.money < flag_price && !flag1)
            {
                game2_message =  "Not enough money to upgrade!";
                if (!blk_printed)
                {
                    blk_printed = true;
                    blk_run = true;
                }
                return;
            }
            game2_message = "Flag price:" + to_string(super_big_price);
            game3_message = "Purchase a flag?(Y/N)";
            if (!blk_printed)
            {
                blk_printed = true;
            }
            if (yes || flag1)
            {
                if (!blk_run)
                {
                    lv++;
                    P.money -= flag_price;
                    yes = false;
                    blk_run = true;
                    flag1 = true;
                }
                game2_message =  "Upgraded to " + to_string(lv-1) + " flag(s).";
                game3_message = " ";
            }
            else if (no || flag2)
            {
                game2_message = "Didn't upgrade.";
                game3_message = " ";
                no = false;
                blk_run = true;
                flag2 = true;
            }
        }

    }
    else
    {
        game2_message = "You paid " + to_string(getrent()) +"$ to " + owner->getname() +".";
        if (!blk_printed)
        {
            P.money -= getrent();
            owner->money += getrent();
            blk_printed = true;
            blk_run = true;
        }
    }
    return;
}
#endif // PROPERTY_H_INCLUDED
