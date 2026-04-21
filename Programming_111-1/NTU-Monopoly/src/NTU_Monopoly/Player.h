#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED
#include "text.h"
using std::string;
LTexture p1;
LTexture p2;
LTexture p3;
LTexture p4;
class player
{
public:
    player(int m, string);
    ~player();
    int money;
    int pos;
    int skip;
    bool paid_student;
    string getname() const {return name;}
    bool bankrupt;
    bool dead;
    static int bankrupt_count;
    bool operator == (player) const;
    int posx();
    int posy();
    LTexture texture;
private:
    const string name;
};
int player::bankrupt_count = 0;
int player::posx()
{
    if (pos < 10)
        return (65 + 64*pos); // d=60
    else if (pos < 20)
        return 725;
    else if (pos < 30)
        return (65 + 64*(10 - (pos-20)));
    else
        return 15;
}
int player::posy()
{
    if (pos <= 10)
        return 725;
    else if (pos < 20)
        return (625 - 64*(pos-11));
    else if (pos <= 30)
        return 15;
    else
        return (625 - 64*(39-pos));
    return 0;
}
player::player(int m, string n):money(m),name(n)
{
    pos = 0;
    skip = 0;
    paid_student = false;
    bankrupt = false;
    dead = false;
}
player::~player()
{

}
bool player::operator==(player P) const
{
    if (name == P.name) return true;
    return false;
}
#endif // PLAYER_H_INCLUDED
