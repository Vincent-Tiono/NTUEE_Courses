
#ifndef BLOCK_H_INCLUDED
#define BLOCK_H_INCLUDED
#include<vector>
#include<string>
#include"Player.h"
using std::string;
using std::vector;

// Declare global variables as extern
extern bool yes;
extern bool no;
extern bool blk_run;
extern bool blk_printed;
extern int skips;
extern string turn_message;
extern string money_message;
extern string dice_message;
extern string game1_message;
extern string game2_message;
extern string game3_message;
extern string mortage1_message;
extern string mortage2_message;
extern string over_message;

class block{
friend void chance1(player&);
public:
    block(int, string);
    ~block(){}
    int getpos () const {return pos;}
    string getname () const {return name;}
    static void InstantiateVector();
    virtual bool mortaged () {return 0;}
    virtual int getrent () {return 0;}
    virtual int getmoney() {return 0;}
    virtual void run_event (){return;}
    virtual void block_function(player&){return;}
protected:
    const int pos; // 0 ~ 39
    const string name;
};
extern vector<block*> board;

#endif // BLOCK_H_INCLUDED
