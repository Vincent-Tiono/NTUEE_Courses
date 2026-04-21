#ifndef WEI_H_INCLUDED
#define WEI_H_INCLUDED

#include <iostream>
#include <string>
using namespace std;

class wei
{
    friend class reservation;
    friend void w_display();
    public:
        wei(const string&, const string&, const string&, const string&, const string&);
        void w_print() const;

    private:
        const string w_mon_time1;
        const string w_tue_time1;
        const string w_wed_time1;
        const string w_thu_time1;
        const string w_fri_time1;
};


#endif // WEI_H_INCLUDED
