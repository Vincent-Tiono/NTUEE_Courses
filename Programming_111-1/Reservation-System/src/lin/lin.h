#ifndef LIN_H_INCLUDED
#define LIN_H_INCLUDED

#include <iostream>
#include <string>
using namespace std;

class lin
{
    public:
        lin(const string&, const string&, const string&, const string&, const string&);
        void l_print() const;

    private:
        const string l_mon_time1;
        const string l_tue_time1;
        const string l_wed_time1;
        const string l_thu_time1;
        const string l_fri_time1;
};

#endif // LIN_H_INCLUDED
