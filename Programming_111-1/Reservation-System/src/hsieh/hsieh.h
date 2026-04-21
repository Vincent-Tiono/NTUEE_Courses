#ifndef HSIEH_H_INCLUDED
#define HSIEH_H_INCLUDED

#include <iostream>
#include <string>
using namespace std;
class hsieh
{
    public:
        hsieh(const string&, const string&, const string&, const string&, const string&);
        void h_print() const;

    private:
        const string h_mon_time1;
        const string h_tue_time1;
        const string h_wed_time1;
        const string h_thu_time1;
        const string h_fri_time1;
};

#endif // HSIEH_H_INCLUDED
