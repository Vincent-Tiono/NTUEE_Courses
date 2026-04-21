#include <iostream>
#include <string>
#include <iomanip>
#include "hsieh.h"
using namespace std;

hsieh::hsieh(const string &mon1, const string &tue1, const string &wed1, const string &thu1, const string &fri1)
    : h_mon_time1(mon1),
      h_tue_time1(tue1),
      h_wed_time1(wed1),
      h_thu_time1(thu1),
      h_fri_time1(fri1)
{
}

void hsieh::h_print() const
{
    cout << setw(2) << "  This is the current schedule" << endl;
    cout << setw(2) << "---------------------------------" << endl;
    cout << setw(2) << " MON: ";
    cout << h_mon_time1 << endl;
    cout << setw(2) << " TUE: ";
    cout << h_tue_time1 << endl;
    cout << setw(2) << " WED: ";
    cout << h_wed_time1 << endl;
    cout << setw(2) << " Thu: ";
    cout << h_thu_time1 << endl;
    cout << setw(2) << " Fri: ";
    cout << h_fri_time1 << endl << endl;
}
