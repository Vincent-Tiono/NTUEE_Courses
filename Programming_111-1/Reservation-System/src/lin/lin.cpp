#include <iostream>
#include <string>
#include <iomanip>
#include "lin.h"
using namespace std;

lin::lin(const string &mon1, const string &tue1, const string &wed1, const string &thu1, const string &fri1)
    : l_mon_time1(mon1),
      l_tue_time1(tue1),
      l_wed_time1(wed1),
      l_thu_time1(thu1),
      l_fri_time1(fri1)
{
}

void lin::l_print() const
{
    cout << setw(2) << "  This is the current schedule" << endl;
    cout << setw(2) << "---------------------------------" << endl;
    cout << setw(2) << " MON: ";
    cout << l_mon_time1 << endl;
    cout << setw(2) << " TUE: ";
    cout << l_tue_time1 << endl;
    cout << setw(2) << " WED: ";
    cout << l_wed_time1 << endl;
    cout << setw(2) << " Thu: ";
    cout << l_thu_time1 << endl;
    cout << setw(2) << " Fri: ";
    cout << l_fri_time1 << endl << endl;
}
