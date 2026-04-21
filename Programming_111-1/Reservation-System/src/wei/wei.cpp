#include <iostream>
#include <iomanip>
#include "wei.h"  // Include the header file for the wei class
using namespace std;

wei::wei(const string &mon1, const string &tue1, const string &wed1, const string &thu1, const string &fri1)
    : w_mon_time1(mon1),
      w_tue_time1(tue1),
      w_wed_time1(wed1),
      w_thu_time1(thu1),
      w_fri_time1(fri1)
{
}

void wei::w_print() const {
    cout << setw(2) << "  This is the current schedule" << endl;
    cout << setw(2) << "---------------------------------" << endl;
    cout << setw(2) << " MON: " << w_mon_time1 << endl;
    cout << setw(2) << " TUE: " << w_tue_time1 << endl;
    cout << setw(2) << " WED: " << w_wed_time1 << endl;
    cout << setw(2) << " Thu: " << w_thu_time1 << endl;
    cout << setw(2) << " Fri: " << w_fri_time1 << endl << endl;
}
