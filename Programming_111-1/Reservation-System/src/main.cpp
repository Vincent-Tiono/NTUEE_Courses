#include <iostream>
#include <string>
#include <iomanip>

#ifdef _WIN32
    #include <conio.h>  // Include conio.h only on Windows
    #define CLEAR_SCREEN "CLS"
    #define PAUSE() _getch()
#else
    #include <cstdio>
    #define CLEAR_SCREEN "clear"
    #define PAUSE() std::cin.get()
#endif

#include "reservation.h"
#include "wei.h"
#include "hsieh.h"
#include "lin.h"
using namespace std;

void instructions()
{
    cout << "Insert available hours" << endl;
    cout << "eg. 6PM-7PM" << endl;
    cout << "|NOTE: insert NA for \"Not Available\"|" << endl << endl;

}

void instructions2()
{
    cout << endl << "Thank you for reserving, press any key to exit.";
    PAUSE();
    system(CLEAR_SCREEN);
    cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
}

int main()
{
    string ID;
    string Teacher;
    string w_mon1, w_tue1, w_wed1, w_thu1, w_fri1;
    string h_mon1, h_tue1, h_wed1, h_thu1, h_fri1;
    string l_mon1, l_tue1, l_wed1, l_thu1, l_fri1;

    cout << "****Welcome to Computer Programming Study Group Reservation Page****";
    PAUSE();
    cout << endl << endl;
    int TorS; //choose between TA or student

    do
    {
    cout << "Please select: " << endl;
    cout << "1. TA" << endl;
    cout << "2. Student" << endl << endl;
    cout << "Press -1 to cancel" << endl << endl;

    cin >> TorS;

    system(CLEAR_SCREEN);

    if(TorS==-1)
    {
        cout << "See you next time." << endl;
        return 0;
    }


    cout << "****TA Interface****";
    cout << endl << endl;

    /*TA Interface*/
    if(TorS==1)
    {

        cout << "Which professor?" << endl;
        cout << "1. Wei" << endl;
        cout << "2. Hsieh" << endl;
        cout << "3. Lin" << endl << endl;

        int prof;
        cin >> prof;
        cout << endl;

        switch(prof)
        {

            case 1: /*Wei*/
                {
                cout << "Enter TA password" << endl;
                string pw;
                cin >> pw; //password is hywei
                int reedit;

                if(pw=="hywei")
                {
                    do
                    {
                        system(CLEAR_SCREEN);
                        cout << "****TA Interface****" << endl << endl;
                        wei schedule_w(w_mon1, w_tue1, w_wed1, w_thu1, w_fri1);
                        schedule_w.w_print();

                        cout << "Enter:" << endl;
                        cout << "1. Edit" << endl;
                        cout << "2. Quit" << endl << endl;
                        int temp;
                        cin >> temp;

                        system(CLEAR_SCREEN);
                        cout << "****TA Interface****" << endl;

                        if(temp==2)
                            continue;

                        else if(temp==1)
                        {
                            cout << endl;
                            instructions();

                            cout << "Mon: ";
                            cin >> w_mon1;

                            cout << "Tue: ";
                            cin >> w_tue1;

                            cout << "Wed: ";
                            cin >> w_wed1;

                            cout << "Thu: ";
                            cin >> w_thu1;

                            cout << "Fri: ";
                            cin >> w_fri1;
                            cout << endl;


                            system(CLEAR_SCREEN);
                            cout << "****TA Interface****" << endl << endl;
                            wei schedule_w(w_mon1, w_tue1, w_wed1, w_thu1, w_fri1);
                            schedule_w.w_print();

                            cout << "Enter:" << endl;
                            cout << "1. Save" << endl;
                            cout << "2. Edit" << endl << endl;
                            cin >> reedit;
                            cout << endl;
                        }
                    }while(reedit==2);

                if(reedit==1)
                    {
                        system(CLEAR_SCREEN);
                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        break;
                    }
                }

                else
                    {
                        cout << endl << "Incorrect password, please try again." << endl;
                        PAUSE();
                        system(CLEAR_SCREEN);
                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        break;
                    }
                }

                system(CLEAR_SCREEN);
                cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                break;


            case 2: /*Hsieh*/
            {
                cout << "Enter TA password" << endl;
                string pw;
                cin >> pw; //password is hyhsieh
                int reedit;

                if(pw=="hyhsieh")
                {
                    do
                    {
                        system(CLEAR_SCREEN);
                        cout << "****TA Interface****" << endl << endl;
                        hsieh schedule_h(h_mon1, h_tue1, h_wed1, h_thu1, h_fri1);
                        schedule_h.h_print();

                        cout << "Enter:" << endl;
                        cout << "1. Edit" << endl;
                        cout << "2. Quit" << endl << endl;
                        int temp; //edit or quit
                        cin >> temp;

                        system(CLEAR_SCREEN);
                        cout << "****TA Interface****" << endl;

                        if(temp==2)
                            continue;

                        else if(temp==1)
                        {
                            cout << endl;
                            instructions();

                            cout << "Mon: ";
                            cin >> h_mon1;

                            cout << "Tue: ";
                            cin >> h_tue1;

                            cout << "Wed: ";
                            cin >> h_wed1;

                            cout << "Thu: ";
                            cin >> h_thu1;

                            cout << "Fri: ";
                            cin >> h_fri1;
                            cout << endl;


                            system(CLEAR_SCREEN);
                            cout << "****TA Interface****" << endl << endl;
                            hsieh schedule_h(h_mon1, h_tue1, h_wed1, h_thu1, h_fri1);
                            schedule_h.h_print();

                            cout << "Enter:" << endl;
                            cout << "1. Save" << endl;
                            cout << "2. Edit" << endl << endl;
                            cin >> reedit;
                            cout << endl;
                        }
                    }while(reedit==2);

                if(reedit==1)
                    {
                        system(CLEAR_SCREEN);
                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        break;
                    }
                }

                else
                    {
                        cout << endl << "Incorrect password, please try again." << endl;
                        PAUSE();
                        system(CLEAR_SCREEN);
                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        break;
                    }
                }

                system(CLEAR_SCREEN);
                cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                break;


        case 3: /*Lin*/
            {
                cout << "Enter TA password" << endl;
                string pw;
                cin >> pw; //password is tnlin
                int reedit;

                if(pw=="tnlin")
                {
                    do
                    {
                        system(CLEAR_SCREEN);
                        cout << "****TA Interface****" << endl << endl;
                        lin schedule_l(l_mon1, l_tue1, l_wed1, l_thu1, l_fri1);
                        schedule_l.l_print();

                        cout << "Enter:" << endl;
                        cout << "1. Edit" << endl;
                        cout << "2. Quit" << endl << endl;
                        int temp;
                        cin >> temp;

                        system(CLEAR_SCREEN);
                        cout << "****TA Interface****" << endl;

                        if(temp==2)
                            continue;

                        else if(temp==1)
                        {
                            cout << endl;
                            instructions();

                            cout << "Mon: ";
                            cin >> l_mon1;

                            cout << "Tue: ";
                            cin >> l_tue1;

                            cout << "Wed: ";
                            cin >> l_wed1;

                            cout << "Thu: ";
                            cin >> l_thu1;

                            cout << "Fri: ";
                            cin >> l_fri1;
                            cout << endl;


                            system(CLEAR_SCREEN);
                            cout << "****TA Interface****" << endl << endl;
                            lin schedule_l(l_mon1, l_tue1, l_wed1, l_thu1, l_fri1);
                            schedule_l.l_print();

                            cout << "Enter:" << endl;
                            cout << "1. Save" << endl;
                            cout << "2. Edit" << endl << endl;
                            cin >> reedit;
                            cout << endl;
                        }
                    }while(reedit==2);

                if(reedit==1)
                    {
                        system(CLEAR_SCREEN);
                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        break;
                    }
                }

                else
                    {
                        cout << endl << "Incorrect password, please try again." << endl;
                        PAUSE();
                        system(CLEAR_SCREEN);
                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        break;
                    }
                }

                system(CLEAR_SCREEN);
                cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                break;


            default:
                {
                    cout << endl << "Invalid Professor, press any key to exit.";
                    PAUSE();
                    system(CLEAR_SCREEN);
                    cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;

                }

        }


    }

    /*Student Interface*/
    else if(TorS==2)
    {
        reservation No1;
        system(CLEAR_SCREEN);
        cout << "****Student Interface****" << endl << endl;
        cout << "Please enter your ID: ";
        cin >> ID;
        No1.setID(ID);

        cout << "Please enter your professor: ";
        cin >> Teacher;
        No1.setTeacher(Teacher);

        if(No1.getID()=="0" || No1.getTeacher()=="0")
            {
                cout << endl << "Invalid ID or Teacher, please try again.";
                PAUSE();
                system(CLEAR_SCREEN);
                cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
            }

        else
            {
                system(CLEAR_SCREEN);
                cout << "****Student Interface****" << endl << endl;

                if(No1.getTeacher()=="Wei" || No1.getTeacher()=="wei")
                {

                        cout << setw(2) << "  This is the current schedule" << endl;
                        cout << setw(2) << "---------------------------------" << endl;
                        cout << setw(2) << " MON: ";
                        cout << w_mon1 << endl;
                        cout << setw(2) << " TUE: ";
                        cout << w_tue1 << endl;
                        cout << setw(2) << " WED: ";
                        cout << w_wed1 << endl;
                        cout << setw(2) << " Thu: ";
                        cout << w_thu1 << endl;
                        cout << setw(2) << " Fri: ";
                        cout << w_fri1 << endl << endl;

                        cout << "Would you like to join this week's study group?" << endl;
                        cout << "Enter:" << endl;
                        cout << "1. Yes" << endl;
                        cout << "2. No" << endl << endl;
                        int YorN;
                        cin >> YorN;

                        if(YorN==2)
                        {
                            cout << endl << "Press any key to exit.";
                            PAUSE();
                            system(CLEAR_SCREEN);
                            cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        }

                        else if(YorN==1)
                        {
                            system(CLEAR_SCREEN);
                            cout << "****Student Interface****" << endl << endl;
                            cout << setw(2) << "  This is the current schedule" << endl;
                            cout << setw(2) << "---------------------------------" << endl;
                            cout << setw(2) << " MON: ";
                            cout << w_mon1 << endl;
                            cout << setw(2) << " TUE: ";
                            cout << w_tue1 << endl;
                            cout << setw(2) << " WED: ";
                            cout << w_wed1 << endl;
                            cout << setw(2) << " Thu: ";
                            cout << w_thu1 << endl;
                            cout << setw(2) << " Fri: ";
                            cout << w_fri1 << endl << endl;

                            cout << "Please pick a time:" << endl;
                            cout << "|NOTE: NA means \"Not Available\"|" << endl << endl;
                            cout << "Enter:" << endl;
                            cout << "1. Monday" << endl;
                            cout << "2. Tuesday" << endl;
                            cout << "3. Wednesday" << endl;
                            cout << "4. Thursday" << endl;
                            cout << "5. Friday" << endl << endl;
                            int pick_a_time;
                            cin >> pick_a_time;

                            switch(pick_a_time)
                            {
                                case 1:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Monday " << w_mon1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 2:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Tuesday " << w_tue1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 3:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Wednesday " << w_wed1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 4:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Thursday " << w_thu1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 5:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Friday " << w_fri1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }



                                default:
                                    {
                                        cout << endl << "Invalid time, press any key to exit.";
                                        PAUSE();
                                        system(CLEAR_SCREEN);
                                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                                    }

                            }

                        }

                }

                else if(No1.getTeacher()=="Hsieh" || No1.getTeacher()=="hsieh")
                {

                        cout << setw(2) << "  This is the current schedule" << endl;
                        cout << setw(2) << "---------------------------------" << endl;
                        cout << setw(2) << " MON: ";
                        cout << h_mon1 << endl;
                        cout << setw(2) << " TUE: ";
                        cout << h_tue1 << endl;
                        cout << setw(2) << " WED: ";
                        cout << h_wed1 << endl;
                        cout << setw(2) << " Thu: ";
                        cout << h_thu1 << endl;
                        cout << setw(2) << " Fri: ";
                        cout << h_fri1 << endl << endl;

                        cout << "Would you like to join this week's study group?" << endl;
                        cout << "Enter:" << endl;
                        cout << "1. Yes" << endl;
                        cout << "2. No" << endl << endl;
                        int YorN;
                        cin >> YorN;

                        if(YorN==2)
                        {
                            cout << endl << "Press any key to exit.";
                            PAUSE();
                            system(CLEAR_SCREEN);
                            cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        }

                        else if(YorN==1)
                        {
                            system(CLEAR_SCREEN);
                            cout << "****Student Interface****" << endl << endl;
                            cout << setw(2) << "  This is the current schedule" << endl;
                            cout << setw(2) << "---------------------------------" << endl;
                            cout << setw(2) << " MON: ";
                            cout << h_mon1 << endl;
                            cout << setw(2) << " TUE: ";
                            cout << h_tue1 << endl;
                            cout << setw(2) << " WED: ";
                            cout << h_wed1 << endl;
                            cout << setw(2) << " Thu: ";
                            cout << h_thu1 << endl;
                            cout << setw(2) << " Fri: ";
                            cout << h_fri1 << endl << endl;

                            cout << "Please pick a time:" << endl;
                            cout << "|NOTE: NA means \"Not Available\"|" << endl << endl;
                            cout << "Enter:" << endl;
                            cout << "1. Monday" << endl;
                            cout << "2. Tuesday" << endl;
                            cout << "3. Wednesday" << endl;
                            cout << "4. Thursday" << endl;
                            cout << "5. Friday" << endl << endl;
                            int pick_a_time;
                            cin >> pick_a_time;

                            switch(pick_a_time)
                            {
                                case 1:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Monday " << h_mon1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 2:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Tuesday " << h_tue1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 3:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Wednesday " << h_wed1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 4:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Thursday " << h_thu1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 5:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Friday " << h_fri1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }



                                default:
                                    {
                                        cout << endl << "Invalid time, press any key to exit.";
                                        PAUSE();
                                        system(CLEAR_SCREEN);
                                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                                    }

                            }

                        }

                }

                if(No1.getTeacher()=="Lin" || No1.getTeacher()=="lin")
                {

                        cout << setw(2) << "  This is the current schedule" << endl;
                        cout << setw(2) << "---------------------------------" << endl;
                        cout << setw(2) << " MON: ";
                        cout << l_mon1 << endl;
                        cout << setw(2) << " TUE: ";
                        cout << l_tue1 << endl;
                        cout << setw(2) << " WED: ";
                        cout << l_wed1 << endl;
                        cout << setw(2) << " Thu: ";
                        cout << l_thu1 << endl;
                        cout << setw(2) << " Fri: ";
                        cout << l_fri1 << endl << endl;

                        cout << "Would you like to join this week's study group?" << endl;
                        cout << "Enter:" << endl;
                        cout << "1. Yes" << endl;
                        cout << "2. No" << endl << endl;
                        int YorN;
                        cin >> YorN;

                        if(YorN==2)
                        {
                            cout << endl << "Press any key to exit.";
                            PAUSE();
                            system(CLEAR_SCREEN);
                            cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                        }

                        else if(YorN==1)
                        {
                            system(CLEAR_SCREEN);
                            cout << "****Student Interface****" << endl << endl;
                            cout << setw(2) << "  This is the current schedule" << endl;
                            cout << setw(2) << "---------------------------------" << endl;
                            cout << setw(2) << " MON: ";
                            cout << l_mon1 << endl;
                            cout << setw(2) << " TUE: ";
                            cout << l_tue1 << endl;
                            cout << setw(2) << " WED: ";
                            cout << l_wed1 << endl;
                            cout << setw(2) << " Thu: ";
                            cout << l_thu1 << endl;
                            cout << setw(2) << " Fri: ";
                            cout << l_fri1 << endl << endl;

                            cout << "Please pick a time:" << endl;
                            cout << "|NOTE: NA means \"Not Available\"|" << endl << endl;
                            cout << "Enter:" << endl;
                            cout << "1. Monday" << endl;
                            cout << "2. Tuesday" << endl;
                            cout << "3. Wednesday" << endl;
                            cout << "4. Thursday" << endl;
                            cout << "5. Friday" << endl << endl;
                            int pick_a_time;
                            cin >> pick_a_time;

                            switch(pick_a_time)
                            {
                                case 1:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Monday " << l_mon1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 2:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Tuesday " << l_tue1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 3:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Wednesday " << l_wed1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 4:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Thursday " << l_thu1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }

                                case 5:
                                {
                                    system(CLEAR_SCREEN);
                                    cout << "****Student Interface****" << endl << endl;
                                    cout << setw(2) << "----------------------------------" << endl;
                                    cout << setw(2) << "        *Confirmation data*" << endl << endl;
                                    cout << "   ID:";
                                    No1.printID();
                                    cout << endl;
                                    cout << "   Prof:";
                                    No1.printTeacher();
                                    cout << endl;
                                    cout << "   Time:    " << "Friday " << l_fri1 << endl;
                                    cout << setw(2) << "----------------------------------" << endl << endl;

                                    instructions2();
                                    break;
                                }



                                default:
                                    {
                                        cout << endl << "Invalid time, press any key to exit.";
                                        PAUSE();
                                        system(CLEAR_SCREEN);
                                        cout << "****Computer Programming Study Group Reservation Page****" << endl << endl;
                                    }

                            }

                        }

                }

                cout << endl;

            }

    }

    }while(TorS!=-1);
}
