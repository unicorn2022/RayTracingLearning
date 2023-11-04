#include <iostream>
#include <thread>
#include <future>
#include <ctime>
using namespace std;
typedef long long LL;

LL maxn = 1000000000ll;

int now = 0;
void calc(LL L, LL R, promise<LL>& p);

void Init(clock_t t) {
    promise<LL> p1, p2;
    future<LL> f1 = p1.get_future(), f2 = p2.get_future();


    thread t1(calc, 1, maxn/2, ref(p1));
    thread t2(calc, maxn / 2 + 1, maxn, ref(p2));
    t1.detach();
    cout << "t1 detach, time is " << clock() - t << endl;
    t2.detach();
    cout << "t2 detach, time is " << clock() - t << endl;


    while (f1.wait_for(chrono::seconds(0)) != future_status::ready);
    cout << "f1 ready, time is " << clock() - t << endl;
    while (f2.wait_for(chrono::seconds(0)) != future_status::ready);
    cout << "f2 ready, time is " << clock() - t << endl;

    LL sum = f1.get() + f2.get();

    cout << "sum = " << sum << endl;
}

void Init2(clock_t t) {
    promise<LL> p;
    future<LL> f = p.get_future();

    calc(1, maxn, p);
    
    LL sum = f.get();
    cout << "sum = " << sum << endl;
}

int main() {
    auto t = clock();

    Init(t);

    cout << "time is " << clock() - t << endl;
    return 0;
}

void calc(LL L, LL R, promise<LL>& p) {
    LL sum = 0;
    for (LL i = L; i <= R; i=i+1)
        sum += i;
    p.set_value(sum);
}