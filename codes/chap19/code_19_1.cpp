#include <iostream>
#include <vector>
using namespace std;

// 最小二乗法によって回帰直線 y = a*x + b の係数 (a, b) を求める
pair<double, double> linear_regression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < n; ++i) {
        sum_x  += x[i];
        sum_y  += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }
    double denom = n * sum_x2 - sum_x * sum_x;
    double a = (n * sum_xy - sum_x * sum_y) / denom;
    double b = (sum_y - a * sum_x) / n;
    return {a, b};
}

int main() {
    // データ点数と予測したい x の値を入力
    int n;
    double future_x;
    cin >> n >> future_x;

    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) cin >> x[i] >> y[i];

    // 線形回帰で係数を求める
    auto [a, b] = linear_regression(x, y);

    // 未来の値を予測
    double predicted = a * future_x + b;

    cout << "回帰直線: y = " << a << " * x + " << b << endl;
    cout << "x = " << future_x << " のときの予測値: " << predicted << endl;
}
