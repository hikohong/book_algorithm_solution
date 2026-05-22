#include <iostream>
#include <vector>
#include <cmath>
#include <string>
using namespace std;

// ======== テスト対象の関数 (code_19_1〜4 と同実装) ========

pair<double,double> linear_regression(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    double sx=0, sy=0, sxy=0, sx2=0;
    for (int i = 0; i < n; ++i) {
        sx += x[i]; sy += y[i]; sxy += x[i]*y[i]; sx2 += x[i]*x[i];
    }
    double D = n*sx2 - sx*sx;
    if (fabs(D) < 1e-9) return {0.0, sy/n};
    double a = (n*sxy - sx*sy) / D;
    double b = (sy - a*sx) / n;
    return {a, b};
}

// ウィンドウ付き線形回帰: s[n-window..n-1] の window 点で次の点を予測
double pred_linear_windowed(const vector<double>& s, int n, int window) {
    int from = max(0, n - window);
    int cnt  = n - from;
    double sx=0, sy=0, sxy=0, sx2=0;
    for (int i = from; i < n; ++i) {
        double xi = i - from;
        sx += xi; sy += s[i]; sxy += xi*s[i]; sx2 += xi*xi;
    }
    double D = cnt*sx2 - sx*sx;
    if (fabs(D) < 1e-9) return sy / cnt;
    double a = (cnt*sxy - sx*sy) / D;
    double b = (sy - a*sx) / cnt;
    return a*cnt + b;
}

vector<double> moving_average(const vector<double>& data, int k) {
    int n = data.size();
    vector<double> ma;
    for (int i = k-1; i < n; ++i) {
        double sum = 0;
        for (int j = i-k+1; j <= i; ++j) sum += data[j];
        ma.push_back(sum / k);
    }
    return ma;
}

vector<double> exponential_smoothing(const vector<double>& data, double alpha) {
    int n = data.size();
    vector<double> s(n);
    s[0] = data[0];
    for (int t = 1; t < n; ++t) s[t] = alpha*data[t] + (1.0-alpha)*s[t-1];
    return s;
}

double mse(const vector<double>& pred, const vector<double>& actual) {
    double sum = 0;
    for (int i = 0; i < (int)pred.size(); ++i) {
        double e = pred[i] - actual[i];
        sum += e*e;
    }
    return sum / pred.size();
}

// ======== テストフレームワーク ========

int pass_count = 0, fail_count = 0;

void check(const string& name, bool ok) {
    if (ok) {
        cout << "  [PASS] " << name << "\n";
        ++pass_count;
    } else {
        cout << "  [FAIL] " << name << "\n";
        ++fail_count;
    }
}

bool near(double a, double b, double eps = 1e-6) {
    return fabs(a - b) < eps;
}

// ======== 線形回帰テスト ========

void test_linear_regression() {
    cout << "\n[線形回帰 テスト]\n";

    // TC1: 完全線形 y = 2x + 1
    {
        vector<double> x = {0,1,2,3,4};
        vector<double> y = {1,3,5,7,9};
        auto [a, b] = linear_regression(x, y);
        check("TC1: 傾き a = 2.0",   near(a, 2.0));
        check("TC1: 切片 b = 1.0",   near(b, 1.0));
        double pred = a*5 + b;
        check("TC1: x=5 の予測 = 11.0", near(pred, 11.0));
    }

    // TC2: 水平線 y = 5 (傾きゼロ)
    {
        vector<double> x = {0,1,2,3,4};
        vector<double> y = {5,5,5,5,5};
        auto [a, b] = linear_regression(x, y);
        check("TC2: 水平線 a = 0.0",  near(a, 0.0));
        check("TC2: 水平線 b = 5.0",  near(b, 5.0));
    }

    // TC3: 単回帰でMSEがゼロ (完全フィット)
    {
        vector<double> x = {1,2,3};
        vector<double> y = {3,5,7}; // y = 2x + 1
        auto [a, b] = linear_regression(x, y);
        vector<double> fitted;
        for (double xi : x) fitted.push_back(a*xi + b);
        check("TC3: 完全線形データのMSE = 0", near(mse(fitted, y), 0.0, 1e-9));
    }

    // TC4: 2点 (傾き確定)
    {
        vector<double> x = {0, 4};
        vector<double> y = {1, 9}; // a=2, b=1
        auto [a, b] = linear_regression(x, y);
        check("TC4: 2点回帰 a = 2.0", near(a, 2.0));
        check("TC4: 2点回帰 b = 1.0", near(b, 1.0));
    }

    // TC5: 負の傾き y = -3x + 10
    {
        vector<double> x = {0,1,2,3};
        vector<double> y = {10,7,4,1};
        auto [a, b] = linear_regression(x, y);
        check("TC5: 負傾き a = -3.0",  near(a, -3.0));
        check("TC5: 負傾き b = 10.0",  near(b, 10.0));
    }

    // TC6: ウィンドウ付き回帰 — window=3 は末尾3点のみ使用
    // データ: [0,0,0,10,20,30] (末尾3点: 10,20,30 → 傾き=10)
    // 次の予測値 = 40
    {
        vector<double> s = {0,0,0,10,20,30};
        int n = s.size();
        double pred = pred_linear_windowed(s, n, 3);
        check("TC6: window=3 で末尾3点のみ回帰, 予測=40", near(pred, 40.0));
    }

    // TC7: window >= n → 全点使用と同等
    {
        vector<double> s = {1,3,5,7,9}; // y=2x+1
        int n = s.size();
        double pred_full   = pred_linear_windowed(s, n, 100); // window>n
        double pred_normal = pred_linear_windowed(s, n, n);   // window=n
        check("TC7: window>=n は全点使用と同等", near(pred_full, pred_normal));
        check("TC7: 完全線形データ window=n の予測=11", near(pred_full, 11.0));
    }

    // TC8: window=10 でトレンド反転を検出
    // 前半12点: 上昇, 後半10点: 下降 → window=10 は下降トレンドを捉える
    {
        vector<double> s = {1,2,3,4,5,6,7,8,9,10,11,12, // 上昇
                            11,10,9,8,7,6,5,4,3,2};      // 下降 (10点)
        int n = s.size(); // 22
        double pred = pred_linear_windowed(s, n, 10);
        check("TC8: window=10 が下降トレンドを捉えて pred < 2", pred < 2.0);
        double pred_all = pred_linear_windowed(s, n, 22); // 全点 (window=n)
        check("TC8: window=22 (全点) の予測は window=10 より大きい", pred_all > pred);
    }
}

// ======== 移動平均テスト ========

void test_moving_average() {
    cout << "\n[移動平均 テスト]\n";

    // TC1: 定数列 → 全MA値が同じ
    {
        vector<double> data = {5,5,5,5,5};
        auto ma = moving_average(data, 3);
        bool ok = true;
        for (double v : ma) if (!near(v, 5.0)) ok = false;
        check("TC1: 定数列のMA = 定数",  ok);
    }

    // TC2: 等差列 [10,20,30,40,50], k=3
    {
        vector<double> data = {10,20,30,40,50};
        auto ma = moving_average(data, 3);
        // 期待値: 20, 30, 40
        check("TC2: MA[0] = 20.0", near(ma[0], 20.0));
        check("TC2: MA[1] = 30.0", near(ma[1], 30.0));
        check("TC2: MA[2] = 40.0", near(ma[2], 40.0));
        check("TC2: MAの要素数 = n-k+1", ma.size() == 3u);
    }

    // TC3: k=1 → MAがデータそのもの
    {
        vector<double> data = {3,1,4,1,5};
        auto ma = moving_average(data, 1);
        bool ok = true;
        for (int i = 0; i < (int)data.size(); ++i) if (!near(ma[i], data[i])) ok = false;
        check("TC3: k=1のMA = 元データ", ok);
    }

    // TC4: k=n → 全体の平均1個
    {
        vector<double> data = {2,4,6,8};
        auto ma = moving_average(data, 4);
        check("TC4: k=n のMA要素数 = 1",  ma.size() == 1u);
        check("TC4: k=n のMA値 = 全体平均", near(ma[0], 5.0));
    }

    // TC5: MSE（定数列に対してMAはMSE=0）
    {
        vector<double> data = {7,7,7,7,7};
        auto ma = moving_average(data, 3);
        vector<double> actual(ma.size(), 7.0);
        check("TC5: 定数列MA の MSE = 0", near(mse(ma, actual), 0.0, 1e-9));
    }
}

// ======== 指数平滑化テスト ========

void test_exponential_smoothing() {
    cout << "\n[指数平滑化 テスト]\n";

    // TC1: alpha=1.0 → S[t] = y[t] (直近値のみ使用)
    {
        vector<double> data = {3,7,2,9,4};
        auto s = exponential_smoothing(data, 1.0);
        bool ok = true;
        for (int i = 0; i < (int)data.size(); ++i) if (!near(s[i], data[i])) ok = false;
        check("TC1: alpha=1.0 で S[t]=y[t]", ok);
    }

    // TC2: alpha=0.0 → S[t] = S[0] = y[0] (初期値固定)
    {
        vector<double> data = {5,10,15,20};
        auto s = exponential_smoothing(data, 0.0);
        bool ok = true;
        for (double v : s) if (!near(v, 5.0)) ok = false;
        check("TC2: alpha=0.0 で S[t]=y[0]", ok);
    }

    // TC3: 手動計算検証 alpha=0.5, data=[10,20]
    //   S[0]=10, S[1]=0.5*20 + 0.5*10 = 15
    {
        vector<double> data = {10, 20};
        auto s = exponential_smoothing(data, 0.5);
        check("TC3: S[0] = 10.0", near(s[0], 10.0));
        check("TC3: S[1] = 15.0", near(s[1], 15.0));
    }

    // TC4: 定数列 → 全S[t]が同じ
    {
        vector<double> data = {4,4,4,4,4};
        auto s = exponential_smoothing(data, 0.3);
        bool ok = true;
        for (double v : s) if (!near(v, 4.0)) ok = false;
        check("TC4: 定数列のESは定数", ok);
    }

    // TC5: 手動計算検証 alpha=0.3, data=[10,20,15]
    //   S[0]=10
    //   S[1]=0.3*20 + 0.7*10 = 6 + 7 = 13
    //   S[2]=0.3*15 + 0.7*13 = 4.5 + 9.1 = 13.6
    {
        vector<double> data = {10, 20, 15};
        auto s = exponential_smoothing(data, 0.3);
        check("TC5: S[0] = 10.0",  near(s[0], 10.0));
        check("TC5: S[1] = 13.0",  near(s[1], 13.0));
        check("TC5: S[2] = 13.6",  near(s[2], 13.6, 1e-6));
    }

    // TC6: MSE（定数列に対してESはMSE=0）
    {
        vector<double> data = {6,6,6,6,6};
        auto s = exponential_smoothing(data, 0.4);
        check("TC6: 定数列ES の MSE = 0", near(mse(s, data), 0.0, 1e-9));
    }
}

// ======== MSE 関数テスト ========

void test_mse() {
    cout << "\n[MSE 関数テスト]\n";

    check("TC1: 完全一致 MSE = 0",
        near(mse({1,2,3}, {1,2,3}), 0.0));

    check("TC2: 全差=2 の MSE = 4",
        near(mse({3,4,5}, {1,2,3}), 4.0));

    check("TC3: 1点 MSE = 誤差の2乗",
        near(mse({5}, {3}), 4.0));

    check("TC4: 対称性 MSE(a,b) == MSE(b,a)",
        near(mse({1,3,5}, {2,4,6}), mse({2,4,6}, {1,3,5})));
}

// ======== メイン ========

int main() {
    cout << "================================================\n";
    cout << "  chap19 回帰テスト\n";
    cout << "================================================\n";

    test_linear_regression();
    test_moving_average();
    test_exponential_smoothing();
    test_mse();

    cout << "\n================================================\n";
    cout << "  結果: " << pass_count << " PASS / "
         << fail_count << " FAIL\n";
    cout << "================================================\n";

    return fail_count > 0 ? 1 : 0;
}
