#include <iostream>
#include <vector>
using namespace std;

// 指数平滑化: S_t = alpha * y_t + (1 - alpha) * S_{t-1}
// 次ステップの予測値は S_n (最後の平滑値)
vector<double> exponential_smoothing(const vector<double>& data, double alpha) {
    int n = data.size();
    vector<double> s(n);
    s[0] = data[0]; // 初期値は最初のデータ点
    for (int t = 1; t < n; ++t) {
        s[t] = alpha * data[t] + (1.0 - alpha) * s[t - 1];
    }
    return s;
}

int main() {
    int n;
    double alpha;
    cin >> n >> alpha;

    vector<double> data(n);
    for (int i = 0; i < n; ++i) cin >> data[i];

    vector<double> s = exponential_smoothing(data, alpha);

    cout << "指数平滑化 (alpha = " << alpha << "):" << endl;
    for (int t = 0; t < n; ++t) {
        cout << "  S[" << t << "] = " << s[t] << endl;
    }

    // S_n が次のステップの予測値
    cout << "次のステップの予測値: " << s.back() << endl;
}
