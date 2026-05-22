#include <iostream>
#include <vector>
using namespace std;

// 直近 k 個の値の単純移動平均を求め，次の値を予測する
vector<double> moving_average(const vector<double>& data, int k) {
    int n = data.size();
    vector<double> ma;
    for (int i = k - 1; i < n; ++i) {
        double sum = 0;
        for (int j = i - k + 1; j <= i; ++j) sum += data[j];
        ma.push_back(sum / k);
    }
    return ma;
}

int main() {
    int n, k;
    cin >> n >> k;

    vector<double> data(n);
    for (int i = 0; i < n; ++i) cin >> data[i];

    vector<double> ma = moving_average(data, k);

    cout << "移動平均 (窓幅 k = " << k << "):" << endl;
    for (int i = 0; i < (int)ma.size(); ++i) {
        cout << "  区間 [" << i << ", " << i + k - 1 << "] の平均: " << ma[i] << endl;
    }

    // 最後の移動平均値を次のステップの予測値とする
    cout << "次のステップの予測値: " << ma.back() << endl;
}
