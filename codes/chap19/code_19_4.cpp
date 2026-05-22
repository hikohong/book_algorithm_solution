#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
using namespace std;

// 表現スコア定義
// 7=優勝, 6=準優勝, 5=3位, 4=4位, 3=ベスト8, 2=ベスト16, 1=グループ敗退, 0=不参加/予選落ち

const int NUM_WC = 22;
const int WC_YEARS[NUM_WC] = {
    1930,1934,1938,1950,1954,1958,1962,1966,
    1970,1974,1978,1982,1986,1990,1994,1998,
    2002,2006,2010,2014,2018,2022
};

struct Team {
    string name;
    vector<double> s; // スコア列
};

// ======== 3種モデル ========

// 線形回帰: s[from..n-1] の window 点で係数を求め, 次の点を予測
// window=0 の場合は全点を使用
double pred_linear(const vector<double>& s, int n, int window = 0) {
    int from = (window > 0) ? max(0, n - window) : 0;
    int cnt  = n - from;
    double sx=0, sy=0, sxy=0, sx2=0;
    for (int i = from; i < n; ++i) {
        double xi = i - from; // ローカルインデックス 0,1,...,cnt-1
        sx += xi; sy += s[i]; sxy += xi*s[i]; sx2 += xi*xi;
    }
    double D = cnt*sx2 - sx*sx;
    if (fabs(D) < 1e-9) return sy / cnt;
    double a = (cnt*sxy - sx*sy) / D;
    double b = (sy - a*sx) / cnt;
    return a*cnt + b; // 次の点 (ローカルインデックス = cnt) を予測
}

// 移動平均: 直近 window 点の平均
double pred_ma(const vector<double>& s, int n, int window) {
    int from = max(0, n - window);
    double sum = 0;
    for (int i = from; i < n; ++i) sum += s[i];
    return sum / (n - from);
}

// 指数平滑化: S_t = alpha*y_t + (1-alpha)*S_{t-1}, 最後の平滑値を予測値とする
double pred_exp(const vector<double>& s, int n, double alpha) {
    double sm = s[0];
    for (int i = 1; i < n; ++i) sm = alpha*s[i] + (1.0-alpha)*sm;
    return sm;
}

// ======== MSE 計算 (後 k 試合でクロスバリデーション) ========

struct MSEResult {
    double lin, ma3, ma4, exp03, exp05;
    string best_name() const {
        vector<pair<double,string>> v = {
            {lin,"線形回帰"},{ma3,"MA(3)"},{ma4,"MA(4)"},
            {exp03,"Exp(α=0.3)"},{exp05,"Exp(α=0.5)"}
        };
        return min_element(v.begin(), v.end())->second;
    }
    double best_val() const {
        return min({lin, ma3, ma4, exp03, exp05});
    }
};

// window=10: 線形回帰は直近10屆のみ使用
const int LIN_WINDOW = 10;

MSEResult compute_mse(const vector<double>& s, int k = 5) {
    int n = s.size();
    MSEResult m = {0,0,0,0,0};
    for (int i = n-k; i < n; ++i) {
        double actual = s[i];
        auto sq = [](double x){ return x*x; };
        m.lin   += sq(actual - pred_linear(s, i, LIN_WINDOW));
        m.ma3   += sq(actual - pred_ma(s, i, 3));
        m.ma4   += sq(actual - pred_ma(s, i, 4));
        m.exp03 += sq(actual - pred_exp(s, i, 0.3));
        m.exp05 += sq(actual - pred_exp(s, i, 0.5));
    }
    m.lin /= k; m.ma3 /= k; m.ma4 /= k; m.exp03 /= k; m.exp05 /= k;
    return m;
}

// 最良モデルで 2026 を予測
double predict_2026(const vector<double>& s, const MSEResult& m) {
    int n = s.size();
    string best = m.best_name();
    double p;
    if      (best == "線形回帰")    p = pred_linear(s, n, LIN_WINDOW);
    else if (best == "MA(3)")       p = pred_ma(s, n, 3);
    else if (best == "MA(4)")       p = pred_ma(s, n, 4);
    else if (best == "Exp(α=0.3)") p = pred_exp(s, n, 0.3);
    else                            p = pred_exp(s, n, 0.5);
    return max(0.0, p);
}

int main() {
    // ======== 歴史データ (1930-2022, 22大会) ========
    vector<Team> teams = {
        // Brazil: 全大会参加 (5回優勝)
        {"Brazil",    {5,2,5,6,3,7,7,1,7,4,5,2,3,2,7,6,7,3,3,4,3,3}},
        // Germany/W.Germany: 1930不参加, 1950出場禁止 (4回優勝)
        {"Germany",   {0,5,2,0,7,4,3,6,5,7,2,6,6,7,3,3,6,5,5,7,1,1}},
        // Argentina (3回優勝)
        {"Argentina", {6,2,0,0,0,1,1,3,0,2,7,2,7,6,2,3,1,3,3,6,2,7}},
        // France (2回優勝)
        {"France",    {1,2,3,0,1,5,0,1,0,1,1,4,5,0,0,7,1,6,1,3,7,6}},
        // Italy (4回優勝, 2018/2022予選落ち)
        {"Italy",     {0,7,7,1,1,0,1,1,6,1,4,7,2,5,6,3,2,7,1,1,0,0}},
        // Spain (1回優勝)
        {"Spain",     {3,3,0,4,1,0,1,1,0,1,1,2,3,2,3,1,3,2,7,1,2,3}},
    };

    cout << fixed << setprecision(2);
    cout << "========================================\n";
    cout << "  FIFA ワールドカップ 歴史データ分析\n";
    cout << "  スコア: 7=優勝 6=準V 5=3位 4=4位\n";
    cout << "         3=8強  2=16強 1=GS敗退 0=不参加\n";
    cout << "========================================\n\n";

    // 歴史スコア一覧
    cout << setw(6) << "年";
    for (auto& t : teams) cout << setw(11) << t.name;
    cout << "\n" << string(72, '-') << "\n";
    for (int i = 0; i < NUM_WC; ++i) {
        cout << setw(6) << WC_YEARS[i];
        for (auto& t : teams) cout << setw(11) << (int)t.s[i];
        cout << "\n";
    }

    // 平均・最大
    cout << string(72, '-') << "\n";
    cout << setw(6) << "平均";
    for (auto& t : teams) {
        double avg = accumulate(t.s.begin(), t.s.end(), 0.0) / t.s.size();
        cout << setw(11) << avg;
    }
    cout << "\n";
    cout << setw(6) << "最大";
    for (auto& t : teams) cout << setw(11) << (int)*max_element(t.s.begin(), t.s.end());
    cout << "\n\n";

    // 直近5大会の勝率曲線 (2002-2022)
    cout << "========================================\n";
    cout << "  直近5大会 表現推移 (2002-2022)\n";
    cout << "========================================\n";
    cout << setw(6) << "年";
    for (auto& t : teams) cout << setw(11) << t.name;
    cout << "\n";
    for (int i = NUM_WC-5; i < NUM_WC; ++i) {
        cout << setw(6) << WC_YEARS[i];
        for (auto& t : teams) cout << setw(11) << (int)t.s[i];
        cout << "\n";
    }

    // MSE クロスバリデーション
    cout << "\n========================================\n";
    cout << "  モデル別 MSE (後5大会クロスバリデーション)\n";
    cout << "  ※ 小さいほど実績データへの適合度が高い\n";
    cout << "========================================\n";
    cout << setw(11) << "チーム"
         << setw(13) << "線形回帰(10屆)"
         << setw(7)  << "MA(3)"
         << setw(7)  << "MA(4)"
         << setw(10) << "Exp(0.3)"
         << setw(10) << "Exp(0.5)"
         << setw(12) << "最良モデル" << "\n";
    cout << string(66, '-') << "\n";

    vector<MSEResult> mse_results;
    for (auto& t : teams) {
        MSEResult m = compute_mse(t.s, 5);
        mse_results.push_back(m);
        cout << setw(11) << t.name
             << setw(13)  << m.lin
             << setw(7)  << m.ma3
             << setw(7)  << m.ma4
             << setw(10) << m.exp03
             << setw(10) << m.exp05
             << setw(12) << m.best_name() << "\n";
    }

    // 2026 予測
    cout << "\n========================================\n";
    cout << "  2026 ワールドカップ 優勝確率予測\n";
    cout << "========================================\n";

    vector<pair<double,int>> ranked;
    vector<double> raw_pred;
    for (int i = 0; i < (int)teams.size(); ++i) {
        double p = predict_2026(teams[i].s, mse_results[i]);
        raw_pred.push_back(p);
        ranked.push_back({p, i});
    }
    sort(ranked.rbegin(), ranked.rend());

    double total = accumulate(raw_pred.begin(), raw_pred.end(), 0.0);
    if (total < 1e-9) total = 1.0;

    cout << setw(4) << "順位"
         << setw(11) << "チーム"
         << setw(12) << "予測スコア"
         << setw(10) << "優勝確率"
         << setw(14) << "使用モデル" << "\n";
    cout << string(51, '-') << "\n";
    for (int rank = 0; rank < (int)ranked.size(); ++rank) {
        auto [score, idx] = ranked[rank];
        double prob = score / total * 100.0;
        cout << setw(4) << rank+1
             << setw(11) << teams[idx].name
             << setw(12) << score
             << setw(9)  << prob << "%"
             << setw(14) << mse_results[idx].best_name() << "\n";
    }

    cout << "\n注意: 本予測は過去の表現スコアのみに基づく統計モデルです。\n";
    cout << "      選手コンディション・組み合わせ・戦術は考慮していません。\n";

    return 0;
}
