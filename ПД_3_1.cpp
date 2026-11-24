// ============================================================================
// Проектная деятельность. Модуль 3. Кластерный анализ технических объектов.
// Вариант 3724-4: 60 объектов, 4 признака, 4 кластера.
//
// Метод кластеризации : агломеративный, метод связи Ward
//                       (пересчёт расстояний по формуле Ланса-Уильямса)
// Метрика расстояния  : евклидова
// Метрика качества    : средний коэффициент силуэта (silhouette)
// Повторные запуски   : случайная подвыборка 80% объектов без возвращения
//
// Разработчики : [ФИО 1], [ФИО 2], [ФИО 3]
// Учебная группа: [НОМЕР ГРУППЫ]
//
// Сборка: g++ -std=c++17 -O2 ward_clustering.cpp -o ward_clustering
//         (MSVC: добавить ключ /utf-8, стандарт C++17)
// Файл должен быть сохранён в кодировке UTF-8.
// ============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
// windows.h определяет макросы max/min, которые ломают std::max, std::min
// и numeric_limits<...>::max(). NOMINMAX отключает эти макросы.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace std;

// ----------------------------- Настройки -----------------------------------
const int      N_OBJECTS = 60;          // число объектов
const int      N_FEATURES = 4;          // число признаков
const int      K_CLUSTERS = 4;          // число кластеров (по заданию)
const double   SUBSAMPLE_FRACTION = 0.8; // доля объектов в подвыборке
const bool     USE_STANDARDIZATION = false; // z-стандартизация признаков пары
const unsigned SEED = 12345;            // для воспроизводимости результатов
const int      PLOT_W = 64;             // ширина графика в консоли (символов)
const int      PLOT_H = 22;             // высота графика в консоли (строк)

typedef array<double, 2> P2;            // точка в двумерном пространстве

// ----------------------- Таблица экспериментальных данных ------------------
const double DATA[N_OBJECTS][N_FEATURES] = {
    {18, 19, 22, 55}, {17, 16, 26, 56}, {15, 12, 38, 57}, {16, 11, 32, 57}, {19, 17, 24, 57},
    {15, 18, 22, 59}, {17, 13, 34, 60}, {18, 11, 36, 60}, {18, 15, 26, 61}, {14, 16, 22, 62},
    {17, 19, 30, 62}, {15, 17, 32, 63}, {18, 12, 38, 63}, {16, 13, 34, 64}, {15, 11, 24, 65},
    {27, 15, 26, 37}, {28, 16, 22, 38}, {25, 19, 30, 39}, {29, 17, 32, 39}, {24, 13, 38, 41},
    {27, 19, 34, 41}, {22, 16, 26, 42}, {26, 12, 38, 42}, {27, 19, 32, 43}, {21, 16, 24, 44},
    {25, 12, 22, 44}, {23, 11, 34, 45}, {26, 17, 36, 45}, {22, 18, 22, 46}, {24, 13, 28, 47},
    {30, 11, 34, 51}, {28, 15, 26, 52}, {32, 16, 38, 52}, {26, 19, 32, 53}, {30, 17, 24, 53},
    {28, 12, 22, 54}, {31, 13, 34, 54}, {33, 11, 36, 54}, {25, 15, 22, 55}, {29, 16, 28, 56},
    {32, 19, 30, 56}, {27, 17, 26, 57}, {26, 13, 38, 58}, {30, 19, 34, 58}, {28, 16, 24, 59},
    {37, 12, 36, 35}, {33, 19, 26, 41}, {34, 16, 22, 37}, {35, 12, 30, 47}, {35, 11, 32, 43},
    {36, 17, 38, 41}, {35, 18, 34, 36}, {37, 13, 24, 43}, {37, 11, 26, 38}, {38, 15, 22, 48},
    {39, 16, 30, 43}, {39, 19, 32, 38}, {39, 17, 38, 34}, {40, 12, 34, 38}, {41, 13, 26, 42}
};

// ----------------------------- Вывод ---------------------------------------
// Всё, что выводится на экран, дублируется в файл протокола protocol.txt
ofstream g_log;

void emit(const string& s) {
    cout << s;
    if (g_log.is_open()) g_log << s;
}

string fmt(double v, int prec) {
    ostringstream o;
    o << fixed << setprecision(prec) << v;
    return o.str();
}

// ----------------------------- Подготовка данных ---------------------------
struct PairData {
    vector<P2> raw;   // исходные значения (для графиков)
    vector<P2> work;  // значения, в которых выполняется кластеризация
};

// Берёт два столбца таблицы (fi, fj нумеруются с 0)
PairData extractPair(int fi, int fj) {
    PairData pd;
    int f[2] = { fi, fj };
    double mean[2] = { 0.0, 0.0 };
    double sd[2] = { 1.0, 1.0 };
    if (USE_STANDARDIZATION) {
        for (int c = 0; c < 2; ++c) {
            double s = 0.0;
            for (int i = 0; i < N_OBJECTS; ++i) s += DATA[i][f[c]];
            mean[c] = s / N_OBJECTS;
            double v = 0.0;
            for (int i = 0; i < N_OBJECTS; ++i) {
                double d = DATA[i][f[c]] - mean[c];
                v += d * d;
            }
            sd[c] = sqrt(v / N_OBJECTS);
        }
    }
    for (int i = 0; i < N_OBJECTS; ++i) {
        P2 r = { DATA[i][fi], DATA[i][fj] };
        P2 w = { (r[0] - mean[0]) / sd[0], (r[1] - mean[1]) / sd[1] };
        pd.raw.push_back(r);
        pd.work.push_back(w);
    }
    return pd;
}

vector<P2> subset(const vector<P2>& v, const vector<int>& idx) {
    vector<P2> res;
    res.reserve(idx.size());
    for (int i : idx) res.push_back(v[i]);
    return res;
}

// ----------------------------- Кластеризация Ward --------------------------
// Возвращает метки кластеров 0..k-1 для каждой точки из pts.
// Метки нумеруются по порядку первого появления в pts.
//
// Хранится матрица D[i][j] = 2*nA*nB/(nA+nB) * |cA-cB|^2.
// Для одиночных точек D = квадрат евклидова расстояния.
// После слияния A и B с кластером L пересчёт по формуле Ланса-Уильямса:
//   D(L,AB) = ((nA+nL)*D(L,A) + (nB+nL)*D(L,B) - nL*D(A,B)) / (nA+nB+nL)
// Высота слияния на дендрограмме (как в scipy) = sqrt(D).
vector<int> wardCluster(const vector<P2>& pts, int k) {
    int m = (int)pts.size();
    vector<vector<double>> D(m, vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j) {
            double dx = pts[i][0] - pts[j][0];
            double dy = pts[i][1] - pts[j][1];
            D[i][j] = D[j][i] = dx * dx + dy * dy;
        }

    vector<int> sz(m, 1);
    vector<bool> active(m, true);
    vector<vector<int>> members(m);
    for (int i = 0; i < m; ++i) members[i].push_back(i);

    int clusters = m;
    while (clusters > k) {
        // поиск пары кластеров с минимальным D
        double best = numeric_limits<double>::max();
        int bi = -1, bj = -1;
        for (int i = 0; i < m; ++i) {
            if (!active[i]) continue;
            for (int j = i + 1; j < m; ++j) {
                if (!active[j]) continue;
                if (D[i][j] < best) { best = D[i][j]; bi = i; bj = j; }
            }
        }
        // пересчёт расстояний от нового кластера (bi U bj) до остальных
        int ni = sz[bi], nj = sz[bj];
        for (int l = 0; l < m; ++l) {
            if (!active[l] || l == bi || l == bj) continue;
            int nl = sz[l];
            double d = ((ni + nl) * D[l][bi] + (nj + nl) * D[l][bj] - nl * D[bi][bj])
                / (double)(ni + nj + nl);
            D[l][bi] = D[bi][l] = d;
        }
        // слияние: bj вливается в bi
        sz[bi] += nj;
        members[bi].insert(members[bi].end(), members[bj].begin(), members[bj].end());
        active[bj] = false;
        --clusters;
    }

    vector<int> owner(m, -1);
    for (int s = 0; s < m; ++s)
        if (active[s])
            for (int p : members[s]) owner[p] = s;

    vector<int> slotLabel(m, -1), labels(m, -1);
    int next = 0;
    for (int p = 0; p < m; ++p) {
        int s = owner[p];
        if (slotLabel[s] < 0) slotLabel[s] = next++;
        labels[p] = slotLabel[s];
    }
    return labels;
}

// ----------------------------- Коэффициент силуэта -------------------------
// a(i) - среднее расстояние до остальных точек своего кластера
// b(i) - наименьшее из средних расстояний до точек чужих кластеров
// s(i) = (b - a) / max(a, b); если в кластере одна точка, s(i) = 0
double silhouette(const vector<P2>& pts, const vector<int>& lab, int k) {
    int m = (int)pts.size();
    double total = 0.0;
    for (int i = 0; i < m; ++i) {
        vector<double> sum(k, 0.0);
        vector<int> cnt(k, 0);
        for (int j = 0; j < m; ++j) {
            if (j == i) continue;
            double dx = pts[i][0] - pts[j][0];
            double dy = pts[i][1] - pts[j][1];
            sum[lab[j]] += sqrt(dx * dx + dy * dy);
            cnt[lab[j]]++;
        }
        int own = lab[i];
        if (cnt[own] == 0) continue;  // одноточечный кластер: s = 0
        double a = sum[own] / cnt[own];
        double b = numeric_limits<double>::max();
        for (int c = 0; c < k; ++c)
            if (c != own && cnt[c] > 0) b = min(b, sum[c] / cnt[c]);
        double mx = max(a, b);
        if (mx > 0.0 && b != numeric_limits<double>::max()) total += (b - a) / mx;
    }
    return total / m;
}

// ----------------------------- Статистика ----------------------------------
struct Stats { double mean, sd, mn, med, mx; };

Stats calcStats(vector<double> v) {
    Stats s;
    int n = (int)v.size();
    s.mean = accumulate(v.begin(), v.end(), 0.0) / n;
    double q = 0.0;
    for (double x : v) q += (x - s.mean) * (x - s.mean);
    s.sd = (n > 1) ? sqrt(q / (n - 1)) : 0.0;  // выборочное СКО
    sort(v.begin(), v.end());
    s.mn = v.front();
    s.mx = v.back();
    s.med = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return s;
}

// ----------------------------- Графики -------------------------------------
// Описание кластеров: номер, число точек, центр
string clusterLegend(const vector<P2>& pts, const vector<int>& lab, int k) {
    vector<int> cnt(k, 0);
    vector<double> sx(k, 0.0), sy(k, 0.0);
    for (size_t i = 0; i < pts.size(); ++i) {
        cnt[lab[i]]++;
        sx[lab[i]] += pts[i][0];
        sy[lab[i]] += pts[i][1];
    }
    ostringstream o;
    for (int c = 0; c < k; ++c) {
        o << "  [" << c + 1 << "] Кластер " << c + 1 << ": n = " << cnt[c];
        if (cnt[c] > 0) o << ", центр = (" << fmt(sx[c] / cnt[c], 1) << "; " << fmt(sy[c] / cnt[c], 1) << ")";
        o << "\n";
    }
    return o.str();
}

// Диаграмма рассеяния символами в консоли. Цифра = номер кластера,
// '*' = в одной ячейке оказались точки разных кластеров.
string asciiPlot(const vector<P2>& pts, const vector<int>& lab, int fi, int fj) {
    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    for (const auto& p : pts) {
        xmin = min(xmin, p[0]); xmax = max(xmax, p[0]);
        ymin = min(ymin, p[1]); ymax = max(ymax, p[1]);
    }
    if (xmax - xmin < 1e-12) { xmin -= 1; xmax += 1; }
    if (ymax - ymin < 1e-12) { ymin -= 1; ymax += 1; }

    vector<string> grid(PLOT_H, string(PLOT_W, ' '));
    for (size_t i = 0; i < pts.size(); ++i) {
        int col = (int)lround((pts[i][0] - xmin) / (xmax - xmin) * (PLOT_W - 1));
        int row = (int)lround((ymax - pts[i][1]) / (ymax - ymin) * (PLOT_H - 1));
        char ch = (char)('1' + lab[i]);
        char& cell = grid[row][col];
        if (cell == ' ') cell = ch;
        else if (cell != ch) cell = '*';
    }

    ostringstream o;
    o << "  Ось Y: признак " << fj + 1 << ",  ось X: признак " << fi + 1 << "\n";
    for (int r = 0; r < PLOT_H; ++r) {
        string lbl = (r == 0) ? fmt(ymax, 1) : (r == PLOT_H - 1 ? fmt(ymin, 1) : "");
        o << setw(7) << lbl << " |" << grid[r] << "\n";
    }
    o << string(8, ' ') << "+" << string(PLOT_W, '-') << "\n";
    string lo = fmt(xmin, 1), hi = fmt(xmax, 1);
    int gap = max(1, PLOT_W - (int)lo.size() - (int)hi.size());
    o << string(9, ' ') << lo << string(gap, ' ') << hi << "\n";
    return o.str();
}

// Диаграмма рассеяния в файле SVG (для вставки в отчёт)
void saveScatterSVG(const string& fname, const vector<P2>& pts, const vector<int>& lab,
    int k, int fi, int fj, const string& title) {
    ofstream f(fname);
    if (!f.is_open()) {
        emit("Не удалось создать файл " + fname + "\n");
        return;
    }
    const int W = 840, H = 560, ml = 70, mr = 250, mt = 60, mb = 60;
    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    for (const auto& p : pts) {
        xmin = min(xmin, p[0]); xmax = max(xmax, p[0]);
        ymin = min(ymin, p[1]); ymax = max(ymax, p[1]);
    }
    if (xmax - xmin < 1e-12) { xmin -= 1; xmax += 1; }
    if (ymax - ymin < 1e-12) { ymin -= 1; ymax += 1; }
    double px = 0.05 * (xmax - xmin), py = 0.05 * (ymax - ymin);
    double X0 = xmin - px, X1 = xmax + px, Y0 = ymin - py, Y1 = ymax + py;
    auto X = [&](double v) { return ml + (v - X0) / (X1 - X0) * (W - ml - mr); };
    auto Y = [&](double v) { return (H - mb) - (v - Y0) / (Y1 - Y0) * (H - mt - mb); };
    static const char* colors[] = { "#e6194B", "#3cb44b", "#4363d8", "#f58231",
                                    "#911eb4", "#42d4f4", "#f032e6", "#808000" };

    f << fixed << setprecision(1);
    f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H << "\">\n";
    f << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    f << "<text x=\"" << ml << "\" y=\"32\" font-family=\"Arial\" font-size=\"18\" font-weight=\"bold\">"
        << title << "</text>\n";
    // оси
    f << "<line x1=\"" << ml << "\" y1=\"" << H - mb << "\" x2=\"" << W - mr << "\" y2=\"" << H - mb
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
    f << "<line x1=\"" << ml << "\" y1=\"" << H - mb << "\" x2=\"" << ml << "\" y2=\"" << mt
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
    // метки на осях (5 штук от минимума до максимума данных)
    for (int t = 0; t <= 4; ++t) {
        double vx = xmin + t * (xmax - xmin) / 4.0;
        double vy = ymin + t * (ymax - ymin) / 4.0;
        f << "<line x1=\"" << X(vx) << "\" y1=\"" << H - mb << "\" x2=\"" << X(vx) << "\" y2=\"" << H - mb + 5
            << "\" stroke=\"black\"/>\n";
        f << "<text x=\"" << X(vx) << "\" y=\"" << H - mb + 20
            << "\" font-family=\"Arial\" font-size=\"12\" text-anchor=\"middle\">" << vx << "</text>\n";
        f << "<line x1=\"" << ml - 5 << "\" y1=\"" << Y(vy) << "\" x2=\"" << ml << "\" y2=\"" << Y(vy)
            << "\" stroke=\"black\"/>\n";
        f << "<text x=\"" << ml - 10 << "\" y=\"" << Y(vy) + 4
            << "\" font-family=\"Arial\" font-size=\"12\" text-anchor=\"end\">" << vy << "</text>\n";
    }
    f << "<text x=\"" << (ml + W - mr) / 2 << "\" y=\"" << H - 15
        << "\" font-family=\"Arial\" font-size=\"14\" text-anchor=\"middle\">Признак " << fi + 1 << "</text>\n";
    f << "<text x=\"20\" y=\"" << (mt + H - mb) / 2
        << "\" font-family=\"Arial\" font-size=\"14\" text-anchor=\"middle\" transform=\"rotate(-90 20,"
        << (mt + H - mb) / 2 << ")\">Признак " << fj + 1 << "</text>\n";
    // точки
    for (size_t i = 0; i < pts.size(); ++i)
        f << "<circle cx=\"" << X(pts[i][0]) << "\" cy=\"" << Y(pts[i][1]) << "\" r=\"5\" fill=\""
        << colors[lab[i] % 8] << "\" fill-opacity=\"0.75\" stroke=\"black\" stroke-width=\"0.6\"/>\n";
    // центры и легенда
    vector<int> cnt(k, 0);
    vector<double> sx(k, 0.0), sy(k, 0.0);
    for (size_t i = 0; i < pts.size(); ++i) {
        cnt[lab[i]]++;
        sx[lab[i]] += pts[i][0];
        sy[lab[i]] += pts[i][1];
    }
    for (int c = 0; c < k; ++c) {
        if (cnt[c] > 0) {
            double cx = X(sx[c] / cnt[c]), cy = Y(sy[c] / cnt[c]);
            f << "<path d=\"M" << cx - 7 << "," << cy - 7 << " L" << cx + 7 << "," << cy + 7
                << " M" << cx - 7 << "," << cy + 7 << " L" << cx + 7 << "," << cy - 7
                << "\" stroke=\"black\" stroke-width=\"3\"/>\n";
        }
        int ly = mt + 20 + c * 30;
        f << "<circle cx=\"" << W - mr + 25 << "\" cy=\"" << ly << "\" r=\"6\" fill=\"" << colors[c % 8]
            << "\" stroke=\"black\" stroke-width=\"0.6\"/>\n";
        f << "<text x=\"" << W - mr + 40 << "\" y=\"" << ly + 5
            << "\" font-family=\"Arial\" font-size=\"14\">Кластер " << c + 1 << ": n = " << cnt[c] << "</text>\n";
    }
    f << "<text x=\"" << W - mr + 25 << "\" y=\"" << mt + 20 + k * 30 + 5
        << "\" font-family=\"Arial\" font-size=\"12\">× — центр кластера</text>\n";
    f << "</svg>\n";
}

// ----------------------------- Режим 1: вся выборка ------------------------
const int PAIRS[6][2] = { {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3} };

string pairName(int p) {
    return "(" + to_string(PAIRS[p][0] + 1) + ", " + to_string(PAIRS[p][1] + 1) + ")";
}

string pairFile(int p) {
    return to_string(PAIRS[p][0] + 1) + "_" + to_string(PAIRS[p][1] + 1);
}

void runFullData() {
    emit("\n========== РЕЖИМ 1. Кластеризация всех " + to_string(N_OBJECTS) + " объектов ==========\n");
    emit(string("Метод: Ward, метрика: евклидова, стандартизация: ") + (USE_STANDARDIZATION ? "да" : "нет")
        + ", кластеров: " + to_string(K_CLUSTERS) + "\n");

    double sil[6];
    for (int p = 0; p < 6; ++p) {
        int fi = PAIRS[p][0], fj = PAIRS[p][1];
        PairData pd = extractPair(fi, fj);
        vector<int> lab = wardCluster(pd.work, K_CLUSTERS);
        sil[p] = silhouette(pd.work, lab, K_CLUSTERS);

        emit("\n--- Пара признаков " + pairName(p) + " ---\n");
        emit(asciiPlot(pd.raw, lab, fi, fj));
        emit(clusterLegend(pd.raw, lab, K_CLUSTERS));
        emit("  Коэффициент силуэта: " + fmt(sil[p], 4) + "\n");

        string fname = "scatter_" + pairFile(p) + ".svg";
        saveScatterSVG(fname, pd.raw, lab, K_CLUSTERS, fi, fj,
            "Признаки " + pairName(p) + ", Ward, силуэт = " + fmt(sil[p], 3));
        emit("  Рисунок сохранён: " + fname + "\n");
    }

    emit("\n--- Итог режима 1 ---\n");
    emit("  Пара      Силуэт\n");
    int bestP = 0;
    for (int p = 0; p < 6; ++p) {
        emit("  " + pairName(p) + "    " + fmt(sil[p], 4) + "\n");
        if (sil[p] > sil[bestP]) bestP = p;
    }
    emit("  Лучшая пара по силуэту: " + pairName(bestP) + "\n");
}

// ----------------------------- Режим 2: N повторов -------------------------
void runSubsampling(int runs) {
    int m = (int)lround(SUBSAMPLE_FRACTION * N_OBJECTS);
    emit("\n========== РЕЖИМ 2. " + to_string(runs) + " повторных кластеризаций ==========\n");
    emit("Каждый раз берётся случайная подвыборка из " + to_string(m) + " объектов (без возвращения), seed = "
        + to_string(SEED) + "\n");

    Stats st[6];
    vector<int> bestIdx[6];
    vector<int> bestLab[6];
    double bestSil[6];

    for (int p = 0; p < 6; ++p) {
        PairData pd = extractPair(PAIRS[p][0], PAIRS[p][1]);
        mt19937 rng(SEED + p);
        vector<int> idx(N_OBJECTS);
        iota(idx.begin(), idx.end(), 0);

        vector<double> scores;
        scores.reserve(runs);
        bestSil[p] = -2.0;
        for (int r = 0; r < runs; ++r) {
            shuffle(idx.begin(), idx.end(), rng);
            vector<int> take(idx.begin(), idx.begin() + m);
            sort(take.begin(), take.end());
            vector<P2> sub = subset(pd.work, take);
            vector<int> lab = wardCluster(sub, K_CLUSTERS);
            double s = silhouette(sub, lab, K_CLUSTERS);
            scores.push_back(s);
            if (s > bestSil[p]) { bestSil[p] = s; bestIdx[p] = take; bestLab[p] = lab; }
        }
        st[p] = calcStats(scores);
        emit("  пара " + pairName(p) + " обработана\n");
    }

    emit("\n--- Статистика коэффициента силуэта ---\n");
    emit("  Пара      Среднее      СКО        Мин     Медиана      Макс\n");
    int bestP = 0;
    for (int p = 0; p < 6; ++p) {
        ostringstream o;
        o << "  " << pairName(p) << "   " << setw(8) << fmt(st[p].mean, 4) << "  " << setw(8) << fmt(st[p].sd, 4)
            << "  " << setw(8) << fmt(st[p].mn, 4) << "  " << setw(8) << fmt(st[p].med, 4) << "  "
            << setw(8) << fmt(st[p].mx, 4) << "\n";
        emit(o.str());
        if (st[p].mean > st[bestP].mean) bestP = p;
    }
    emit("\n  Лучшая пара по среднему силуэту: " + pairName(bestP) + "\n");

    // лучший по метрике запуск для каждой пары -> SVG
    emit("\n--- Лучший по метрике запуск для каждой пары ---\n");
    for (int p = 0; p < 6; ++p) {
        PairData pd = extractPair(PAIRS[p][0], PAIRS[p][1]);
        vector<P2> sub = subset(pd.raw, bestIdx[p]);
        string fname = "best_" + pairFile(p) + "_N" + to_string(runs) + ".svg";
        saveScatterSVG(fname, sub, bestLab[p], K_CLUSTERS, PAIRS[p][0], PAIRS[p][1],
            "Признаки " + pairName(p) + ": лучший запуск из " + to_string(runs) + ", силуэт = "
            + fmt(bestSil[p], 3));
        emit("  " + pairName(p) + ": силуэт = " + fmt(bestSil[p], 4) + ", рисунок " + fname + "\n");
    }

    // график лучшего запуска для победившей пары - в консоль
    {
        PairData pd = extractPair(PAIRS[bestP][0], PAIRS[bestP][1]);
        vector<P2> sub = subset(pd.raw, bestIdx[bestP]);
        emit("\n--- Лучший запуск для пары " + pairName(bestP) + " ---\n");
        emit(asciiPlot(sub, bestLab[bestP], PAIRS[bestP][0], PAIRS[bestP][1]));
        emit(clusterLegend(sub, bestLab[bestP], K_CLUSTERS));
    }
}

// ----------------------------- Главная функция -----------------------------
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    g_log.open("protocol.txt", ios::app);

    while (true) {
        cout << "\n=============== МЕНЮ ===============\n"
            << " 1 - Кластеризация всей выборки (6 пар признаков)\n"
            << " 2 - Повторные кластеризации (статистика силуэта)\n"
            << " 0 - Выход\n"
            << "Ваш выбор: ";
        int choice;
        if (!(cin >> choice)) {
            if (cin.eof()) break;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Ошибка ввода: нужно ввести число.\n";
            continue;
        }
        if (choice == 0) break;
        if (choice == 1) {
            runFullData();
        }
        else if (choice == 2) {
            cout << "Число повторов (например, 100, 1000 или 10000): ";
            int runs;
            if (!(cin >> runs) || runs < 1 || runs > 1000000) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Ошибка: число повторов должно быть от 1 до 1000000.\n";
                continue;
            }
            runSubsampling(runs);
        }
        else {
            cout << "Нет такого пункта меню.\n";
        }
    }
    cout << "Протокол сохранён в protocol.txt\n";
    return 0;
}