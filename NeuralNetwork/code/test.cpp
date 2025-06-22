#define _CRT_SECURE_NO_WARNINGS
#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

// 全局配置
float learningRate = 0.01f;
int epochs = 500;
bool NeutralNetwork::if_update_lr = true; // 启用学习率衰减

int main() {
    srand(static_cast<unsigned>(time(0))); // 初始化随机种子

    // =========================
    // 初始化数据与权重
    // =========================
    vector<vector<float>> W1 = {
        {0.15, 0.25, 0.35, 0.15, 0.25, 0.35},
        {0.15, 0.25, 0.35, 0.15, 0.25, 0.35},
        {0.15, 0.25, 0.35, 0.15, 0.25, 0.35}
    };
    vector<vector<float>> W2 = {
        {0.4}, {0.45}, {0.5}, {0.4}, {0.4}, {0.4}
    };

    vector<vector<float>> train_input = {
        {0.1, 0.2, 0.3}, {0.5, 0.6, 0.7}, {0.15, 0.25, 0.1},
        {0.1, 0.1, 0.1}, {0.2, 0.3, 0.2}, {0.3, 0.4, 0.3},
        {0.5, 0.6, 0.7}, {0.6, 0.7, 0.8}, {0.7, 0.8, 0.9}
    };
    vector<vector<float>> train_target = {
        {0}, {1}, {0}, {0}, {0}, {0}, {1}, {1}, {1}
    };

    vector<float> test_input = { 0.6f, 0.8f, 0.9f };

    // =========================
    // 配置列表（权重+激活函数）
    // =========================
    struct Config {
        NeutralNetwork net;
        string filename;
        string logname;
        Config(NeutralNetwork n, string f, string l) : net(n), filename(f), logname(l) {}
    };

    vector<Config> configs = {
        {NeutralNetwork(3, 6, 1, W1, W2), "case1.txt", "train1.txt"},
        {NeutralNetwork(3, 6, 1, W1, W2), "case2.txt", "train2.txt"},
        {NeutralNetwork(3, 6, 1, W1, W2), "case3.txt", "train3.txt"},
        {NeutralNetwork(3, 6, 1, true),     "case4.txt", "train4.txt"}
    };

    vector<pair<ActivationType, ActivationType>> act_types = {
        {SIGMOID, SIGMOID},
        {TANH, SIGMOID},
        {RELU, SIGMOID},
        {SIGMOID, SIGMOID}
    };

    // =========================
    // 遍历各组配置进行训练与测试
    // =========================
    for (size_t i = 0; i < configs.size(); ++i) {
        configs[i].net.reset(act_types[i].first, act_types[i].second);

        ofstream result(configs[i].filename);
        ofstream log(configs[i].logname);

        vector<float> output_before = configs[i].net.forward_propagate(test_input).output;
        result << "Before training: ";
        for (auto val : output_before) result << val << " ";
        result << endl;

        configs[i].net.train(train_input, train_target, learningRate, epochs, log);

        vector<float> output_after = configs[i].net.forward_propagate(test_input).output;
        result << "After training: ";
        for (auto val : output_after) result << val << " ";
        result << endl;

        result.close();
        log.close();
    }

    cout << "程序结束，训练详情见 case*.txt 与 train*.txt 文件。" << endl;
    return 0;
}