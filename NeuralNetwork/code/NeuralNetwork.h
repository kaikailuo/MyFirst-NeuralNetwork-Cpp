#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

// 枚举类型：支持的激活函数
enum ActivationType {
    SIGMOID,
    TANH,
    RELU
};

// 用于前向传播结果的结构体
struct ForwardResult {
    vector<float> hidden_output;  // 隐藏层输出
    vector<float> output;         // 输出层输出
};

// 三层神经网络类定义
class NeutralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;  // 各层节点数
    vector<vector<float>> W1; // 输入层到隐藏层权重矩阵
    vector<vector<float>> W2; // 隐藏层到输出层权重矩阵
    ActivationType hidden_AT = SIGMOID; // 隐藏层激活函数类型
    ActivationType out_AT = SIGMOID;    // 输出层激活函数类型
    static bool if_update_lr;           // 是否启用学习率衰减

public:
    NeutralNetwork(int in, int hn, int on, vector<vector<float>>& w1, vector<vector<float>>& w2); // 指定初始权重
    NeutralNetwork(int in, int hn, int on, bool rand_w); // 使用随机初始化权重

    float activation(float x, ActivationType type);       // 激活函数
    float dactivation(float x, ActivationType type);      // 激活函数导数

    float loss(const vector<float>& output, const vector<float>& target);     // 损失函数（交叉熵）
    float loss_grad(float output, float target);                              // 损失函数导数

    ForwardResult forward_propagate(const vector<float>& input);              // 前向传播
    void back_propagation(const vector<float>& input, const vector<float>& target, float learningRate); // 反向传播
    void train(const vector<vector<float>>& input, const vector<vector<float>>& target, float learningRate, int epochs, ostream& out); // 训练函数

    void reset(ActivationType hidden, ActivationType out) { hidden_AT = hidden; out_AT = out; } // 重设激活函数类型
};