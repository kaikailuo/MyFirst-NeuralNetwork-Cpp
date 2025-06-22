#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

// ö�����ͣ�֧�ֵļ����
enum ActivationType {
    SIGMOID,
    TANH,
    RELU
};

// ����ǰ�򴫲�����Ľṹ��
struct ForwardResult {
    vector<float> hidden_output;  // ���ز����
    vector<float> output;         // ��������
};

// �����������ඨ��
class NeutralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;  // ����ڵ���
    vector<vector<float>> W1; // ����㵽���ز�Ȩ�ؾ���
    vector<vector<float>> W2; // ���ز㵽�����Ȩ�ؾ���
    ActivationType hidden_AT = SIGMOID; // ���ز㼤�������
    ActivationType out_AT = SIGMOID;    // ����㼤�������
    static bool if_update_lr;           // �Ƿ�����ѧϰ��˥��

public:
    NeutralNetwork(int in, int hn, int on, vector<vector<float>>& w1, vector<vector<float>>& w2); // ָ����ʼȨ��
    NeutralNetwork(int in, int hn, int on, bool rand_w); // ʹ�������ʼ��Ȩ��

    float activation(float x, ActivationType type);       // �����
    float dactivation(float x, ActivationType type);      // ���������

    float loss(const vector<float>& output, const vector<float>& target);     // ��ʧ�����������أ�
    float loss_grad(float output, float target);                              // ��ʧ��������

    ForwardResult forward_propagate(const vector<float>& input);              // ǰ�򴫲�
    void back_propagation(const vector<float>& input, const vector<float>& target, float learningRate); // ���򴫲�
    void train(const vector<vector<float>>& input, const vector<vector<float>>& target, float learningRate, int epochs, ostream& out); // ѵ������

    void reset(ActivationType hidden, ActivationType out) { hidden_AT = hidden; out_AT = out; } // ���輤�������
};