#define _CRT_SECURE_NO_WARNINGS
#include"NeuralNetwork.h"

// ������� [-1,1] ����ĳ�ʼȨ��
inline float random_weight() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

// ѧϰ�ʸ��²��ԣ�ָ��˥����
inline void get_new_lr(float& lr, int epoch, float decay_rate = 0.01f) {
    lr *= exp(-decay_rate * epoch);
}

// ʹ�ø���Ȩ�س�ʼ������
NeutralNetwork::NeutralNetwork(int in, int hn, int on, vector<vector<float>>& w1, vector<vector<float>>& w2)
    : inputNodes(in), hiddenNodes(hn), outputNodes(on), W1(w1), W2(w2) {
}

// ʹ�����Ȩ�س�ʼ������
NeutralNetwork::NeutralNetwork(int in, int hn, int on, bool rand_w)
    : inputNodes(in), hiddenNodes(hn), outputNodes(on) {
    W1.resize(inputNodes, vector<float>(hiddenNodes));
    W2.resize(hiddenNodes, vector<float>(outputNodes));
    for (auto& row : W1)
        for (auto& w : row)
            w = random_weight();
    for (auto& row : W2)
        for (auto& w : row)
            w = random_weight();
}

// �����ʵ��
float NeutralNetwork::activation(float x, ActivationType type) {
    switch (type) {
    case SIGMOID: return 1.0f / (1 + exp(-x));
    case TANH: return tanh(x);
    case RELU: return max(0.0f, x);
    default: return x;
    }
}

// ���������ʵ��
float NeutralNetwork::dactivation(float x, ActivationType type) {
    switch (type) {
    case SIGMOID: {
        float s = 1.0f / (1 + exp(-x));
        return s * (1 - s);
    }
    case TANH: {
        float t = tanh(x);
        return 1 - t * t;
    }
    case RELU:
        return x > 0 ? 1.0f : 0.0f;
    default:
        return 1.0f;
    }
}

// ���㽻������ʧ
float NeutralNetwork::loss(const vector<float>& output, const vector<float>& target) {
    float total_loss = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        total_loss -= target[i] * log(output[i]) + (1 - target[i]) * log(1 - output[i]);
    }
    return total_loss;
}

// ��ʧ����������ĵ���
float NeutralNetwork::loss_grad(float output, float target) {
    return -(target / output) + (1 - target) / (1 - output);
}

// ǰ�򴫲�ʵ��
ForwardResult NeutralNetwork::forward_propagate(const vector<float>& input) {
    ForwardResult res;
    res.hidden_output.resize(hiddenNodes);
    res.output.resize(outputNodes);

    // ����� -> ���ز�
    for (int j = 0; j < hiddenNodes; ++j) {
        float sum = 0;
        for (int i = 0; i < inputNodes; ++i) {
            sum += input[i] * W1[i][j];
        }
        res.hidden_output[j] = activation(sum, hidden_AT);
    }

    // ���ز� -> �����
    for (int j = 0; j < outputNodes; ++j) {
        float sum = 0;
        for (int i = 0; i < hiddenNodes; ++i) {
            sum += res.hidden_output[i] * W2[i][j];
        }
        res.output[j] = activation(sum, out_AT);
    }

    return res;
}

// ���򴫲�ʵ��
void NeutralNetwork::back_propagation(const vector<float>& input, const vector<float>& target, float learningRate) {
    ForwardResult F_res = forward_propagate(input);
    const auto& hidden_output = F_res.hidden_output;
    const auto& output = F_res.output;

    // ������ݶ�
    vector<float> output_grad(outputNodes);
    for (int i = 0; i < outputNodes; ++i) {
        output_grad[i] = loss_grad(output[i], target[i]) * dactivation(output[i], out_AT);
    }

    // ���ز��ݶ�
    vector<float> hidden_grad(hiddenNodes);
    for (int i = 0; i < hiddenNodes; ++i) {
        float error = 0;
        for (int j = 0; j < outputNodes; ++j) {
            error += output_grad[j] * W2[i][j];
        }
        hidden_grad[i] = error * dactivation(hidden_output[i], hidden_AT);
    }

    // ���� W2 Ȩ��
    for (int i = 0; i < outputNodes; ++i) {
        for (int j = 0; j < hiddenNodes; ++j) {
            W2[j][i] -= learningRate * output_grad[i] * hidden_output[j];
        }
    }

    // ���� W1 Ȩ��
    for (int i = 0; i < hiddenNodes; ++i) {
        for (int j = 0; j < inputNodes; ++j) {
            W1[j][i] -= learningRate * hidden_grad[i] * input[j];
        }
    }
}

// ѵ������ʵ��
void NeutralNetwork::train(const vector<vector<float>>& input, const vector<vector<float>>& target, float learningRate, int epochs, ostream& out) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (if_update_lr) {
            get_new_lr(learningRate, epoch);
        }

        for (size_t i = 0; i < input.size(); ++i) {
            vector<float> output = forward_propagate(input[i]).output;
            back_propagation(input[i], target[i], learningRate);

            // ÿ 100 �� epoch ���һ��ѵ����־
            if (epoch % 100 == 0) {
                out << "Epoch: " << epoch << "\nInput: ";
                for (auto val : input[i]) out << val << " ";
                out << "\nTarget: ";
                for (auto val : target[i]) out << val << " ";
                out << "\nOutput: ";
                for (auto val : output) out << val << " ";
                out << "\n-------------------------\n";
            }
        }
    }
}
