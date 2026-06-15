#include "dpo_loss.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ops;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(float actual, float expected, float tol, const std::string& name) {
    if (std::fabs(actual - expected) > tol) {
        throw std::runtime_error(name + " mismatch actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* ad = a->data<float>();
    const float* bd = b->data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < a->numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(ad[i] - bd[i]));
    }
    return max_diff;
}

TensorPtr make_i32(const std::vector<int64_t>& shape, const std::vector<int32_t>& values) {
    return std::make_shared<Tensor>(shape, values.data(), kInt32, kCPU);
}

void test_metrics_from_logps() {
    const std::vector<float> pc{-1.0f, -2.0f};
    const std::vector<float> pr{-2.0f, -1.0f};
    const std::vector<float> rc{-1.5f, -2.5f};
    const std::vector<float> rr{-2.5f, -1.5f};
    const float beta = 0.1f;
    const DPOMetrics m = compute_dpo_metrics_from_logps(pc, pr, rc, rr, beta);

    const double margin0 = beta * ((pc[0] - pr[0]) - (rc[0] - rr[0]));
    const double margin1 = beta * ((pc[1] - pr[1]) - (rc[1] - rr[1]));
    const double expected =
        (std::log1p(std::exp(-margin0)) + std::log1p(std::exp(-margin1))) / 2.0;
    require_close(m.loss, static_cast<float>(expected), 1e-6f, "DPO metrics loss");
    require_close(m.reward_accuracy, 0.0f, 1e-6f, "DPO reward accuracy");
}

void test_dense_backward_has_policy_grads() {
    auto A = zeros({2, 2}, kFloat32, kCPU);
    auto B = zeros({2, 3}, kFloat32, kCPU);
    A->set_requires_grad(true);
    B->set_requires_grad(true);
    {
        float* a = A->data<float>();
        float* b = B->data<float>();
        a[0] = 0.4f; a[1] = -0.2f; a[2] = 0.1f; a[3] = 0.3f;
        b[0] = 0.2f; b[1] = -0.1f; b[2] = 0.5f;
        b[3] = -0.3f; b[4] = 0.4f; b[5] = 0.1f;
    }

    auto x_chosen = zeros({1, 2, 2}, kFloat32, kCPU);
    auto x_rejected = zeros({1, 2, 2}, kFloat32, kCPU);
    {
        float* xc = x_chosen->data<float>();
        float* xr = x_rejected->data<float>();
        xc[0] = 1.0f; xc[1] = 0.5f; xc[2] = -0.2f; xc[3] = 0.7f;
        xr[0] = 1.0f; xr[1] = 0.5f; xr[2] = 0.3f; xr[3] = -0.4f;
    }

    auto policy_chosen_logits = matmul(matmul(x_chosen, A), B);
    auto policy_rejected_logits = matmul(matmul(x_rejected, A), B);
    auto ref_chosen_logits = policy_chosen_logits->detach();
    auto ref_rejected_logits = policy_rejected_logits->detach();

    auto chosen_ids = make_i32({1, 2}, {0, 1});
    auto rejected_ids = make_i32({1, 2}, {0, 2});
    auto response_mask = make_i32({1, 2}, {0, 1});

    auto loss = dpo_loss(policy_chosen_logits, policy_rejected_logits,
                         ref_chosen_logits, ref_rejected_logits,
                         chosen_ids, rejected_ids,
                         response_mask, response_mask,
                         0.1f);
    loss->backward();

    require(std::isfinite(loss->data<float>()[0]), "dense DPO loss is not finite");
    require(A->grad() != nullptr, "dense DPO missing A grad");
    require(B->grad() != nullptr, "dense DPO missing B grad");
    require(max_abs_diff(A->grad(), zeros(A->shape(), kFloat32, kCPU)) > 0.0f,
            "dense DPO A grad is zero");
    require(max_abs_diff(B->grad(), zeros(B->shape(), kFloat32, kCPU)) > 0.0f,
            "dense DPO B grad is zero");
}

void test_streaming_matches_dense_loss_and_grad() {
    const int64_t Bsz = 2;
    const int64_t S = 3;
    const int64_t H = 2;
    const int64_t V = 4;
    std::vector<float> chosen_values = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.2f,
        -0.1f, 0.2f, 0.0f, 0.3f, 0.4f, -0.2f,
    };
    std::vector<float> rejected_values = {
        -0.2f, 0.1f, 0.2f, -0.3f, 0.5f, -0.4f,
        0.3f, 0.2f, -0.1f, 0.4f, -0.3f, 0.1f,
    };
    std::vector<float> weight_values = {
        0.2f, -0.1f,
        -0.3f, 0.4f,
        0.1f, 0.2f,
        -0.2f, 0.3f,
    };
    auto chosen_ids = make_i32({Bsz, S}, {0, 1, 2, 0, 2, 3});
    auto rejected_ids = make_i32({Bsz, S}, {0, 2, 1, 0, 1, 3});
    auto response_mask = make_i32({Bsz, S}, {0, 1, 1, 0, 1, 1});

    auto chosen_dense = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, chosen_values.data(), kFloat32, kCPU);
    auto rejected_dense = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, rejected_values.data(), kFloat32, kCPU);
    auto weight_dense = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
    chosen_dense->set_requires_grad(true);
    rejected_dense->set_requires_grad(true);
    weight_dense->set_requires_grad(true);
    auto dense_chosen_logits = matmul_rhs_T(chosen_dense, weight_dense);
    auto dense_rejected_logits = matmul_rhs_T(rejected_dense, weight_dense);
    std::vector<float> ref_chosen{0.0f, -0.1f};
    std::vector<float> ref_rejected{-0.2f, 0.1f};
    auto dense_loss = dpo_loss_with_ref_logps(dense_chosen_logits, dense_rejected_logits,
                                              ref_chosen, ref_rejected,
                                              chosen_ids, rejected_ids,
                                              response_mask, response_mask,
                                              0.2f);
    dense_loss->backward();

    auto chosen_stream = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, chosen_values.data(), kFloat32, kCPU);
    auto rejected_stream = std::make_shared<Tensor>(std::vector<int64_t>{Bsz, S, H}, rejected_values.data(), kFloat32, kCPU);
    auto weight_stream = std::make_shared<Tensor>(std::vector<int64_t>{V, H}, weight_values.data(), kFloat32, kCPU);
    chosen_stream->set_requires_grad(true);
    rejected_stream->set_requires_grad(true);
    weight_stream->set_requires_grad(true);
    auto streaming_loss = streaming_dpo_loss_with_ref_logps(chosen_stream, rejected_stream, weight_stream,
                                                            ref_chosen, ref_rejected,
                                                            chosen_ids, rejected_ids,
                                                            response_mask, response_mask,
                                                            0.2f);
    streaming_loss->backward();

    require_close(streaming_loss->data<float>()[0], dense_loss->data<float>()[0],
                  1e-5f, "streaming DPO loss");
    require(max_abs_diff(chosen_stream->grad(), chosen_dense->grad()) < 1e-5f,
            "streaming chosen hidden grad mismatch");
    require(max_abs_diff(rejected_stream->grad(), rejected_dense->grad()) < 1e-5f,
            "streaming rejected hidden grad mismatch");
    require(max_abs_diff(weight_stream->grad(), weight_dense->grad()) < 1e-5f,
            "streaming weight grad mismatch");
}

}  // namespace

int main() {
    test_metrics_from_logps();
    test_dense_backward_has_policy_grads();
    test_streaming_matches_dense_loss_and_grad();
    std::cout << "[PASS] DPO loss tests passed\n";
    return 0;
}
