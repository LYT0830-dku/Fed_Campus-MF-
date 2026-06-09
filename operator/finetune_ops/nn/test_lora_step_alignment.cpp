#include "lora_linear.h"
#include "../core/ops.h"
#include "../optim/adam.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ops;

namespace {

TensorPtr make_f32(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    auto t = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    if (t->numel() != static_cast<int64_t>(values.size())) {
        std::cerr << "value count mismatch" << std::endl;
        std::exit(1);
    }
    std::memcpy(t->data<float>(), values.data(), values.size() * sizeof(float));
    return t;
}

float max_abs_diff(const TensorPtr& tensor, const std::vector<float>& expected) {
    if (tensor->numel() != static_cast<int64_t>(expected.size())) {
        std::cerr << "shape mismatch: got numel=" << tensor->numel()
                  << " expected=" << expected.size() << std::endl;
        std::exit(1);
    }
    const float* actual = tensor->data<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        max_diff = std::max(max_diff, std::fabs(actual[i] - expected[static_cast<size_t>(i)]));
    }
    return max_diff;
}

void require_close(const std::string& name,
                   const TensorPtr& tensor,
                   const std::vector<float>& expected,
                   float tolerance) {
    const float diff = max_abs_diff(tensor, expected);
    if (diff > tolerance) {
        std::cerr << name << " mismatch: max_abs_diff=" << diff
                  << " tolerance=" << tolerance << std::endl;
        std::exit(1);
    }
}

void require_scalar_close(const std::string& name, const TensorPtr& tensor, float expected, float tolerance) {
    const float actual = tensor->data<float>()[0];
    const float diff = std::fabs(actual - expected);
    if (diff > tolerance) {
        std::cerr << name << " mismatch: actual=" << actual
                  << " expected=" << expected
                  << " diff=" << diff
                  << " tolerance=" << tolerance << std::endl;
        std::exit(1);
    }
}

}  // namespace

int main() {
    // Fixture generated with PyTorch 2.x:
    // out = x @ W + ((x @ A) @ B) * 2.0
    // loss = torch.nn.functional.mse_loss(out, target, reduction="mean")
    // torch.optim.Adam([A, B], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    auto x = make_f32({1, 2, 3}, {
        0.25f, -0.50f, 1.25f,
        -1.00f, 0.75f, 0.50f,
    });
    auto W = make_f32({3, 4}, {
        0.10f, -0.20f, 0.30f, 0.40f,
        -0.50f, 0.60f, -0.70f, 0.80f,
        0.90f, -1.00f, 1.10f, -1.20f,
    });
    auto A = make_f32({3, 2}, {
        0.02f, -0.03f,
        0.04f, 0.05f,
        -0.06f, 0.07f,
    });
    auto B = make_f32({2, 4}, {
        0.08f, -0.09f, 0.10f, -0.11f,
        0.12f, 0.13f, -0.14f, 0.15f,
    });
    auto target = make_f32({1, 2, 4}, {
        0.50f, -0.25f, 1.00f, -0.75f,
        -0.60f, 0.80f, -1.10f, 0.30f,
    });

    LoRALinear layer(W);
    layer.attach_lora(A, B, 2.0f);

    auto out = layer.forward(x);
    require_close("LoRA forward", out, {
        1.3988000154f, -1.5694999695f, 1.7666001319f, -1.7637001276f,
        -0.0036000051f, 0.1802500337f, -0.3076999784f, -0.3648500443f,
    }, 1e-6f);

    auto loss = mse_loss(out, target, "mean");
    require_scalar_close("LoRA loss", loss, 0.7467166781f, 1e-6f);

    loss->backward();
    require_close("LoRA grad_A", A->grad(), {
        -0.0805732608f, 0.0694422424f,
        0.0012383759f, -0.0016038641f,
        0.3007295132f, -0.2568235099f,
    }, 1e-6f);
    require_close("LoRA grad_B", B->grad(), {
        -0.0464099981f, 0.0655749962f, -0.0424200036f, 0.0522650033f,
        0.0552825034f, -0.0680484399f, 0.0616868846f, -0.0619503222f,
    }, 1e-6f);

    AdamConfig adam_config(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f, 1e9f, false);
    Adam optimizer(adam_config);
    optimizer.step({A, B}, {A->grad(), B->grad()});

    require_close("LoRA Adam A", A, {
        0.0699999928f, -0.0799999908f,
        -0.0099995956f, 0.0999996886f,
        -0.1099999994f, 0.1199999899f,
    }, 2e-6f);
    require_close("LoRA Adam B", B, {
        0.1299999803f, -0.1399999857f, 0.1499999911f, -0.1599999815f,
        0.0700000077f, 0.1799999774f, -0.1899999976f, 0.1999999881f,
    }, 2e-6f);

    std::cout << "[PASS] LoRA step alignment with PyTorch" << std::endl;
    return 0;
}
