#include "batch_provider.h"

#include <stdexcept>

namespace ops {

int count_valid_shifted_labels(const TensorPtr& labels, int ignore_index) {
    if (!labels || labels->ndim() != 2 || labels->dtype() != kInt32) {
        return 0;
    }
    const auto& shape = labels->shape();
    const int64_t batch = shape[0];
    const int64_t seq = shape[1];
    const int32_t* data = labels->data<int32_t>();
    int valid = 0;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 1; s < seq; ++s) {
            if (data[b * seq + s] != ignore_index) {
                ++valid;
            }
        }
    }
    return valid;
}

CausalLMBatch make_causal_lm_batch_from_tensors(const TensorPtr& input_ids,
                                                const TensorPtr& attention_mask,
                                                const TensorPtr& labels,
                                                int ignore_index) {
    if (!input_ids && !attention_mask && !labels) {
        return CausalLMBatch{};
    }
    if (!input_ids || !attention_mask || !labels) {
        throw std::invalid_argument("make_causal_lm_batch_from_tensors requires input_ids, attention_mask, and labels");
    }
    if (input_ids->ndim() != 2 || attention_mask->ndim() != 2 || labels->ndim() != 2) {
        throw std::invalid_argument("training batch tensors must be rank-2 [batch, sequence_length]");
    }
    if (input_ids->shape() != attention_mask->shape() || input_ids->shape() != labels->shape()) {
        throw std::invalid_argument("training batch tensor shapes must match");
    }
    if (input_ids->dtype() != kInt32 || labels->dtype() != kInt32 || attention_mask->dtype() != kFloat32) {
        throw std::invalid_argument("training batch dtypes must be input_ids=int32, attention_mask=float32, labels=int32");
    }

    CausalLMBatch out;
    out.input_ids = input_ids;
    out.attention_mask = attention_mask;
    out.labels = labels;
    out.batch_size = static_cast<int>(input_ids->shape()[0]);
    out.sequence_length = static_cast<int>(input_ids->shape()[1]);
    out.valid_label_count = count_valid_shifted_labels(labels, ignore_index);
    return out;
}

WikiText2BatchProvider::WikiText2BatchProvider(WikiText2Dataset& dataset)
    : dataset_(dataset) {}

CausalLMBatch WikiText2BatchProvider::next_batch(std::size_t batch_size, bool loop) {
    Batch batch = dataset_.next_batch(batch_size, loop);
    return make_causal_lm_batch_from_tensors(batch.input_ids, batch.attention_mask, batch.labels);
}

void WikiText2BatchProvider::reset() {
    dataset_.reset_cursor();
}

std::size_t WikiText2BatchProvider::num_sequences() const {
    return dataset_.num_sequences();
}

}  // namespace ops

