#pragma once

#include "causal_lm_batch.h"
#include "wikitext2_dataset.h"

#include <cstddef>

namespace ops {

class BatchProvider {
public:
    virtual ~BatchProvider() = default;

    virtual CausalLMBatch next_batch(std::size_t batch_size, bool loop) = 0;
    virtual void reset() = 0;
    virtual std::size_t num_sequences() const = 0;
};

class WikiText2BatchProvider final : public BatchProvider {
public:
    explicit WikiText2BatchProvider(WikiText2Dataset& dataset);

    CausalLMBatch next_batch(std::size_t batch_size, bool loop) override;
    void reset() override;
    std::size_t num_sequences() const override;

private:
    WikiText2Dataset& dataset_;
};

int count_valid_shifted_labels(const TensorPtr& labels, int ignore_index);
CausalLMBatch make_causal_lm_batch_from_tensors(const TensorPtr& input_ids,
                                                const TensorPtr& attention_mask,
                                                const TensorPtr& labels,
                                                int ignore_index = -100);

}  // namespace ops

