package com.mobilefinetuner.sdk;

import java.io.File;
import java.util.Arrays;

/**
 * Java-facing Android SDK for MobileFineTuner.
 *
 * <p>The SDK owns one native MF session at a time: model, optional LoRA
 * adapters, and an optimizer-backed trainer. Model weights are supplied by the
 * application as files on device storage; they are not bundled in the AAR.</p>
 */
public final class MobileFineTuner implements AutoCloseable {
    static {
        System.loadLibrary("mobilefinetuner_jni");
    }

    private long nativeHandle;

    private MobileFineTuner(long nativeHandle) {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("Native MobileFineTuner session was not created");
        }
        this.nativeHandle = nativeHandle;
    }

    public static String buildInfo() {
        return nativeBuildInfo();
    }

    public static SelfTestResult selfTest(File workingDir) {
        if (workingDir == null) {
            throw new IllegalArgumentException("workingDir must not be null");
        }
        double[] result = nativeSelfTest(workingDir.getAbsolutePath());
        return new SelfTestResult((float) result[0], (int) result[1], result[2]);
    }

    public static MobileFineTuner open(String modelDir) {
        return open(modelDir, true);
    }

    public static MobileFineTuner open(String modelDir, boolean loadWeights) {
        if (modelDir == null || modelDir.isEmpty()) {
            throw new IllegalArgumentException("modelDir must not be empty");
        }
        return new MobileFineTuner(nativeCreate(modelDir, loadWeights));
    }

    public void initLora(LoraConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeInitLora(
                requireOpen(),
                config.rank,
                config.alpha,
                config.dropout,
                config.seed,
                config.targetModules
        );
    }

    public void createTrainer(TrainerConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeCreateTrainer(
                requireOpen(),
                config.learningRate,
                config.weightDecay,
                config.maxGradNorm,
                config.ignoreIndex,
                config.useStreamingLmLoss,
                config.gradientAccumulationSteps
        );
    }

    public void createDpoTrainer(DpoTrainerConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeCreateDpoTrainer(
                requireOpen(),
                config.learningRate,
                config.weightDecay,
                config.maxGradNorm,
                config.beta,
                config.useStreamingDpoLoss,
                config.gradientAccumulationSteps
        );
    }

    public TrainStepResult trainStep(
            int[] inputIds,
            float[] attentionMask,
            int[] labels,
            int batchSize,
            int sequenceLength
    ) {
        if (batchSize <= 0 || sequenceLength <= 1) {
            throw new IllegalArgumentException("batchSize must be positive and sequenceLength must be > 1");
        }
        int expected = batchSize * sequenceLength;
        requireLength(inputIds, expected, "inputIds");
        requireLength(attentionMask, expected, "attentionMask");
        requireLength(labels, expected, "labels");

        double[] result = nativeTrainStep(
                requireOpen(),
                inputIds,
                attentionMask,
                labels,
                batchSize,
                sequenceLength
        );
        return new TrainStepResult(
                (float) result[0],
                (int) result[1],
                (int) result[2],
                (float) result[3],
                (int) result[4],
                (int) result[5],
                result[6] != 0.0
        );
    }

    public TrainStepResult trainTextBatch(String[] texts, int sequenceLength) {
        return trainTextBatch(texts, sequenceLength, false);
    }

    public TrainStepResult trainTextBatch(String[] texts, int sequenceLength, boolean appendEos) {
        if (texts == null || texts.length == 0) {
            throw new IllegalArgumentException("texts must contain at least one item");
        }
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }
        String[] copied = Arrays.copyOf(texts, texts.length);
        for (String text : copied) {
            if (text == null) {
                throw new IllegalArgumentException("texts must not contain null entries");
            }
        }
        double[] result = nativeTrainTextBatch(requireOpen(), copied, sequenceLength, appendEos);
        return new TrainStepResult(
                (float) result[0],
                (int) result[1],
                (int) result[2],
                (float) result[3],
                (int) result[4],
                (int) result[5],
                result[6] != 0.0
        );
    }

    public PreferenceStepResult trainPreferenceBatch(
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength
    ) {
        return trainPreferenceBatch(
                prompts,
                chosen,
                rejected,
                refChosenLogps,
                refRejectedLogps,
                sequenceLength,
                false
        );
    }

    public PreferenceStepResult trainPreferenceBatch(
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength,
            boolean appendEosToResponse
    ) {
        int batchSize = requireSameStringBatch(prompts, chosen, rejected);
        requireLength(refChosenLogps, batchSize, "refChosenLogps");
        requireLength(refRejectedLogps, batchSize, "refRejectedLogps");
        if (sequenceLength <= 1) {
            throw new IllegalArgumentException("sequenceLength must be > 1");
        }

        double[] result = nativeTrainPreferenceBatch(
                requireOpen(),
                Arrays.copyOf(prompts, prompts.length),
                Arrays.copyOf(chosen, chosen.length),
                Arrays.copyOf(rejected, rejected.length),
                Arrays.copyOf(refChosenLogps, refChosenLogps.length),
                Arrays.copyOf(refRejectedLogps, refRejectedLogps.length),
                sequenceLength,
                appendEosToResponse
        );
        return new PreferenceStepResult(
                (float) result[0],
                (int) result[1],
                (int) result[2],
                (int) result[3],
                (float) result[4],
                (float) result[5],
                (float) result[6],
                (float) result[7],
                (float) result[8],
                (int) result[9],
                (int) result[10],
                result[11] != 0.0
        );
    }

    public int trainableTensorCount() {
        return nativeTrainableTensorCount(requireOpen());
    }

    @Override
    public void close() {
        long handle = nativeHandle;
        nativeHandle = 0L;
        if (handle != 0L) {
            nativeClose(handle);
        }
    }

    private long requireOpen() {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("MobileFineTuner session is closed");
        }
        return nativeHandle;
    }

    private static void requireLength(Object array, int expected, String name) {
        if (array == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        int length;
        if (array instanceof int[]) {
            length = ((int[]) array).length;
        } else if (array instanceof float[]) {
            length = ((float[]) array).length;
        } else {
            throw new IllegalArgumentException(name + " has unsupported array type");
        }
        if (length != expected) {
            throw new IllegalArgumentException(
                    name + " length must be " + expected + ", got " + length
            );
        }
    }

    private static int requireSameStringBatch(String[] prompts, String[] chosen, String[] rejected) {
        if (prompts == null || chosen == null || rejected == null) {
            throw new IllegalArgumentException("preference text arrays must not be null");
        }
        if (prompts.length == 0) {
            throw new IllegalArgumentException("preference batch must contain at least one pair");
        }
        if (chosen.length != prompts.length || rejected.length != prompts.length) {
            throw new IllegalArgumentException("preference text arrays must have the same length");
        }
        for (int i = 0; i < prompts.length; ++i) {
            if (prompts[i] == null || chosen[i] == null || rejected[i] == null) {
                throw new IllegalArgumentException("preference text arrays must not contain null entries");
            }
        }
        return prompts.length;
    }

    private static native String nativeBuildInfo();
    private static native double[] nativeSelfTest(String workingDir);
    private static native long nativeCreate(String modelDir, boolean loadWeights);
    private static native void nativeInitLora(
            long handle,
            int rank,
            float alpha,
            float dropout,
            long seed,
            String[] targetModules
    );
    private static native void nativeCreateTrainer(
            long handle,
            float learningRate,
            float weightDecay,
            float maxGradNorm,
            int ignoreIndex,
            boolean useStreamingLmLoss,
            int gradientAccumulationSteps
    );
    private static native void nativeCreateDpoTrainer(
            long handle,
            float learningRate,
            float weightDecay,
            float maxGradNorm,
            float beta,
            boolean useStreamingDpoLoss,
            int gradientAccumulationSteps
    );
    private static native double[] nativeTrainStep(
            long handle,
            int[] inputIds,
            float[] attentionMask,
            int[] labels,
            int batchSize,
            int sequenceLength
    );
    private static native double[] nativeTrainTextBatch(
            long handle,
            String[] texts,
            int sequenceLength,
            boolean appendEos
    );
    private static native double[] nativeTrainPreferenceBatch(
            long handle,
            String[] prompts,
            String[] chosen,
            String[] rejected,
            float[] refChosenLogps,
            float[] refRejectedLogps,
            int sequenceLength,
            boolean appendEosToResponse
    );
    private static native int nativeTrainableTensorCount(long handle);
    private static native void nativeClose(long handle);

    public static final class LoraConfig {
        public final int rank;
        public final float alpha;
        public final float dropout;
        public final long seed;
        public final String[] targetModules;

        public LoraConfig(int rank, float alpha, float dropout, long seed, String[] targetModules) {
            if (rank <= 0) {
                throw new IllegalArgumentException("rank must be positive");
            }
            if (alpha <= 0.0f) {
                throw new IllegalArgumentException("alpha must be positive");
            }
            if (dropout < 0.0f || dropout >= 1.0f) {
                throw new IllegalArgumentException("dropout must be in [0, 1)");
            }
            this.rank = rank;
            this.alpha = alpha;
            this.dropout = dropout;
            this.seed = seed;
            this.targetModules = targetModules == null
                    ? new String[0]
                    : Arrays.copyOf(targetModules, targetModules.length);
        }

        public static LoraConfig attentionQkvo() {
            return new LoraConfig(
                    8,
                    16.0f,
                    0.05f,
                    42L,
                    new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}
            );
        }
    }

    public static final class DpoTrainerConfig {
        public final float learningRate;
        public final float weightDecay;
        public final float maxGradNorm;
        public final float beta;
        public final boolean useStreamingDpoLoss;
        public final int gradientAccumulationSteps;

        public DpoTrainerConfig(float learningRate, float weightDecay, float maxGradNorm, float beta) {
            this(learningRate, weightDecay, maxGradNorm, beta, true, 1);
        }

        public DpoTrainerConfig(
                float learningRate,
                float weightDecay,
                float maxGradNorm,
                float beta,
                boolean useStreamingDpoLoss,
                int gradientAccumulationSteps
        ) {
            if (learningRate <= 0.0f) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            if (weightDecay < 0.0f) {
                throw new IllegalArgumentException("weightDecay must be non-negative");
            }
            if (maxGradNorm <= 0.0f) {
                throw new IllegalArgumentException("maxGradNorm must be positive");
            }
            if (beta <= 0.0f) {
                throw new IllegalArgumentException("beta must be positive");
            }
            if (gradientAccumulationSteps <= 0) {
                throw new IllegalArgumentException("gradientAccumulationSteps must be positive");
            }
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.maxGradNorm = maxGradNorm;
            this.beta = beta;
            this.useStreamingDpoLoss = useStreamingDpoLoss;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
        }

        public static DpoTrainerConfig defaults() {
            return new DpoTrainerConfig(2e-4f, 0.0f, 1.0f, 0.1f);
        }
    }

    public static final class TrainerConfig {
        public final float learningRate;
        public final float weightDecay;
        public final float maxGradNorm;
        public final int ignoreIndex;
        public final boolean useStreamingLmLoss;
        public final int gradientAccumulationSteps;

        public TrainerConfig(float learningRate, float weightDecay, float maxGradNorm, int ignoreIndex) {
            this(learningRate, weightDecay, maxGradNorm, ignoreIndex, true);
        }

        public TrainerConfig(
                float learningRate,
                float weightDecay,
                float maxGradNorm,
                int ignoreIndex,
                boolean useStreamingLmLoss
        ) {
            this(
                    learningRate,
                    weightDecay,
                    maxGradNorm,
                    ignoreIndex,
                    useStreamingLmLoss,
                    1
            );
        }

        public TrainerConfig(
                float learningRate,
                float weightDecay,
                float maxGradNorm,
                int ignoreIndex,
                boolean useStreamingLmLoss,
                int gradientAccumulationSteps
        ) {
            if (learningRate <= 0.0f) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            if (weightDecay < 0.0f) {
                throw new IllegalArgumentException("weightDecay must be non-negative");
            }
            if (maxGradNorm <= 0.0f) {
                throw new IllegalArgumentException("maxGradNorm must be positive");
            }
            if (gradientAccumulationSteps <= 0) {
                throw new IllegalArgumentException("gradientAccumulationSteps must be positive");
            }
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.maxGradNorm = maxGradNorm;
            this.ignoreIndex = ignoreIndex;
            this.useStreamingLmLoss = useStreamingLmLoss;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
        }

        public static TrainerConfig defaults() {
            return new TrainerConfig(2e-4f, 0.0f, 1.0f, -100);
        }
    }

    public static final class PreferenceStepResult {
        public final float loss;
        public final int trainableTensorCount;
        public final int pairCount;
        public final int validResponseTokenCount;
        public final float accumulatedLoss;
        public final float chosenReward;
        public final float rejectedReward;
        public final float rewardMargin;
        public final float rewardAccuracy;
        public final int accumulationStep;
        public final int gradientAccumulationSteps;
        public final boolean optimizerStep;

        private PreferenceStepResult(
                float loss,
                int trainableTensorCount,
                int pairCount,
                int validResponseTokenCount,
                float accumulatedLoss,
                float chosenReward,
                float rejectedReward,
                float rewardMargin,
                float rewardAccuracy,
                int accumulationStep,
                int gradientAccumulationSteps,
                boolean optimizerStep
        ) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.pairCount = pairCount;
            this.validResponseTokenCount = validResponseTokenCount;
            this.accumulatedLoss = accumulatedLoss;
            this.chosenReward = chosenReward;
            this.rejectedReward = rejectedReward;
            this.rewardMargin = rewardMargin;
            this.rewardAccuracy = rewardAccuracy;
            this.accumulationStep = accumulationStep;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
            this.optimizerStep = optimizerStep;
        }
    }

    public static final class TrainStepResult {
        public final float loss;
        public final int trainableTensorCount;
        public final int validLabelCount;
        public final float accumulatedLoss;
        public final int accumulationStep;
        public final int gradientAccumulationSteps;
        public final boolean optimizerStep;

        private TrainStepResult(
                float loss,
                int trainableTensorCount,
                int validLabelCount,
                float accumulatedLoss,
                int accumulationStep,
                int gradientAccumulationSteps,
                boolean optimizerStep
        ) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.validLabelCount = validLabelCount;
            this.accumulatedLoss = accumulatedLoss;
            this.accumulationStep = accumulationStep;
            this.gradientAccumulationSteps = gradientAccumulationSteps;
            this.optimizerStep = optimizerStep;
        }
    }

    public static final class SelfTestResult {
        public final float loss;
        public final int trainableTensorCount;
        public final double elapsedMillis;

        private SelfTestResult(float loss, int trainableTensorCount, double elapsedMillis) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.elapsedMillis = elapsedMillis;
        }
    }
}
