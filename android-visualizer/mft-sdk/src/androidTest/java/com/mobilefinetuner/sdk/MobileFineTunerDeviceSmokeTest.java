package com.mobilefinetuner.sdk;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import android.content.Context;
import android.os.Bundle;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class MobileFineTunerDeviceSmokeTest {
    @Test
    public void nativeLibraryLoads() {
        String buildInfo = MobileFineTuner.buildInfo();
        assertTrue(buildInfo.contains("MobileFineTuner Android SDK JNI"));
    }

    @Test
    public void nativeSelfTestRunsOneTrainingStep() {
        Context context = ApplicationProvider.getApplicationContext();
        MobileFineTuner.SelfTestResult result = MobileFineTuner.selfTest(context.getFilesDir());
        assertTrue(Float.isFinite(result.loss));
        assertTrue(result.trainableTensorCount > 0);
        assertTrue(result.elapsedMillis >= 0.0);
    }

    @Test
    public void textBatchTrainingUsesNativeTokenizerAndBatchBuilder() throws IOException {
        Context context = ApplicationProvider.getApplicationContext();
        File modelDir = new File(context.getFilesDir(), "mft_sdk_text_batch_gpt2");
        deleteRecursively(modelDir);
        assertTrue(modelDir.mkdirs());
        writeFile(new File(modelDir, "config.json"),
                "{\"model_type\":\"gpt2\",\"vocab_size\":50257,\"n_positions\":8,"
                        + "\"n_embd\":8,\"n_layer\":1,\"n_head\":2}");
        writeFile(new File(modelDir, "vocab.json"),
                "{\"H\":0,\"e\":1,\"<|endoftext|>\":50256}");
        writeFile(new File(modelDir, "merges.txt"), "#version: 0.2\n");
        writeFile(new File(modelDir, "tokenizer.json"), "{\"model\":{\"type\":\"BPE\"}}");

        try (MobileFineTuner mf = MobileFineTuner.open(modelDir.getAbsolutePath(), false)) {
            mf.initLora(new MobileFineTuner.LoraConfig(
                    2,
                    4.0f,
                    0.0f,
                    7L,
                    new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}
            ));
            mf.createTrainer(new MobileFineTuner.TrainerConfig(1e-3f, 0.0f, 1.0f, -100));
            MobileFineTuner.TrainStepResult result =
                    mf.trainTextBatch(new String[]{"He"}, 4);
            assertTrue(Float.isFinite(result.loss));
            assertTrue(result.trainableTensorCount > 0);
            assertTrue(result.validLabelCount > 0);
        }
    }

    @Test
    public void optionalRealAssetTextBatchSmoke() {
        Bundle args = InstrumentationRegistry.getArguments();
        int executed = 0;
        executed += runRealAssetCaseIfConfigured(args, "mft.gpt2.modelDir", "GPT-2");
        executed += runRealAssetCaseIfConfigured(args, "mft.gemma.modelDir", "Gemma");
        executed += runRealAssetCaseIfConfigured(args, "mft.qwen.modelDir", "Qwen");
        assumeTrue("No real model asset directories were provided for optional SDK smoke", executed > 0);
    }

    private static int runRealAssetCaseIfConfigured(Bundle args, String key, String familyName) {
        String modelDir = args.getString(key, "");
        if (modelDir == null || modelDir.isEmpty()) {
            return 0;
        }
        File dir = new File(modelDir);
        assertTrue(familyName + " modelDir does not exist: " + modelDir, dir.isDirectory());

        boolean loadWeights = Boolean.parseBoolean(args.getString(key + ".loadWeights", "true"));
        int sequenceLength = Integer.parseInt(args.getString(key + ".sequenceLength", "8"));
        try (MobileFineTuner mf = MobileFineTuner.open(modelDir, loadWeights)) {
            mf.initLora(MobileFineTuner.LoraConfig.attentionQkvo());
            mf.createTrainer(new MobileFineTuner.TrainerConfig(2e-4f, 0.0f, 1.0f, -100));
            MobileFineTuner.TrainStepResult result =
                    mf.trainTextBatch(new String[]{"MobileFineTuner real asset smoke."}, sequenceLength, true);
            assertTrue(familyName + " loss is not finite", Float.isFinite(result.loss));
            assertTrue(familyName + " trainable tensor count is empty", result.trainableTensorCount > 0);
            assertTrue(familyName + " valid label count is empty", result.validLabelCount > 0);
        }
        return 1;
    }

    private static void writeFile(File file, String content) throws IOException {
        try (FileWriter writer = new FileWriter(file)) {
            writer.write(content);
        }
    }

    private static void deleteRecursively(File file) {
        if (!file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        //noinspection ResultOfMethodCallIgnored
        file.delete();
    }
}
